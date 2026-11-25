"""Model transport boundary; samplers exchange messages and text completions."""

import json
import os
from collections import deque
from pathlib import Path
import sys
import time
from typing import Protocol
from urllib import request
from urllib.error import HTTPError, URLError
import torch

from .config import resolve_arguments
from .json_response import json_body


class LLMGateway(Protocol):
    resolved_revision: str

    def count_tokens(self, messages: list[dict]) -> int: ...
    def generate(self, messages: list[dict], count: int, budget, *, seed: int) -> list[str]: ...
    def metrics(self) -> dict: ...
    def close(self) -> None: ...



_RETRIABLE_HTTP = {429, 500, 502, 503, 504}


def _remaining(budget):
    value = getattr(budget, "request_remaining", getattr(budget, "remaining", None))
    return None if value is None else max(0., float(value))


def _request_timeout(gateway, budget):
    limit = float(getattr(getattr(gateway, "args", None), "llm_api_timeout", 900.))
    remaining = _remaining(budget)
    return max(.1, min(limit, remaining)) if remaining is not None else limit


def _wait_for_retry(budget, retry):
    remaining = _remaining(budget)
    delay = min(30., 2.**min(retry, 5))
    if remaining is not None:
        delay = min(delay, remaining)
    if delay <= 0 or budget.expired:
        return False
    time.sleep(delay)
    return not budget.expired


def _request_json(gateway, call, budget, *, pace=True):
    retry = 0
    while not budget.expired:
        before_request = getattr(gateway, "before_request", None) if pace else None
        if before_request is not None and not before_request(budget):
            return None
        timeout = _request_timeout(gateway, budget)
        gateway.attempts += 1
        try:
            with gateway.transport(call, timeout=timeout) as response:
                result = json.load(response)
            gateway.requests += 1
            return result
        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            gateway.http_errors += 1
            gateway.last_error = f"HTTP {error.code}: {detail or error.reason}"
            if error.code not in _RETRIABLE_HTTP:
                raise RuntimeError(gateway.last_error) from error
        except (TimeoutError, URLError) as error:
            gateway.http_errors += 1
            gateway.last_error = f"{type(error).__name__}: {error}"
        print(f"{gateway.provider} request failed ({gateway.last_error}); retrying within wall-clock budget", file=sys.stderr, flush=True)
        if not _wait_for_retry(budget, retry):
            return None
        gateway.http_retries += 1
        retry += 1
    return None



def _read_sse(response):
    content, reasoning, usage, finish_reason = [], [], {}, None
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if thought:
                reasoning.append(thought)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    return dict(choices=[dict(message=dict(content="".join(content), reasoning_content="".join(reasoning)), finish_reason=finish_reason)], usage=usage)


def _request_sse(gateway, call, budget):
    retry = 0
    while not budget.expired:
        timeout = _request_timeout(gateway, budget)
        gateway.attempts += 1
        try:
            with gateway.transport(call, timeout=timeout) as response:
                result = _read_sse(response)
            gateway.requests += 1
            return result
        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            gateway.http_errors += 1
            gateway.last_error = f"HTTP {error.code}: {detail or error.reason}"
            if error.code not in _RETRIABLE_HTTP:
                raise RuntimeError(gateway.last_error) from error
        except (TimeoutError, URLError) as error:
            gateway.http_errors += 1
            gateway.last_error = f"{type(error).__name__}: {error}"
        print(f"{gateway.provider} request failed ({gateway.last_error}); retrying within wall-clock budget", file=sys.stderr, flush=True)
        if not _wait_for_retry(budget, retry):
            return None
        gateway.http_retries += 1
        retry += 1
    return None


class HuggingFaceGateway:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.args = args
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("the Hugging Face gateway requires CUDA")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("the Hugging Face gateway requires BF16 hardware")
        self.index = device.index if device.index is not None else torch.cuda.current_device()
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model, revision=args.llm_revision, trust_remote_code=False, local_files_only=args.llm_local_files_only)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        with torch.random.fork_rng(devices=[self.index]):
            self.model = AutoModelForCausalLM.from_pretrained(args.llm_model, revision=args.llm_revision, torch_dtype=torch.bfloat16, device_map={"": str(device)}, attn_implementation="sdpa", trust_remote_code=False, local_files_only=args.llm_local_files_only)
        self.model.eval().requires_grad_(False)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or args.llm_revision

    def count_tokens(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def generate(self, messages, count, budget, *, seed):
        from transformers import StoppingCriteria, StoppingCriteriaList
        class Deadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return budget.expired
        if self.count_tokens(messages) + self.args.llm_max_new_tokens > self.model.config.max_position_embeddings:
            raise ValueError("full template history exceeds the model context window")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt]*count, return_tensors="pt", padding=True, add_special_tokens=False).to(self.args.device)
        with torch.random.fork_rng(devices=[self.index]), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.default_generators[self.index].manual_seed(seed)
            result = self.model.generate(**inputs, do_sample=True, temperature=self.args.llm_temperature, top_p=self.args.llm_top_p, top_k=20, max_new_tokens=self.args.llm_max_new_tokens, stopping_criteria=StoppingCriteriaList([Deadline()]), pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(result[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def metrics(self):
        return {"llm/weight_bytes": sum(p.numel()*p.element_size() for p in self.model.parameters()), "gpu/peak_allocated_bytes": torch.cuda.max_memory_allocated(self.args.device), "gpu/peak_reserved_bytes": torch.cuda.max_memory_reserved(self.args.device)}

    def close(self):
        self.model = self.tokenizer = None


class CallableGateway:
    """Adapter for lightweight injected test backends."""
    resolved_revision = "injected"

    def __init__(self, backend):
        self.backend = backend

    def count_tokens(self, messages):
        return 0

    def generate(self, messages, count, budget, *, seed):
        return self.backend(messages, count, budget)

    def metrics(self):
        return {}

    def close(self):
        pass


class ManualStop(Exception):
    """End an interactive search after saving the current PPO state."""


class ManualGateway:
    """Print a transferable prompt and read one complete human-supplied batch."""
    batched_templates = True
    resolved_revision = "human"

    def __init__(self, args):
        self.args = args
        self.requests = self.wait_seconds = 0

    def count_tokens(self, messages):
        return (len(json.dumps(messages, ensure_ascii=False)) + 2) // 3

    def generate(self, messages, count, budget, *, seed):
        self.requests += 1
        context = json.loads(messages[1]["content"])
        sheet = context.get("search_state", {}).get("render_contact_sheet")
        text = f"SYSTEM INSTRUCTIONS\n{messages[0]['content']}\n\nUSER CONTEXT\n{json.dumps(context, indent=2, ensure_ascii=False)}\n"
        if sheet:
            text += f"\nVISUAL CONTEXT\nAttach the labeled contact-sheet image: {sheet}\n"
        directory = Path(self.args.output_dir) / "manual_prompts"
        directory.mkdir(exist_ok=True)
        path = directory / f"round_{context['round']:04d}.txt"
        path.write_text(text, encoding="utf-8")
        current = Path(self.args.output_dir) / "current_prompt.txt"
        current.write_text(text, encoding="utf-8")
        print(f"\n=== EMBODIMENT PROMPT: {count} templates ===", flush=True)
        print(f"Copy-paste prompt: {current.resolve()}", flush=True)
        print(f"Round archive: {path.resolve()}", flush=True)
        print(text, flush=True)
        print(f"\nPaste one JSON object with exactly {count} lists: {{\"embodiments\": [[...], ...]}}", flush=True)
        print("Multiline JSON is accepted. Type END to submit incomplete or malformed text; /quit saves and exits.", flush=True)
        started = budget.clock() if hasattr(budget, "clock") else time.monotonic()
        lines = []
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    raise ManualStop("stdin closed")
                if line.strip() == "/quit":
                    raise ManualStop("manual search ended")
                if line.strip() == "END":
                    break
                lines.append(line)
                try:
                    json.loads("".join(lines))
                except ValueError:
                    continue
                break
        finally:
            elapsed = (budget.clock() if hasattr(budget, "clock") else time.monotonic())-started
            budget.started_at += elapsed
            self.wait_seconds += elapsed
        content = "".join(lines).strip()
        try:
            payload = json.loads(content)
            entries = payload["embodiments"]
            if not isinstance(payload, dict) or set(payload) != {"embodiments"} or not isinstance(entries, list) or len(entries) != count:
                raise ValueError("wrong manual template count or schema")
            return [json.dumps({"embodiment": value}, allow_nan=False) for value in entries]
        except (ValueError, TypeError, KeyError) as error:
            print(f"Invalid batch ({error}); the entire batch will be retried.", flush=True)
            return [content]*count

    def metrics(self):
        return {"llm/manual_submissions": self.requests, "llm/manual_wait_seconds": self.wait_seconds}

    def close(self):
        pass


class OllamaGateway:
    """One Ollama Cloud completion supplies at most four templates."""
    provider = "Ollama"
    batched_templates = True
    max_templates_per_request = 4
    endpoint = "https://ollama.com/api/chat"

    def __init__(self, args, *, transport=None):
        self.args = resolve_arguments(args)
        self.resolved_revision = args.llm_model
        self.transport = transport or request.urlopen
        self.requests = self.attempts = self.http_errors = self.http_retries = self.prompt_tokens = self.completion_tokens = self.cached_tokens = 0
        self.last_error = ""
        self.last_response = {}
        self.thinking_chars = self.content_chars = 0

    def count_tokens(self, messages):
        # The API reports exact usage after generation; this only guards prompt growth.
        return (len(json.dumps(messages, ensure_ascii=False)) + 2) // 3

    def generate(self, messages, count, budget, *, seed):
        key = os.environ.get("OLLAMA_API_KEY")
        if not key:
            raise ValueError("OLLAMA_API_KEY is required for --llm-gateway ollama")
        remaining = getattr(budget, "remaining", None)
        if remaining is not None and remaining <= 0:
            return []
        payload = dict(model=self.args.llm_model, messages=messages, stream=False, think=self.args.llm_think, options=dict(temperature=self.args.llm_temperature, top_p=self.args.llm_top_p, num_predict=self.args.llm_max_new_tokens, seed=seed))
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        call = request.Request(self.endpoint, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept-Language": "en-US,en"}, method="POST")
        self.last_response = {}
        result = _request_json(self, call, budget)
        if result is None:
            return []
        self.prompt_tokens += int(result.get("prompt_eval_count") or 0)
        self.completion_tokens += int(result.get("eval_count") or 0)
        try:
            content = result["message"]["content"]
            thinking = result["message"].get("thinking") or ""
        except (KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned no chat completion") from error
        self.content_chars += len(content) if isinstance(content, str) else 0
        self.thinking_chars += len(thinking) if isinstance(thinking, str) else 0
        self.last_response = dict(requested_templates=count, seed=seed, done_reason=result.get("done_reason"), prompt_tokens=result.get("prompt_eval_count"), generated_tokens=result.get("eval_count"), thinking_chars=len(thinking) if isinstance(thinking, str) else 0, content_chars=len(content) if isinstance(content, str) else 0)
        if not isinstance(content, str) or result.get("done_reason") == "length":
            return [str(content)]*count
        try:
            parsed = json.loads(content)
            entries = parsed["embodiments"]
            if not isinstance(parsed, dict) or set(parsed) != {"embodiments"} or not isinstance(entries, list) or len(entries) != count:
                raise ValueError("wrong template count or response schema")
            return [json.dumps({"embodiment": entry}, allow_nan=False) for entry in entries]
        except (ValueError, TypeError, KeyError):
            return [content]*count

    def metrics(self):
        return {"llm/api_attempts": self.attempts, "llm/api_requests": self.requests, "llm/api_http_errors": self.http_errors, "llm/api_http_retries": self.http_retries, "llm/api_prompt_tokens": self.prompt_tokens, "llm/api_completion_tokens": self.completion_tokens, "llm/api_last_error": self.last_error, "llm/api_thinking_chars": self.thinking_chars, "llm/api_content_chars": self.content_chars, "llm/api_done_reason": self.last_response.get("done_reason") or ""}

    def close(self):
        pass



class NvidiaNIMGateway:
    """Hosted NVIDIA NIM chat completion for one full template round."""
    provider = "NVIDIA NIM"
    batched_templates = True
    max_templates_per_request = 4
    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    specifications = {
        "nvidia-kimi": dict(temperature=1., top_p=.95, reasoning_effort="max"),
        "nvidia-deepseek": dict(temperature=1., top_p=.95, chat_template_kwargs=dict(thinking=True, reasoning_effort="high")),
        "nvidia-nemo": dict(temperature=1., top_p=.95, chat_template_kwargs=dict(enable_thinking=True), reasoning_budget=16384),
    }

    def __init__(self, args, *, transport=None):
        self.alias = args.llm_gateway
        if self.alias not in self.specifications:
            raise ValueError(f"unknown NVIDIA NIM gateway {self.alias}")
        self.args = resolve_arguments(args)
        self.resolved_revision = self.args.llm_model
        self.transport = transport or request.urlopen
        self.requests = self.attempts = self.http_errors = self.http_retries = self.prompt_tokens = self.completion_tokens = self.reasoning_tokens = 0
        self.last_error = ""
        self.last_response = {}

    def count_tokens(self, messages):
        return (len(json.dumps(messages, ensure_ascii=False))+2)//3

    def generate(self, messages, count, budget, *, seed):
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise ValueError(f"NVIDIA_API_KEY is required for --llm-gateway {self.alias}")
        remaining = getattr(budget, "remaining", None)
        if remaining is not None and remaining <= 0:
            return []
        streamed = self.alias == "nvidia-kimi"
        payload = dict(model=self.args.llm_model, messages=messages, max_tokens=self.args.llm_max_new_tokens, seed=seed, stream=streamed, temperature=self.args.llm_temperature, top_p=self.args.llm_top_p)
        payload.update(self.specifications[self.alias])
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        accept = "text/event-stream" if streamed else "application/json"
        call = request.Request(self.endpoint, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": accept}, method="POST")
        result = _request_sse(self, call, budget) if streamed else _request_json(self, call, budget)
        if result is None:
            return []
        usage = result.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (IndexError, KeyError, TypeError) as error:
            raise RuntimeError(f"{self.provider} returned no chat completion") from error
        self.last_response = dict(requested_templates=count, seed=seed, finish_reason=choice.get("finish_reason"), prompt_tokens=usage.get("prompt_tokens"), generated_tokens=usage.get("completion_tokens"), reasoning_tokens=details.get("reasoning_tokens"))
        if not isinstance(content, str) or choice.get("finish_reason") == "length":
            return [str(content)]*count
        try:
            parsed = json.loads(json_body(content))
            entries = parsed["embodiments"]
            if not isinstance(parsed, dict) or set(parsed) != {"embodiments"} or not isinstance(entries, list) or len(entries) != count:
                raise ValueError("wrong template count or response schema")
            return [json.dumps({"embodiment": entry}, allow_nan=False) for entry in entries]
        except (ValueError, TypeError, KeyError):
            return [content]*count

    def metrics(self):
        return {"llm/api_attempts": self.attempts, "llm/api_requests": self.requests, "llm/api_http_errors": self.http_errors, "llm/api_http_retries": self.http_retries, "llm/api_prompt_tokens": self.prompt_tokens, "llm/api_completion_tokens": self.completion_tokens, "llm/api_reasoning_tokens": self.reasoning_tokens, "llm/api_last_error": self.last_error, "llm/api_finish_reason": self.last_response.get("finish_reason") or ""}

    def close(self):
        pass


class GemmaGateway:
    """Hosted Gemma 4 proposals through Google GenerateContent."""
    provider = "Gemma"
    batched_templates = True
    max_templates_per_request = 16
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    count_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"

    def __init__(self, args, *, transport=None):
        self.args = resolve_arguments(args)
        self.resolved_revision = self.args.llm_model
        self.transport = transport or request.urlopen
        self.requests = self.attempts = self.http_errors = self.http_retries = self.prompt_tokens = self.completion_tokens = self.cached_tokens = self.thought_tokens = 0
        self.last_error = ""
        self.input_attempts = deque()
        self.request_tokens_estimate = 0
        self.token_count_requests = 0
        self.history_rows_total = self.history_rows_dropped = 0

    def count_tokens(self, messages):
        # The API reports exact usage after generation; this only guards prompt growth.
        return (len(json.dumps(messages, ensure_ascii=False)) + 2) // 3

    def before_request(self, budget):
        limit = self.args.llm_gemma_input_tpm
        if limit <= 0 or self.request_tokens_estimate > limit:
            raise ValueError(f"Gemma request has {self.request_tokens_estimate} input tokens, exceeding --llm-gemma-input-tpm={limit}")
        while not budget.expired:
            now = time.monotonic()
            while self.input_attempts and now-self.input_attempts[0][0] >= 60:
                self.input_attempts.popleft()
            if sum(tokens for _, tokens in self.input_attempts)+self.request_tokens_estimate <= limit:
                self.input_attempts.append((now, self.request_tokens_estimate))
                return True
            delay = max(0., 60-(now-self.input_attempts[0][0]))
            remaining = _remaining(budget)
            if remaining is not None and remaining <= delay:
                if remaining > 0:
                    time.sleep(remaining)
                return False
            print(f"Gemma input-token pacing: waiting {delay:.1f}s before the next request", file=sys.stderr, flush=True)
            time.sleep(delay)
        return False

    def generate(self, messages, count, budget, *, seed):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for --llm-gateway gemma")
        remaining = getattr(budget, "remaining", None)
        if remaining is not None and remaining <= 0:
            return []
        system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        user_messages = [message for message in messages if message["role"] != "system"]
        try:
            context = json.loads(user_messages[-1]["content"])
        except (json.JSONDecodeError, TypeError):
            context = None
        rows = context.get("template_statistics_rows", []) if isinstance(context, dict) else []
        self.history_rows_total, self.history_rows_dropped = len(rows), 0
        config = dict(maxOutputTokens=self.args.llm_max_new_tokens, thinkingConfig=dict(thinkingLevel="high"))
        def request_payload(dropped):
            if context is None:
                contents = [dict(role=message["role"], parts=[dict(text=message["content"])]) for message in user_messages]
                return dict(contents=contents, systemInstruction=dict(parts=[dict(text=system)]), generationConfig=config)
            current = dict(context)
            current["template_statistics_rows"] = rows[dropped:]
            current["history_rows_total"] = len(rows)
            current["history_rows_dropped_oldest"] = dropped
            contents = [dict(role=message["role"], parts=[dict(text=json.dumps(current, separators=(",", ":"), allow_nan=False) if message is user_messages[-1] else message["content"])]) for message in user_messages]
            return dict(contents=contents, systemInstruction=dict(parts=[dict(text=system)]), generationConfig=config)
        def exact_tokens(payload):
            count_payload = dict(payload, model=f"models/{self.args.llm_model}")
            count_body = json.dumps(dict(generateContentRequest=count_payload), allow_nan=False).encode("utf-8")
            count_call = request.Request(self.count_endpoint.format(model=self.args.llm_model), data=count_body, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, method="POST")
            count_result = _request_json(self, count_call, budget, pace=False)
            if count_result is None:
                return None
            try:
                tokens = int(count_result["totalTokens"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Gemma countTokens returned no totalTokens") from error
            self.token_count_requests += 1
            return tokens
        payload = request_payload(0)
        tokens = exact_tokens(payload)
        if tokens is None:
            return []
        counted = {0: (payload, tokens)}
        limit = self.args.llm_gemma_input_tpm
        if limit > 0 and tokens > limit:
            if not rows:
                raise ValueError(f"Gemma static prompt has {tokens} input tokens, exceeding --llm-gemma-input-tpm={limit}")
            low, high = 1, len(rows)
            while low < high:
                middle = (low+high)//2
                candidate_payload = request_payload(middle)
                candidate_tokens = exact_tokens(candidate_payload)
                if candidate_tokens is None:
                    return []
                counted[middle] = (candidate_payload, candidate_tokens)
                if candidate_tokens <= limit:
                    high = middle
                else:
                    low = middle+1
            if low in counted:
                payload, tokens = counted[low]
            else:
                payload = request_payload(low)
                tokens = exact_tokens(payload)
                if tokens is None:
                    return []
            if tokens > limit:
                raise ValueError(f"Gemma static prompt has {tokens} input tokens after dropping all {len(rows)} history rows, exceeding --llm-gemma-input-tpm={limit}")
            self.history_rows_dropped = low
            print(f"Gemma prompt truncation: dropped {low} oldest of {len(rows)} history rows to fit {tokens}/{limit} input tokens", file=sys.stderr, flush=True)
        self.request_tokens_estimate = tokens
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        call = request.Request(self.endpoint.format(model=self.args.llm_model), data=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, method="POST")
        result = _request_json(self, call, budget)
        if result is None:
            return []
        usage = result.get("usageMetadata") or {}
        self.prompt_tokens += int(usage.get("promptTokenCount") or 0)
        self.completion_tokens += int(usage.get("candidatesTokenCount") or 0)
        self.cached_tokens += int(usage.get("cachedContentTokenCount") or 0)
        self.thought_tokens += int(usage.get("thoughtsTokenCount") or 0)
        try:
            candidate = result["candidates"][0]
            content = "".join(part.get("text", "") for part in candidate["content"]["parts"] if not part.get("thought"))
        except (IndexError, KeyError, TypeError) as error:
            raise RuntimeError("Gemma returned no text completion") from error
        if not content or candidate.get("finishReason") == "MAX_TOKENS":
            return [content]*count
        try:
            parsed = json.loads(json_body(content))
            entries = parsed["embodiments"]
            if not isinstance(parsed, dict) or set(parsed) != {"embodiments"} or not isinstance(entries, list) or len(entries) != count:
                raise ValueError("wrong template count or response schema")
            return [json.dumps({"embodiment": entry}, allow_nan=False) for entry in entries]
        except (ValueError, TypeError, KeyError):
            return [content]*count

    def metrics(self):
        return {"llm/api_attempts": self.attempts, "llm/api_requests": self.requests, "llm/api_http_errors": self.http_errors, "llm/api_http_retries": self.http_retries, "llm/api_token_count_requests": self.token_count_requests, "llm/api_counted_prompt_tokens": self.request_tokens_estimate, "llm/history_rows_total": self.history_rows_total, "llm/history_rows_dropped_oldest": self.history_rows_dropped, "llm/api_prompt_tokens": self.prompt_tokens, "llm/api_completion_tokens": self.completion_tokens, "llm/api_cached_tokens": self.cached_tokens, "llm/api_thought_tokens": self.thought_tokens, "llm/api_last_error": self.last_error}

    def close(self):
        pass


def make_gateway(args):
    if args.llm_gateway == "manual":
        return ManualGateway(args)
    if args.llm_gateway == "huggingface":
        return HuggingFaceGateway(args)
    if args.llm_gateway == "ollama":
        return OllamaGateway(args)
    if args.llm_gateway == "gemma":
        return GemmaGateway(args)
    if args.llm_gateway in NvidiaNIMGateway.specifications:
        return NvidiaNIMGateway(args)
    # An API adapter can be supplied as module:factory without changing search.
    from importlib import import_module
    module, factory = args.llm_gateway.rsplit(":", 1)
    return getattr(import_module(module), factory)(args)
