"""Send an ordinary chat request to one NVIDIA NIM endpoint."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib import error, request

from .config import DEFAULT_MODELS


ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
ALIASES = ("nvidia-kimi", "nvidia-deepseek", "nvidia-nemo")


def payload_for(alias, prompt, max_tokens, seed):
    payload = dict(model=DEFAULT_MODELS[alias], messages=[dict(role="user", content=prompt)], temperature=1., top_p=.95, max_tokens=max_tokens, seed=seed, stream=alias == "nvidia-kimi")
    if alias == "nvidia-kimi":
        payload["reasoning_effort"] = "max"
    elif alias == "nvidia-deepseek":
        payload["chat_template_kwargs"] = dict(thinking=True, reasoning_effort="high")
    else:
        payload["chat_template_kwargs"] = dict(enable_thinking=True)
        payload["reasoning_budget"] = max_tokens
    return payload


def streamed_result(response):
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


def probe(alias, prompt, max_tokens, seed, timeout, key, *, transport=request.urlopen):
    payload = payload_for(alias, prompt, max_tokens, seed)
    accept = "text/event-stream" if payload["stream"] else "application/json"
    call = request.Request(ENDPOINT, data=json.dumps(payload, allow_nan=False).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": accept}, method="POST")
    started = time.monotonic()
    try:
        with transport(call, timeout=timeout) as response:
            result = streamed_result(response) if payload["stream"] else json.load(response)
        choice = result["choices"][0]
        message = choice["message"]
        usage = result.get("usage") or {}
        return dict(ok=True, gateway=alias, model=payload["model"], elapsed_seconds=round(time.monotonic()-started, 3), finish_reason=choice.get("finish_reason"), content=message.get("content") or "", reasoning=message.get("reasoning_content") or message.get("reasoning") or "", usage=usage)
    except error.HTTPError as failure:
        return dict(ok=False, gateway=alias, model=payload["model"], elapsed_seconds=round(time.monotonic()-started, 3), http_status=failure.code, error=failure.read().decode(errors="replace"))
    except (error.URLError, TimeoutError) as failure:
        return dict(ok=False, gateway=alias, model=payload["model"], elapsed_seconds=round(time.monotonic()-started, 3), http_status=None, error=f"{type(failure).__name__}: {failure}")
    except (IndexError, KeyError, TypeError) as failure:
        return dict(ok=False, gateway=alias, model=payload["model"], elapsed_seconds=round(time.monotonic()-started, 3), http_status=200, error=f"invalid response: {failure}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", choices=ALIASES, required=True)
    parser.add_argument("--prompt", default="Reply with exactly one sentence confirming that this NVIDIA NIM endpoint is working.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--output", help="Optional fresh JSON result file")
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_KEY")
    if not key:
        parser.error("set NIM_KEY or NVIDIA_API_KEY")
    output = Path(args.output) if args.output else None
    if output and output.exists():
        parser.error(f"choose a fresh --output file; {output} already exists")
    result = probe(args.gateway, args.prompt, args.max_tokens, args.seed, args.timeout, key)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered+"\n")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
