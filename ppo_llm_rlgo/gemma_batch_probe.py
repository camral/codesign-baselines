"""Probe hosted Gemma design batches without starting PPO."""

import argparse
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from urllib import error, request

from .json_response import json_body

MODEL = "gemma-4-31b-it"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def load_prompt(path):
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            if "prompt" in row:
                messages = row["prompt"]
                context = json.loads(next(message["content"] for message in messages if message["role"] == "user"))
                return messages, context["task_context"]["parameters"]
    raise ValueError(f"no audited prompt in {path}")


def request_messages(messages, count):
    result = [dict(message) for message in messages]
    system = next(message for message in result if message["role"] == "system")
    system["content"] = re.sub(r"\bexactly 4\b", f"exactly {count}", system["content"])
    if (count != 4 and re.search(r"\bexactly 4\b", system["content"])) or f"exactly {count}" not in system["content"]:
        raise ValueError("saved prompt lacks the expected four-template instruction")
    return result


def payload_for(messages, max_output_tokens):
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    contents = [dict(role=message["role"], parts=[dict(text=message["content"])]) for message in messages if message["role"] != "system"]
    return dict(contents=contents, systemInstruction=dict(parts=[dict(text=system)]), generationConfig=dict(maxOutputTokens=max_output_tokens, thinkingConfig=dict(thinkingLevel="high")))


def validate_response(result, count, parameters):
    candidates = result.get("candidates") or []
    if not candidates:
        return 0, "no candidates", ""
    candidate = candidates[0]
    content = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []) if not part.get("thought"))
    try:
        parsed = json.loads(json_body(content))
        designs = parsed["embodiments"]
        if set(parsed) != {"embodiments"} or not isinstance(designs, list) or len(designs) != count:
            raise ValueError("wrong response schema or design count")
        valid = 0
        for design in designs:
            if not isinstance(design, list) or len(design) != len(parameters):
                continue
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and parameter["low"] <= value <= parameter["high"] and (not parameter["integer"] or float(value).is_integer()) for value, parameter in zip(design, parameters)):
                valid += 1
        return valid, "" if valid == count else f"{valid}/{count} valid designs", content[:300]
    except (KeyError, TypeError, ValueError):
        return 0, "invalid JSON or response schema", content[:300]


def one_request(messages, count, parameters, args, key):
    payload = payload_for(messages, args.max_output_tokens)
    body = json.dumps(payload, allow_nan=False).encode()
    call = request.Request(ENDPOINT, data=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    row = dict(mode=args.mode, requested_designs=count, estimated_input_tokens=(len(json.dumps(messages, ensure_ascii=False))+1)//2, body_bytes=len(body))
    try:
        with request.urlopen(call, timeout=args.timeout) as response:
            result = json.load(response)
        usage = result.get("usageMetadata") or {}
        valid, problem, preview = validate_response(result, count, parameters)
        row.update(http_status=200, valid_designs=valid, error=problem, response_preview=preview, prompt_tokens=usage.get("promptTokenCount"), output_tokens=usage.get("candidatesTokenCount"), thought_tokens=usage.get("thoughtsTokenCount"), cached_tokens=usage.get("cachedContentTokenCount"), finish_reason=(result.get("candidates") or [{}])[0].get("finishReason"))
        if args.save_full_response:
            row["full_response"] = "".join(part.get("text", "") for part in (result.get("candidates") or [{}])[0].get("content", {}).get("parts", []) if not part.get("thought"))
    except error.HTTPError as failure:
        row.update(http_status=failure.code, valid_designs=0, error=failure.read().decode(errors="replace")[:1000])
    except (error.URLError, TimeoutError) as failure:
        row.update(http_status=None, valid_designs=0, error=str(failure))
    row["elapsed_seconds"] = round(time.monotonic()-started, 3)
    return row


def wait_for_input_budget(attempts, tokens, limit):
    now = time.monotonic()
    while attempts and now-attempts[0][0] >= 60:
        attempts.popleft()
    if sum(value for _, value in attempts)+tokens <= limit:
        return 0.
    return max(0., 60-(now-attempts[0][0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare Gemma four-by-four and one-by-sixteen design requests")
    parser.add_argument("--prompt-file", required=True, help="llm_proposals.jsonl from a real training run")
    parser.add_argument("--mode", choices=("four-by-four", "one-by-sixteen"), required=True)
    parser.add_argument("--output", required=True, help="Fresh JSONL result file; no API key or full prompt stored")
    parser.add_argument("--input-tpm", type=int, default=16000)
    parser.add_argument("--max-output-tokens", type=int, default=32768, help="Match current training requests")
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--save-full-response", action="store_true", help="Store the complete model completion for strict offline validation")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    messages, parameters = load_prompt(args.prompt_file)
    counts = (4, 4, 4, 4) if args.mode == "four-by-four" else (16,)
    attempts = deque()
    key = os.environ.get("GEMINI_API_KEY")
    if not key and not args.dry_run:
        parser.error("set GEMINI_API_KEY before a live probe")
    output = Path(args.output)
    if output.exists() and not args.dry_run:
        parser.error(f"choose a fresh --output file; {output} already exists")
    for count in counts:
        current = request_messages(messages, count)
        estimate = (len(json.dumps(current, ensure_ascii=False))+1)//2
        if estimate > args.input_tpm:
            parser.error(f"one request estimates {estimate} input tokens, above --input-tpm={args.input_tpm}")
        if args.dry_run:
            print(json.dumps(dict(mode=args.mode, requested_designs=count, estimated_input_tokens=estimate, body_bytes=len(json.dumps(payload_for(current, args.max_output_tokens)).encode()))))
            continue
        delay = wait_for_input_budget(attempts, estimate, args.input_tpm)
        if delay:
            print(f"Waiting {delay:.1f}s for the input-token budget", flush=True)
            time.sleep(delay)
            wait_for_input_budget(attempts, 0, args.input_tpm)
        attempts.append((time.monotonic(), estimate))
        row = one_request(current, count, parameters, args, key)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a") as target:
            target.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps({key: value for key, value in row.items() if key != "full_response"}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
