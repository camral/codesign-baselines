"""Exercise the production Gemma gateway on a saved prompt without an environment."""

import argparse
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .gateway import GemmaGateway


def latest_prompt(path):
    selected = None
    with Path(path).open() as source:
        for line in source:
            row = json.loads(line)
            if "prompt" in row:
                selected = row
    if selected is None:
        raise ValueError(f"no saved prompt in {path}")
    return selected["prompt"], int(selected["request_templates"])


class Deadline:
    def __init__(self, seconds):
        self.deadline = time.monotonic()+seconds

    @property
    def remaining(self):
        return max(0., self.deadline-time.monotonic())

    @property
    def expired(self):
        return self.remaining <= 0


def validate(responses, dimension):
    valid = 0
    for response in responses:
        try:
            payload = json.loads(response)
            values = payload["embodiment"]
            if set(payload) == {"embodiment"} and isinstance(values, list) and len(values) == dimension and all(type(value) in (int, float) and math.isfinite(value) for value in values):
                valid += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True, help="llm_proposals.jsonl from a training run")
    parser.add_argument("--output", required=True, help="Fresh JSON result file")
    parser.add_argument("--input-tpm", type=int, default=16000)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=900.)
    args = parser.parse_args(argv)
    if args.input_tpm < 1 or args.max_output_tokens < 1 or args.timeout <= 0:
        parser.error("input-tpm, max-output-tokens and timeout must be positive")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        parser.error("set GEMINI_API_KEY")
    output = Path(args.output)
    if output.exists():
        parser.error(f"choose a fresh --output file; {output} already exists")
    messages, count = latest_prompt(args.prompt_file)
    context = json.loads(next(message["content"] for message in messages if message["role"] == "user"))
    dimension = len(context["base_embodiment"])
    gateway_args = SimpleNamespace(llm_gateway="gemma", llm_model="", llm_context_tokens=250000, llm_max_new_tokens=args.max_output_tokens, llm_gemma_input_tpm=args.input_tpm)
    gateway = GemmaGateway(gateway_args)
    started = time.monotonic()
    responses = gateway.generate(messages, count, Deadline(args.timeout), seed=0)
    result = dict(ok=len(responses) == count and validate(responses, dimension) == count, requested_designs=count, returned_designs=len(responses), valid_designs=validate(responses, dimension), dimension=dimension, elapsed_seconds=round(time.monotonic()-started, 3), metrics=gateway.metrics(), response_previews=[response[:300] for response in responses[:2]])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
