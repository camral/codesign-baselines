"""Collect a reusable validated corpus: python -m loki.collectors --help."""

import argparse
from dataclasses import asdict
import json

from . import COLLECTORS, make_collector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=sorted(COLLECTORS), required=True)
    parser.add_argument("--output-dir", required=True, help="One corpus directory per preset")
    parser.add_argument("--count", type=int, default=500_000, help="Total accepted unique designs, including resumed rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-attempts", type=int, help="Total proposal limit, including previous attempts")
    parser.add_argument("--max-stalled-attempts", type=int, default=10_000)
    parser.add_argument("--audit-every", type=int, default=10_000,
                        help="Reset-audit the first design and every N accepted designs; 0 disables")
    parser.add_argument("--env-kwargs", default="{}", help="JSON native environment overrides; saved in the manifest")
    args = parser.parse_args(argv)
    if args.count < 1 or args.shard_size < 1 or args.max_stalled_attempts < 1:
        parser.error("count, shard-size, and max-stalled-attempts must be positive")
    if args.seed < 0 or args.audit_every < 0 or (args.max_attempts is not None and args.max_attempts < 1):
        parser.error("seed/audit-every must be nonnegative and max-attempts must be positive")
    try:
        kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as error:
        parser.error(f"--env-kwargs must be a JSON object: {error}")
    if not isinstance(kwargs, dict):
        parser.error("--env-kwargs must be a JSON object")
    try:
        with make_collector(args.env_id, seed=args.seed, device=args.device, env_kwargs=kwargs) as collector:
            result = collector.collect(args.output_dir, count=args.count, shard_size=args.shard_size,
                                       resume=args.resume, max_attempts=args.max_attempts,
                                       max_stalled_attempts=args.max_stalled_attempts, audit_every=args.audit_every,
                                       progress=lambda value: print(json.dumps(value), flush=True))
        print(json.dumps(asdict(result)), flush=True)
        return 0 if result.complete else 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
