"""Command-line and JSON configuration for the CMA-ES baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


DEFAULT_CONFIG = Path(__file__).with_name("config") / "default.json"


def _read_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read CMA-ES config '{path}': {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"CMA-ES config '{path}' must contain a JSON object")
    return config


def build_parser(defaults: Optional[Dict[str, Any]] = None) -> argparse.ArgumentParser:
    """Create a parser whose command-line values override JSON defaults."""

    parser = argparse.ArgumentParser(
        description="Run a Nevergrad CMA-ES co-design smoke test."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--design-dimension", type=int)
    parser.add_argument("--policy-dimension", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--evaluation-repeats", type=int)
    parser.add_argument("--design-lower-bound", type=float)
    parser.add_argument("--design-upper-bound", type=float)
    parser.add_argument("--design-initial", type=float)
    parser.add_argument("--policy-lower-bound", type=float)
    parser.add_argument("--policy-upper-bound", type=float)
    parser.add_argument("--policy-initial", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--diagonal", action=argparse.BooleanOptionalAction)
    parser.add_argument("--elitist", action=argparse.BooleanOptionalAction)
    parser.add_argument("--maximize", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:N, or mps.",
    )
    parser.add_argument(
        "--wandb-project",
        "--wandb_project",
        dest="wandb_project",
        help="Log the run to this Weights & Biases project.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally save the smoke-test result as JSON.",
    )
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Read the selected config file, validate its keys, and parse overrides."""

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        defaults = _read_config(config_args.config)
    except ValueError as error:
        config_parser.error(str(error))

    parser = build_parser(defaults)
    destinations = {action.dest for action in parser._actions}  # noqa: SLF001
    unknown_keys = sorted(set(defaults) - destinations)
    if unknown_keys:
        parser.error(f"unknown config key(s): {', '.join(unknown_keys)}")
    return parser.parse_args(argv)
