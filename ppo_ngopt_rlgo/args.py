"""Command-line parsing for the independent PPO and NGOpt runners."""

from __future__ import annotations

import argparse
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppo-ngopt-rlgo",
        description="Run the PPO policy or NGOpt design-search component.",
    )
    component = parser.add_subparsers(dest="component", required=True)
    ppo = component.add_parser("ppo", help="Stable-Baselines3 PPO commands")
    ppo_action = ppo.add_subparsers(dest="action", required=True)
    for action in ("train", "evaluate"):
        command = ppo_action.add_parser(action)
        command.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "ppo.json")
        command.add_argument("--seed", type=int, help="Override config seed.")
        command.add_argument("--model-path", type=Path, help="Override checkpoint path.")
        command.add_argument("--device", default="auto", help="Compute device passed to PPO.")
        command.add_argument(
            "--wandb-project", "--wandb_project", dest="wandb_project",
            help="Log the command result to this Weights & Biases project.",
        )
    ngopt = component.add_parser("ngopt", help="Nevergrad NGOpt commands")
    ngopt_action = ngopt.add_subparsers(dest="action", required=True)
    optimize = ngopt_action.add_parser("optimize")
    optimize.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "ngopt.json")
    optimize.add_argument("--seed", type=int, help="Override config seed.")
    optimize.add_argument("--budget", type=int, help="Override evaluation budget.")
    optimize.add_argument(
        "--device", default="auto",
        help="Device forwarded to the design objective: auto, cpu, cuda, cuda:N, or mps.",
    )
    optimize.add_argument(
        "--wandb-project", "--wandb_project", dest="wandb_project",
        help="Log the command result to this Weights & Biases project.",
    )
    return parser
