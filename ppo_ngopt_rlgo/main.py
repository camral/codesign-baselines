"""Entry point for independently running PPO and NGOpt."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from .args import build_parser
from .config import load_ngopt_config, load_ppo_config


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_config = {
            key: str(value) if hasattr(value, "__fspath__") else value
            for key, value in vars(args).items()
        }
        wandb_run = wandb.init(project=args.wandb_project, config=wandb_config)
    if args.component == "ppo":
        from .ppo import evaluate, train

        config = load_ppo_config(args.config)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        if args.model_path is not None:
            config = replace(config, model_path=args.model_path)
        config = replace(config, device=args.device)
        record = train(config) if args.action == "train" else evaluate(config)
    elif args.component == "ngopt":
        from .ngopt import optimize

        config = load_ngopt_config(args.config)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        if args.budget is not None:
            config = replace(config, budget=args.budget)
        config = replace(config, device=args.device)
        record = optimize(config)
    else:
        raise AssertionError(f"Unhandled component: {args.component}")
    if wandb_run is not None:
        wandb_run.log({"score": record.score, **dict(record.metrics)})
        wandb_run.finish()
    print(record.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
