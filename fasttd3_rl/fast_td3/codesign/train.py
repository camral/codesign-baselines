"""Run upstream FastTD3 on a general Gymnasium co-design environment."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

FASTTD3_SOURCE = Path(__file__).resolve().parents[1]
if not (FASTTD3_SOURCE / "train.py").is_file():
    raise RuntimeError("The FastTD3 training source is missing")
sys.path.insert(0, str(FASTTD3_SOURCE))

import tyro  # noqa: E402
from hyperparams import BaseArgs  # noqa: E402

try:  # Support both ``python -m`` and direct script execution.
    from .gymnasium_env import make_envs  # noqa: E402
except ImportError:
    from gymnasium_env import make_envs  # noqa: E402


@dataclass
class CoDesignArgs(BaseArgs):
    """General defaults; tune value support and capacity for each problem."""

    env_name: str = ""
    env_factory: str = ""
    env_module: str = ""
    env_kwargs: str = "{}"
    max_episode_steps: int | None = None
    normalize_actions: bool = True
    device: str = "auto"
    """Compute device: auto, cpu, cuda, cuda:N, or mps."""
    wandb_project: str | None = None
    """W&B project; supplying it also enables W&B logging."""

    project: str = "codesign-fasttd3"
    use_wandb: bool = False
    num_envs: int = 16
    num_eval_envs: int = 4
    total_timesteps: int = 100_000
    buffer_size: int = 50_000
    batch_size: int = 1_024
    learning_starts: int = 100
    compile: bool = False
    amp: bool = False
    render_interval: int = 0
    v_min: float = -100.0
    v_max: float = 100.0


_args: CoDesignArgs | None = None


def get_args() -> CoDesignArgs:
    global _args
    sys.argv = [value.replace("--wandb_project", "--wandb-project", 1) for value in sys.argv]
    _args = tyro.cli(CoDesignArgs)
    if not _args.env_name and not _args.env_factory:
        raise ValueError("Set either --env-name or --env-factory")
    if not _args.env_name:
        _args.env_name = _args.env_factory.replace(":", ".")
    if _args.num_envs < 1 or _args.num_eval_envs < 1:
        raise ValueError("--num-envs and --num-eval-envs must be positive")
    if _args.v_min >= _args.v_max:
        raise ValueError("--v-min must be smaller than --v-max")
    if _args.render_interval:
        raise ValueError(
            "Generic rendering is not supported; use --render-interval 0"
        )
    requested = _args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    _args.device = str(resolved)
    _args.cuda = resolved.type != "cpu"
    _args.device_rank = resolved.index or 0
    if _args.wandb_project:
        _args.project = _args.wandb_project
        _args.use_wandb = True
    return _args


def _make_env(*_unused, **_unused_kwargs):
    if _args is None:
        raise RuntimeError("Arguments must be parsed before environments are created")
    device = torch.device(_args.device)
    return make_envs(_args, device)


def main() -> None:
    # Upstream treats unknown IDs as MuJoCo Playground tasks. Supply the same
    # factory boundary while retaining the upstream algorithm implementation.
    shim = types.ModuleType("environments.mujoco_playground_env")
    shim.make_env = _make_env
    sys.modules[shim.__name__] = shim

    import train as upstream_train

    upstream_train.get_args = get_args
    upstream_train.main()


if __name__ == "__main__":
    main()
