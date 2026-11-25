"""Stable-Baselines3 PPO runner with no Nevergrad dependency."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

import torch

from .config import PPOConfig
from .exchange import EvaluationResult, PolicyArtifact
from .loading import load_callable


def _dependencies():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "PPO dependencies are missing; install the root requirements.txt"
        ) from exc
    return PPO, evaluate_policy, Monitor


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return str(device)


def make_environment(config: PPOConfig) -> Any:
    """Build any Gymnasium-compatible environment from the configured hook."""
    factory = load_callable(config.environment_factory)
    kwargs = dict(config.environment_kwargs)
    signature = inspect.signature(factory)
    accepts_device = "device" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_device:
        kwargs.setdefault("device", _resolve_device(config.device))
    return factory(**kwargs)


def train(
    config: PPOConfig,
    environment: Optional[Any] = None,
    candidate_id: Optional[str] = None,
    callback: Optional[Any] = None,
) -> PolicyArtifact:
    """Train PPO; a future coordinator may inject a design-specific environment.

    ``callback`` is forwarded verbatim to Stable-Baselines3's
    ``model.learn(callback=...)``, so callers can pass e.g.
    ``wandb.integration.sb3.WandbCallback()`` to stream a performance curve.
    """
    PPO, _, Monitor = _dependencies()
    device = _resolve_device(config.device)
    env = Monitor(environment if environment is not None else make_environment(config))
    try:
        model = PPO(
            config.policy,
            env,
            policy_kwargs=dict(config.policy_kwargs),
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            gamma=config.gamma,
            seed=config.seed,
            device=device,
            verbose=1,
            tensorboard_log="runs/ppo_ngopt_rlgo/tb" if callback is not None else None,
        )
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        config.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(config.model_path))
    finally:
        env.close()
    return PolicyArtifact(
        path=str(_checkpoint_path(config.model_path)),
        environment_factory=config.environment_factory,
        seed=config.seed,
        training_steps=config.total_timesteps,
        candidate_id=candidate_id,
        metadata={"algorithm": "PPO", "library": "stable-baselines3", "device": device},
    )


def evaluate(config: PPOConfig, environment: Optional[Any] = None) -> EvaluationResult:
    PPO, evaluate_policy, Monitor = _dependencies()
    device = _resolve_device(config.device)
    env = Monitor(environment if environment is not None else make_environment(config))
    try:
        model = PPO.load(str(config.model_path), env=env, device=device)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=config.evaluation_episodes, deterministic=True
        )
    finally:
        env.close()
    return EvaluationResult(
        score=float(mean_reward),
        policy_path=str(_checkpoint_path(config.model_path)),
        metrics={"mean_reward": float(mean_reward), "std_reward": float(std_reward)},
    )


def _checkpoint_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")
