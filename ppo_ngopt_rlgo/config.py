"""Validated, task-agnostic configuration for each optimizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class DesignVariable:
    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Design variable names cannot be empty")
        if self.lower >= self.upper:
            raise ValueError(f"Lower bound must be below upper bound for {self.name}")


@dataclass(frozen=True)
class PPOConfig:
    environment_factory: str
    environment_kwargs: Mapping[str, Any] = field(default_factory=dict)
    policy: str = "MlpPolicy"
    policy_kwargs: Mapping[str, Any] = field(default_factory=dict)
    total_timesteps: int = 10_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 0
    device: str = "auto"
    model_path: Path = Path("artifacts/ppo/model")
    evaluation_episodes: int = 10

    def __post_init__(self) -> None:
        if ":" not in self.environment_factory:
            raise ValueError("environment_factory must have 'module:callable' form")
        if min(self.total_timesteps, self.n_steps, self.batch_size) <= 0:
            raise ValueError("PPO timestep and batch settings must be positive")
        if self.n_steps % self.batch_size:
            raise ValueError("PPO n_steps must be divisible by batch_size")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("PPO gamma must be in (0, 1]")
        if self.evaluation_episodes <= 0:
            raise ValueError("PPO evaluation_episodes must be positive")


@dataclass(frozen=True)
class NGOptConfig:
    variables: Tuple[DesignVariable, ...]
    objective: str
    objective_kwargs: Mapping[str, Any] = field(default_factory=dict)
    budget: int = 100
    num_workers: int = 1
    seed: int = 0
    device: str = "auto"
    maximize: bool = False
    result_path: Path = Path("artifacts/ngopt/result.json")

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("NGOpt requires at least one design variable")
        names = [variable.name for variable in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("NGOpt design variable names must be unique")
        if ":" not in self.objective:
            raise ValueError("objective must have 'module:callable' form")
        if min(self.budget, self.num_workers) <= 0:
            raise ValueError("NGOpt budget and num_workers must be positive")


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_ppo_config(path: Path) -> PPOConfig:
    values = dict(_read_json(path))
    if "model_path" in values:
        values["model_path"] = Path(values["model_path"])
    return PPOConfig(**values)


def load_ngopt_config(path: Path) -> NGOptConfig:
    values = dict(_read_json(path))
    values["variables"] = tuple(DesignVariable(**item) for item in values["variables"])
    if "result_path" in values:
        values["result_path"] = Path(values["result_path"])
    return NGOptConfig(**values)
