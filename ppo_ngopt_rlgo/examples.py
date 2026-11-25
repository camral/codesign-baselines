"""Small dependency-light hooks used only to verify each installation."""

from __future__ import annotations

from typing import Any

from .exchange import DesignCandidate


def make_cartpole(device: str = "cpu", **kwargs: Any) -> Any:
    import gymnasium as gym

    del device  # CartPole itself is CPU-only; real factories may use this value.
    return gym.make("CartPole-v1", **kwargs)


def quadratic_design_cost(
    candidate: DesignCandidate, target: float = 0.0, device: str = "cpu"
) -> float:
    """Minimize squared distance to a target for every configured variable."""
    import torch

    values = torch.tensor(list(candidate.parameters.values()), device=device)
    return float(((values - target) ** 2).sum().item())
