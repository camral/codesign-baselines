"""Gymnasium base class for co-design environments.

A co-design environment layers an embodiment (design) axis on top of the
usual Gymnasium contract: alongside the policy action, both ``reset()`` and
``step()`` accept an optional ``embodiment`` that sets or updates the shape of
the agent. ``embodiment_space`` declares its bounds, mirroring
``action_space``.

Set ``is_eval_env=True`` to freeze the shape ``reset()`` establishes: an
``embodiment`` passed to ``step()`` is then ignored, so evaluating one fixed
design cannot be reshaped mid-episode by an exploratory or accidental
embodiment action. A caller that never supplies ``embodiment`` sees the plain
Gymnasium contract.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces


class CoDesignEnv(gym.Env):

    embodiment_space: spaces.Space

    def __init__(self, embodiment_space: spaces.Space, is_eval_env: bool = False):
        super().__init__()
        self.embodiment_space = embodiment_space
        self.is_eval_env = is_eval_env

    def reset(self, *, seed: int | None = None, options: dict | None = None,
              embodiment: Any = None) -> tuple[Any, dict]:
        return self._reset(seed=seed, options=options, embodiment=embodiment)

    def step(self, action: Any, embodiment: Any = None):
        if self.is_eval_env:
            embodiment = None
        return self._step(action, embodiment)

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed: int | None, options: dict | None,
               embodiment: Any) -> tuple[Any, dict]:
        raise NotImplementedError

    def _step(self, action: Any, embodiment: Any):
        raise NotImplementedError
