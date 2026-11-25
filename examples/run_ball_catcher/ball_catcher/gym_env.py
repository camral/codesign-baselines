"""Single-instance Gymnasium adapter around ``BallCatcherParallelEnv``.

``BallCatcherParallelEnv`` is a batched, GPU-resident env (leading dim
``num_envs``) with a Gymnasium-*style* but not Gymnasium-*compatible*
``reset()``/``step()`` contract. ``ppo_ngopt_rlgo``'s Stable-Baselines3 PPO
runner instead expects a single, unbatched ``gymnasium.Env`` built from an
``environment_factory`` hook of the form ``module:callable`` (see
``ppo_ngopt_rlgo/examples.py``). This module provides that hook by wrapping a
``num_envs=1`` instance and squeezing/unsqueezing the batch dimension.

It also implements ``common.codesign_env.CoDesignEnv``: the arm's offsettable
links (and, when enabled, its glove and legs -- see ``ball_catcher_env``'s
module docstring) are its embodiment, bounded by the underlying env's own
``offset_bounds``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from common.codesign_env import CoDesignEnv

from .ball_catcher_env import BallCatcherParallelEnv


class BallCatcherGymEnv(CoDesignEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, device: str = "cpu", render_mode: Optional[str] = None,
                 is_eval_env: bool = False, **env_kwargs: Any):
        env_kwargs.pop("num_envs", None)
        self._env = BallCatcherParallelEnv(
            num_envs=1, device=device, render_mode=render_mode, **env_kwargs)
        lo, hi = self._env.offset_bounds
        embodiment_space = spaces.Box(
            low=lo, high=hi, dtype=np.float32, seed=self._env.seed)
        super().__init__(embodiment_space=embodiment_space, is_eval_env=is_eval_env)
        self.observation_space = self._env.single_observation_space
        self.action_space = self._env.single_action_space
        self.render_mode = render_mode

    def _to_batch(self, arr: Any) -> torch.Tensor:
        return torch.as_tensor(
            np.asarray(arr), dtype=torch.float32, device=self._env.device
        ).reshape(1, -1)

    def _reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
               embodiment: Any = None):
        obs = self._env.reset()
        if embodiment is not None:
            # Offsets only take effect through the reset kernel at the head of
            # a step program, so applying an embodiment costs one physics step
            # of zero control -- there is no way to deform the model without
            # it. reset_mask is still set from reset() above, so this step
            # respawns every world with the requested shape rather than
            # advancing an episode.
            zero_ctrl = torch.zeros(
                (1, self._env.act_dim()), dtype=torch.float32, device=self._env.device)
            obs, _, _, _, _ = self._env.step(zero_ctrl, embodiment=self._to_batch(embodiment))
        return obs[0].detach().cpu().numpy(), {}

    def _step(self, action, embodiment: Any = None):
        actions = self._to_batch(action)
        emb = None if embodiment is None else self._to_batch(embodiment)
        obs, reward, terminated, truncated, info = self._env.step(actions, embodiment=emb)
        step_info = {
            key: (value[0].detach().cpu().numpy() if torch.is_tensor(value) else value)
            for key, value in info.items()
        }
        return (
            obs[0].detach().cpu().numpy(),
            float(reward[0].item()),
            bool(terminated[0].item()),
            bool(truncated[0].item()),
            step_info,
        )

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_ball_catcher(device: str = "cpu", **kwargs: Any) -> BallCatcherGymEnv:
    """Factory hook consumed by ``ppo_ngopt_rlgo``'s ``environment_factory``."""
    return BallCatcherGymEnv(device=device, **kwargs)
