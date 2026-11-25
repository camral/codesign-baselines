"""Adapt continuous Gymnasium environments to FastTD3's vector API."""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _load_factory(path: str) -> Callable[..., gym.Env]:
    try:
        module_name, attribute = path.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError("--env-factory must have the form 'module:callable'") from exc
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Environment factory {path!r} is not callable")
    return factory


def _parse_kwargs(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--env-kwargs must decode to a JSON object")
    return parsed


class GymnasiumVectorEnv:
    """FastTD3 adapter for vectorised continuous Gymnasium environments.

    A co-design environment should place all policy-visible plant, design and
    controller state in its observation. Its continuous action may represent a
    design update, a control command, or their concatenation. The environment
    owns the cadence and feasibility rules for design changes.
    """

    asymmetric_obs = False
    num_privileged_obs = 0

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int,
        device: torch.device,
        *,
        env_factory: str = "",
        env_module: str = "",
        env_kwargs: str = "{}",
        max_episode_steps: int | None = None,
        normalize_actions: bool = True,
    ) -> None:
        if env_module:
            importlib.import_module(env_module)
        factory = _load_factory(env_factory) if env_factory else None
        kwargs = _parse_kwargs(env_kwargs)
        if factory is not None:
            signature = inspect.signature(factory)
            accepts_device = "device" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if accepts_device:
                kwargs.setdefault("device", str(device))

        def make_one() -> gym.Env:
            env = factory(**kwargs) if factory else gym.make(env_name, **kwargs)
            if max_episode_steps is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
            if not isinstance(env.observation_space, gym.spaces.Box) or len(
                env.observation_space.shape
            ) != 1:
                env = gym.wrappers.FlattenObservation(env)
            return env

        vector_kwargs: dict[str, Any] = {}
        if "autoreset_mode" in inspect.signature(gym.vector.SyncVectorEnv).parameters:
            vector_kwargs["autoreset_mode"] = "SameStep"
        self.envs = gym.vector.SyncVectorEnv(
            [make_one for _ in range(num_envs)], **vector_kwargs
        )
        self.num_envs = num_envs
        self.sim_device = device
        self.normalize_actions = normalize_actions

        observation_space = self.envs.single_observation_space
        action_space = self.envs.single_action_space
        if not isinstance(observation_space, gym.spaces.Box) or len(
            observation_space.shape
        ) != 1:
            raise TypeError("FastTD3 requires observations that flatten to one Box")
        if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
            raise TypeError("FastTD3 requires a one-dimensional continuous Box action")
        if normalize_actions and (
            not np.all(np.isfinite(action_space.low))
            or not np.all(np.isfinite(action_space.high))
        ):
            raise ValueError("Action normalization requires finite action-space bounds")

        self.num_obs = int(np.prod(observation_space.shape))
        self.num_actions = int(np.prod(action_space.shape))
        self._action_low = action_space.low.astype(np.float32)
        self._action_high = action_space.high.astype(np.float32)

        inferred_horizon = getattr(self.envs.envs[0], "_max_episode_steps", None)
        if inferred_horizon is None and self.envs.envs[0].spec is not None:
            inferred_horizon = self.envs.envs[0].spec.max_episode_steps
        if inferred_horizon is None:
            raise ValueError(
                "The environment has no episode horizon; set --max-episode-steps"
            )
        self.max_episode_steps = int(inferred_horizon)
        self._initial_seeds = [seed + rank for rank in range(num_envs)]

    def _tensor(self, value: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, device=self.sim_device, dtype=dtype)

    def reset(self) -> torch.Tensor:
        observations, _ = self.envs.reset(seed=self._initial_seeds)
        self._initial_seeds = None
        return self._tensor(observations, dtype=torch.float32)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        numpy_actions = actions.detach().cpu().numpy()
        if self.normalize_actions:
            numpy_actions = self._action_low + 0.5 * (numpy_actions + 1.0) * (
                self._action_high - self._action_low
            )
        numpy_actions = np.clip(numpy_actions, self._action_low, self._action_high)

        observations, rewards, terminated, truncated, raw_infos = self.envs.step(
            numpy_actions
        )
        dones = np.logical_or(terminated, truncated)
        true_observations = np.array(observations, copy=True)

        final_observations = raw_infos.get(
            "final_obs", raw_infos.get("final_observation")
        )
        final_mask = raw_infos.get(
            "_final_obs", raw_infos.get("_final_observation", dones)
        )
        if final_observations is not None:
            for index in np.flatnonzero(final_mask):
                if final_observations[index] is not None:
                    true_observations[index] = final_observations[index]

        observations_tensor = self._tensor(observations, dtype=torch.float32)
        infos = {
            "time_outs": self._tensor(truncated, dtype=torch.bool),
            "observations": {
                "raw": {
                    "obs": self._tensor(true_observations, dtype=torch.float32)
                }
            },
            "gymnasium": raw_infos,
        }
        return (
            observations_tensor,
            self._tensor(rewards, dtype=torch.float32),
            self._tensor(dones, dtype=torch.bool),
            infos,
        )

    def close(self) -> None:
        self.envs.close()


def make_envs(args: Any, device: torch.device):
    """Construct independent training and evaluation vector environments."""

    common = {
        "env_name": args.env_name,
        "device": device,
        "env_factory": args.env_factory,
        "env_module": args.env_module,
        "env_kwargs": args.env_kwargs,
        "max_episode_steps": args.max_episode_steps,
        "normalize_actions": args.normalize_actions,
    }
    training = GymnasiumVectorEnv(
        num_envs=args.num_envs, seed=args.seed, **common
    )
    evaluation = GymnasiumVectorEnv(
        num_envs=args.num_eval_envs, seed=args.seed + 10_000, **common
    )
    return training, evaluation, evaluation
