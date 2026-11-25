"""SB3 transport adapter; Codesign Gym owns all simulator vectorization."""

from collections import deque

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from .gym_batch import GymBatch


class SB3GymBatch(VecEnv):
    def __init__(self, batch: GymBatch):
        self.batch = batch
        self.episode_returns = torch.zeros(batch.env.num_envs, device=batch.env.device)
        self.completed_returns = deque(maxlen=100)
        super().__init__(batch.env.num_envs, batch.single_observation_space, batch.env.single_action_space)

    def reset(self):
        observation = self.batch.reset()
        self.episode_returns.zero_()
        self._reset_seeds()
        self._reset_options()
        return observation.cpu().numpy()

    def step_async(self, actions):
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=self.batch.env.device)

    def step_wait(self):
        obs, rewards, dones, info = self.batch.step(self.actions)
        self.episode_returns += rewards.reshape(-1)
        done_indices = np.flatnonzero(dones.cpu().numpy())
        for index in done_indices:
            self.completed_returns.append(float(self.episode_returns[index]))
            self.episode_returns[index] = 0
        rows = [{} for _ in range(self.num_envs)]
        for index in np.flatnonzero(dones.cpu().numpy()):
            rows[index]["terminal_observation"] = info["terminal_observation"][index].cpu().numpy()
            rows[index]["TimeLimit.truncated"] = bool(info["time_outs"][index])
        return obs.cpu().numpy(), rewards.cpu().numpy(), dones.cpu().numpy(), rows

    @property
    def training_reward(self):
        return float(np.mean(self.completed_returns)) if self.completed_returns else None

    def get_attr(self, name, indices=None):
        return [getattr(self.batch.env, name) for _ in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        raise NotImplementedError("per-world attributes must be configured through Codesign Gym")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("use the native batch API")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        self.batch.close()
