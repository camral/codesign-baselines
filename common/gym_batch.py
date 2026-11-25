"""Native Codesign Gym vectorization with explicit same-step episode resets."""

import numpy as np
import torch

from .codesign_adapter import make_baseline_vec
from .embodiment_conditioning import condition_observation, conditioned_observation_space


class EvaluationAccumulator:
    """Identical one-episode-per-world accounting for held-out evaluation."""

    def __init__(self, env):
        self.env = env
        device = torch.device(env.device)
        self.returns = torch.zeros(env.num_envs, device=device)
        self.lengths = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        self.successes = torch.zeros_like(self.finished)
        self.diverged = torch.zeros_like(self.finished)
        self.frames = 0

    @property
    def complete(self):
        return bool(self.finished.all())

    def selected_complete(self, indices):
        return bool(self.finished[indices].all()) if indices else self.complete

    def add(self, reward, terminated, truncated):
        active = ~self.finished
        self.returns += reward.reshape(-1) * active
        self.lengths += active
        self.successes |= getattr(self.env, "last_success", getattr(self.env, "last_held_at_goal", torch.zeros_like(self.finished))).reshape(-1) & active
        self.diverged |= getattr(self.env, "last_diverged", torch.zeros_like(self.finished)).reshape(-1) & active
        self.finished |= (terminated | truncated).reshape(-1)
        self.frames += self.env.num_envs

    def reject(self, rejected, reward):
        rejected = rejected.reshape(-1) & ~self.finished
        self.returns[rejected] = reward
        self.finished |= rejected

    def summary(self):
        if not self.complete:
            raise RuntimeError("evaluation summary requires one completed episode per world")
        return {"return_mean": float(self.returns.mean()), "return_std": float(self.returns.std(unbiased=False)), "length_mean": float(self.lengths.float().mean()), "success_rate": float(self.successes.float().mean()), "divergence_rate": float(self.diverged.float().mean()), "episode_returns": self.returns.cpu().tolist(), "episode_lengths": self.lengths.cpu().tolist()}


class GymBatch:
    def __init__(self, designs, *, env_id="BallCatcher-Pitch", device="cuda", seed=0, env_kwargs=None,
                 control_mode="native", rejection_reward=None, env=None):
        if env is None:
            self.env, self.base = make_baseline_vec(env_id, len(designs), device=device, seed=seed,
                control_mode=control_mode, rejection_reward=rejection_reward, **(env_kwargs or {}))
        else:
            expected_designs = getattr(env, "num_designs", env.num_envs)
            if len(designs) != expected_designs:
                raise ValueError(f"received {len(designs)} designs for {expected_designs} embodiment candidates")
            self.env, self.base = env, np.asarray(env.base_embodiment).copy()
        self.designs = torch.as_tensor(np.asarray(designs), dtype=torch.float32, device=self.env.device)
        self.single_observation_space = conditioned_observation_space(self.env)
        self.observation_size = self.single_observation_space.shape[0]
        self.action_size = self.env.single_action_space.shape[0]
        self.frames = 0

    def reset(self):
        obs, rejected, _ = self.env.reset(embodiment=self.designs)
        if bool(rejected.any()):
            raise ValueError("controller batch contains rejected embodiments")
        return condition_observation(self.env, obs, self.designs)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.frames += self.env.num_envs
        done = terminated | truncated
        info = dict(info)
        info["time_outs"] = truncated & ~terminated
        if bool(done.any()):
            info["terminal_observation"] = condition_observation(self.env, obs, self.designs)
            obs, rejected, reset_info = self.env.reset(embodiment=self.designs, options={"reset_mask": done})
            if bool(rejected.any()):
                raise ValueError("episode reset rejected a previously accepted embodiment")
        return condition_observation(self.env, obs, self.designs), reward, done, info

    def close(self):
        self.env.close()
