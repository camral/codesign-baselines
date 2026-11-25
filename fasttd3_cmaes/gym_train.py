"""FastTD3 controller training alternating with frozen-policy CMA-ES design generations.

Each generation:
1. FastTD3 trains on the current training designs (initially the base design)
   for `--search-frequency` vector steps (or `--warmup-steps` before generation
   0 when set), or by default until every world has finished at least one episode.
2. CMA-ES proposes a population. The frozen, deterministic actor plays one
   complete episode in every world, and the averaged returns are told to
   CMA-ES. These rollouts are also added to the replay buffer, without updates.
3. The evaluator receives the best accepted design of this generation together
   with the current policy. The scored population becomes the next training
   designs, with rejected candidates replaced by that design.

The schedule mirrors PPO + NGOpt (`ppo_ngopt_rlgo/gym_train.py`). The actor and
critic see the observation plus the installed design scaled to [-1, 1], like
the other embodiment-conditioned Gym baselines. When the environment exposes a
privileged `critic` observation (e.g. `NeroGraspAlt-Tools-*`) and
`--asymmetric-critic` is enabled, the critic receives the privileged observation
plus the installed design while the actor receives the actor observation plus
the installed design. FastTD3 hyperparameters default to `BaseArgs`, and the
CMA-ES settings default to the `cmaes` baseline's, with CLI overrides for both.
"""

import argparse
import json
import math
import traceback
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from common.async_evaluation import AsyncEvaluator
from common.codesign_adapter import make_baseline_vec
from common.embodiment_conditioning import condition_observation
from common.gym_batch import EvaluationAccumulator
from common.run_log import RunLog
from common.wall_clock import WallClockBudget
from cmaes_evolutionary.gym_train import population_assignment, population_scores
from fast_td3.codesign.evaluation import policy_snapshot
from fast_td3.codesign.gym_train import REPLAY_REFERENCE_ENVS, replay_capacity
from fast_td3.codesign.inference import RolloutInference
from fast_td3.fast_td3 import Actor, Critic
from fast_td3.fast_td3_utils import EmpiricalNormalization
from fast_td3.hyperparams import BaseArgs
from .design_optimiser import build_design_optimiser, decode_design
from .normalization import PrefixEmpiricalNormalization


METHOD = "fasttd3_cmaes"
# Training resets draw fresh initial states; scoring always resets with --seed so
# that every candidate of every generation starts from the same states.
TRAINING_SEED_OFFSET = 7_000_003


def generation_best(designs, scores, rejected, *, selection_scores=None):
    """Best accepted candidate of one generation, or (None, None, None) if all were rejected.

    When `selection_scores` is provided, selects the accepted candidate with the
    highest `selection_scores` while returning its raw `scores` entry and index.
    """
    accepted = np.flatnonzero(~np.asarray(rejected, dtype=bool))
    if not len(accepted):
        return (None, None) if selection_scores is None else (None, None, None)
    rank_by = np.asarray(scores if selection_scores is None else selection_scores)
    index = int(accepted[np.argmax(rank_by[accepted])])
    if selection_scores is None:
        return np.asarray(designs[index]).copy(), float(scores[index])
    return np.asarray(designs[index]).copy(), float(scores[index]), index


def extract_pad_grids(env):
    """Return list of (col0, nrow, ncol) pad grid slices if the env exposes `_pads`, else []."""
    pads = getattr(getattr(env, "unwrapped", env), "_pads", None)
    if not isinstance(pads, dict) or not all(k in pads for k in ("col0", "nrow", "ncol")):
        return []
    return [
        (int(c0), int(nr), int(nc))
        for c0, nr, nc in zip(pads["col0"], pads["nrow"], pads["ncol"])
        if int(nr) > 0 and int(nc) > 0
    ]


def design_smoothness_penalty(designs, pad_grids):
    """Mean squared difference between 4-connected neighbor cells across all heightfield pads."""
    designs = np.asarray(designs, dtype=np.float32)
    if designs.ndim == 1:
        designs = designs[None, :]
    if not pad_grids or len(designs) == 0:
        return np.zeros(len(designs), dtype=np.float32)
    total_sq = np.zeros(len(designs), dtype=np.float64)
    total_edges = 0
    for col0, nrow, ncol in pad_grids:
        patch = designs[:, col0:col0 + nrow * ncol].reshape(-1, nrow, ncol)
        if nrow > 1:
            dr = patch[:, 1:, :] - patch[:, :-1, :]
            total_sq += np.sum(dr * dr, axis=(1, 2))
            total_edges += (nrow - 1) * ncol
        if ncol > 1:
            dc = patch[:, :, 1:] - patch[:, :, :-1]
            total_sq += np.sum(dc * dc, axis=(1, 2))
            total_edges += nrow * (ncol - 1)
    if total_edges == 0:
        return np.zeros(len(designs), dtype=np.float32)
    return (total_sq / total_edges).astype(np.float32)


def slot_designs(designs, rejected, fallback, assignment):
    """One design per simulator design slot; rejected candidates become `fallback`."""
    designs = np.asarray(designs).copy()
    designs[np.asarray(rejected, dtype=bool)] = fallback
    return designs[assignment]


def phase_complete(phase_steps, search_frequency, finished_once):
    """A training phase lasts a fixed step count, or by default one episode in every world."""
    if search_frequency is None:
        return bool(finished_once.all())
    return phase_steps >= search_frequency


def initial_condition_designs(condition, low, high, base, num_designs, *, seed=0, cma_init_mid=False, binary=False):
    """Return (snapshot_design, training_designs, snapshot_kind) for --design-condition."""
    base = np.asarray(base, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    finite = np.isfinite(low) & np.isfinite(high)
    if condition == "cmaes":
        if cma_init_mid and not binary:
            design = np.where(finite, 0.5 * (low + high), base).astype(np.float32)
        else:
            design = base.copy()
        kind = "initial_candidate"
    elif condition == "mid":
        design = np.where(finite, 0.5 * (low + high), base).astype(np.float32)
        kind = "fixed_mid"
    elif condition == "random":
        rng = np.random.default_rng(int(seed))
        sampled = rng.uniform(np.where(finite, low, 0.0), np.where(finite, high, 1.0)).astype(np.float32)
        design = np.where(finite, sampled, base).astype(np.float32)
        kind = "fixed_random"
    else:
        raise ValueError(f"Unsupported design_condition: {condition!r}")
    return design, np.repeat(design[None], num_designs, axis=0), kind


def mix_exploration_designs(designs, frac, low, high, base, rng, *, binary=False, gaussian=False,
                            max_unique=None, gaussian_scale=1.0):
    """Replace `frac` of training design slots with fresh random designs in [low, high].

    When `gaussian=True` (used for MeshSpace environments), samples clipped Gaussian
    perturbations around `base` with `sigma = (high - low) / 6` (matching CMA-ES)
    instead of uniform hypercube corners that produce extreme self-intersecting meshes.
    When `gaussian_scale > 1.0`, draws a per-design radial multiplier in
    `[0.5, gaussian_scale]` so exploration spans mild to bold deformations without
    concentrating on a single thin shell.
    When `max_unique` is set, draws at most `max_unique` unique designs and tiles them
    across the replaced slots so large `num_envs` does not multiply convex decomposition work.
    """
    out = np.asarray(designs, dtype=np.float32).copy()
    if frac <= 0.0 or len(out) <= 1:
        return out
    count = min(len(out) - 1, max(1, int(round(len(out) * float(frac)))))
    n_unique = count if max_unique is None else min(count, max(1, int(max_unique)))
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    finite = np.isfinite(low) & np.isfinite(high)
    if binary:
        unique = rng.integers(0, 2, size=(n_unique, len(base))).astype(np.float32)
    elif gaussian:
        lo = np.where(finite, low, 0.0)
        hi = np.where(finite, high, 1.0)
        sigma = (hi - lo) / 6.0
        g_scale = float(gaussian_scale)
        if g_scale > 1.0:
            mult = rng.uniform(0.5, g_scale, size=(n_unique, 1))
        else:
            mult = np.full((n_unique, 1), g_scale, dtype=np.float64)
        unique = np.clip(
            base[None, :] + rng.normal(0.0, 1.0, size=(n_unique, len(base))) * (mult * sigma[None, :]),
            lo, hi,
        ).astype(np.float32)
        unique = np.where(finite[None, :], unique, base[None, :]).astype(np.float32)
    else:
        lo = np.where(finite, low, 0.0)
        hi = np.where(finite, high, 1.0)
        unique = rng.uniform(lo, hi, size=(n_unique, len(base))).astype(np.float32)
        unique = np.where(finite[None, :], unique, base[None, :]).astype(np.float32)
    sampled = unique[np.arange(count) % n_unique] if n_unique < count else unique
    slots = rng.choice(len(out), size=count, replace=False)
    out[slots] = sampled
    return out


class CorrelatedExploration:
    """Per-env stationary AR(1) Gaussian exploration noise with rho_i ~ U(0, max_rho) per episode.

    When `burst_frac > 0`, that fraction of episodes runs low-noise execution (`std_min`)
    until a random mid-episode switch step `t_switch ~ U(0.15*H, 0.70*H)` and then switches
    to a fresh high-persistence AR(1) burst applied to a random subset of action dimensions
    (`dim_prob`, guaranteeing >= 1 active dimension). When `override=True`, masked dimensions
    are zero-centered (`eps * burst_scale`) during the burst window instead of added onto a
    potentially tanh-saturated deterministic action, enabling escape from boundary plateaus.
    """

    def __init__(self, num_envs, action_dim, max_rho, device, *,
                 burst_frac=0.0, dim_prob=1.0, override=False, std_min=0.02, std_max=0.8):
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.max_rho = float(max_rho)
        self.burst_frac = float(burst_frac)
        self.dim_prob = float(dim_prob)
        self.override = bool(override)
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        self.device = torch.device(device)
        self.rho = torch.zeros((self.num_envs, 1), device=self.device)
        self.eps = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self.step_count = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
        self.is_burst = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        self.t_switch = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
        self.t_end = torch.zeros((self.num_envs, 1), dtype=torch.long, device=self.device)
        self.burst_scale = torch.full((self.num_envs, 1), self.std_max, device=self.device)
        self.dim_mask = torch.ones((self.num_envs, self.action_dim), device=self.device)
        self.ep_horizon = 80

    @torch.no_grad()
    def _step_state(self, dones):
        dones_mask = dones.reshape(self.num_envs, 1).bool()
        if self.burst_frac <= 0.0:
            new_rho = torch.rand((self.num_envs, 1), device=self.device) * self.max_rho
            self.rho.copy_(torch.where(dones_mask, new_rho, self.rho))
            decay = torch.where(dones_mask, torch.zeros_like(self.rho), self.rho)
            innovation_scale = torch.where(
                dones_mask,
                torch.ones_like(self.rho),
                torch.sqrt((1.0 - self.rho * self.rho).clamp_min(0.0)),
            )
            self.eps.mul_(decay).add_(torch.randn_like(self.eps) * innovation_scale)
            return None, None

        if bool(dones_mask.any()):
            finished_steps = self.step_count[dones_mask]
            max_fin = int(finished_steps.max().item())
            if max_fin >= 20:
                self.ep_horizon = max(20, min(2000, max_fin))
            self.step_count.masked_fill_(dones_mask, 0)
            new_burst = torch.rand((self.num_envs, 1), device=self.device) < self.burst_frac
            self.is_burst.copy_(torch.where(dones_mask, new_burst, self.is_burst))
            lo = max(1, int(0.15 * self.ep_horizon))
            hi = max(lo + 1, int(0.65 * self.ep_horizon))
            new_tsw = torch.randint(lo, hi, (self.num_envs, 1), device=self.device)
            dur_lo = max(5, int(0.15 * self.ep_horizon))
            dur_hi = max(dur_lo + 1, int(0.75 * self.ep_horizon))
            new_tend = new_tsw + torch.randint(dur_lo, dur_hi, (self.num_envs, 1), device=self.device)
            self.t_switch.copy_(torch.where(dones_mask, torch.where(new_burst, new_tsw, torch.zeros_like(new_tsw)), self.t_switch))
            self.t_end.copy_(torch.where(dones_mask, torch.where(new_burst, new_tend, torch.zeros_like(new_tend)), self.t_end))
            rho_lo = max(0.5 * self.max_rho, self.max_rho - 0.15)
            base_rho = torch.rand((self.num_envs, 1), device=self.device) * self.max_rho
            burst_rho = rho_lo + torch.rand((self.num_envs, 1), device=self.device) * (self.max_rho - rho_lo)
            new_rho = torch.where(new_burst, burst_rho, base_rho)
            self.rho.copy_(torch.where(dones_mask, new_rho, self.rho))
            s_lo = max(self.std_min, 0.4 * self.std_max)
            new_bscale = s_lo + torch.rand((self.num_envs, 1), device=self.device) * max(self.std_max - s_lo, 0.0)
            self.burst_scale.copy_(torch.where(dones_mask, new_bscale, self.burst_scale))
            if self.dim_prob < 1.0:
                mask = (torch.rand((self.num_envs, self.action_dim), device=self.device) < self.dim_prob).float()
                empty = mask.sum(dim=1) == 0
                if bool(empty.any()):
                    rand_dim = torch.randint(0, self.action_dim, (int(empty.sum().item()),), device=self.device)
                    mask[empty, rand_dim] = 1.0
                new_mask = torch.where(new_burst, mask, torch.ones_like(mask))
                self.dim_mask.copy_(torch.where(dones_mask, new_mask, self.dim_mask))

        burst_now = self.is_burst & (self.step_count == self.t_switch)
        reset_ar = dones_mask | burst_now
        decay = torch.where(reset_ar, torch.zeros_like(self.rho), self.rho)
        innovation_scale = torch.where(
            reset_ar,
            torch.ones_like(self.rho),
            torch.sqrt((1.0 - self.rho * self.rho).clamp_min(0.0)),
        )
        self.eps.mul_(decay).add_(torch.randn_like(self.eps) * innovation_scale)
        in_burst = self.is_burst & (self.step_count >= self.t_switch) & (self.step_count < self.t_end)
        low_noise = self.is_burst & (~in_burst)
        self.step_count.add_(1)
        return in_burst, low_noise

    @torch.no_grad()
    def sample(self, dones, noise_scales=None):
        in_burst, low_noise = self._step_state(dones)
        if in_burst is None:
            return self.eps if noise_scales is None else self.eps * noise_scales
        if noise_scales is None:
            return self.eps * torch.where(in_burst, self.dim_mask, torch.ones_like(self.dim_mask))
        eff_scale = torch.where(low_noise, torch.full_like(noise_scales, self.std_min),
                                torch.where(in_burst, self.burst_scale, noise_scales))
        eff_mask = torch.where(in_burst, self.dim_mask, torch.ones_like(self.dim_mask))
        return self.eps * eff_scale * eff_mask

    @torch.no_grad()
    def apply(self, det_action, dones, noise_scales):
        """Combine deterministic policy action with correlated exploration (additive or override burst)."""
        in_burst, low_noise = self._step_state(dones)
        if in_burst is None:
            return det_action + self.eps * noise_scales
        eff_scale = torch.where(low_noise, torch.full_like(noise_scales, self.std_min),
                                torch.where(in_burst, self.burst_scale, noise_scales))
        if self.override:
            base_keep = torch.where(in_burst, 1.0 - self.dim_mask, torch.ones_like(self.dim_mask))
            noise_keep = torch.where(in_burst, self.dim_mask, torch.ones_like(self.dim_mask))
            return det_action * base_keep + self.eps * eff_scale * noise_keep
        eff_mask = torch.where(in_burst, self.dim_mask, torch.ones_like(self.dim_mask))
        return det_action + self.eps * eff_scale * eff_mask


def fit_replay_capacity(requested, state_dim, action_dim, num_envs, max_bytes, *, critic_state_dim=None):
    """Cap storage without reducing the simulator batch or training batch size."""
    critic_floats = 0 if critic_state_dim is None else 2 * critic_state_dim
    bytes_per_transition = 4 * (2 * state_dim + critic_floats + action_dim + 3)
    capacity = min(requested, int(max_bytes) // bytes_per_transition)
    if capacity < num_envs:
        required = num_envs * bytes_per_transition / 2**30
        raise ValueError(f"replay memory budget cannot hold one simulator batch ({required:.3f} GiB)")
    return capacity


class AsymmetricReplay:
    """Flat GPU circular replay buffer with optional separate privileged critic states."""

    def __init__(self, capacity, state_dim, action_dim, device, *, critic_state_dim=None):
        self.capacity, self.count, self.position = capacity, 0, 0
        self.asymmetric = critic_state_dim is not None
        fields = [
            ("state", state_dim),
            ("action", action_dim),
            ("next", state_dim),
            ("reward", 1),
            ("bootstrap", 1),
            ("discount", 1),
        ]
        if self.asymmetric:
            fields.extend([("critic_state", critic_state_dim), ("next_critic", critic_state_dim)])
        self.data = {key: torch.empty((capacity, width), device=device) for key, width in fields}

    def add(self, state, action, reward, next_state, bootstrap, discount, *, critic_state=None, next_critic=None):
        count = len(state)
        if count == 0:
            return
        if count > self.capacity:
            raise ValueError("replay capacity must hold a complete insertion batch")
        indices = (torch.arange(count, device=state.device) + self.position) % self.capacity
        values = [state, action, next_state, reward, bootstrap, discount]
        keys = ["state", "action", "next", "reward", "bootstrap", "discount"]
        if self.asymmetric:
            if critic_state is None or next_critic is None:
                raise ValueError("asymmetric replay requires critic_state and next_critic")
            keys.extend(["critic_state", "next_critic"])
            values.extend([critic_state, next_critic])
        for key, value in zip(keys, values):
            self.data[key][indices] = value.reshape(count, -1)
        self.position = (self.position + count) % self.capacity
        self.count = min(self.capacity, self.count + count)

    def sample(self, size):
        index = torch.randint(self.count, (size,), device=self.data["state"].device)
        return {key: value[index] for key, value in self.data.items()}


class NStepAccumulator:
    """Vectorized per-world n-step return accumulator feeding an `AsymmetricReplay`.

    Supports arbitrary per-step `active` masks (for frozen-policy scoring episodes
    where worlds finish at different steps) and flushes cleanly on episode end or
    phase boundaries so transitions never cross episodes or designs.
    """

    def __init__(self, replay: AsymmetricReplay, num_envs: int, state_dim: int, action_dim: int,
                 n_steps: int, gamma: float, device, *, critic_state_dim=None):
        self.replay = replay
        self.num_envs = num_envs
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.asymmetric = critic_state_dim is not None
        if self.n_steps > 1:
            n = self.n_steps
            self.state = torch.empty((n, num_envs, state_dim), device=self.device)
            self.next_state = torch.empty((n, num_envs, state_dim), device=self.device)
            self.action = torch.empty((n, num_envs, action_dim), device=self.device)
            self.reward = torch.empty((n, num_envs), device=self.device)
            self.terminated = torch.empty((n, num_envs), dtype=torch.bool, device=self.device)
            if self.asymmetric:
                self.critic_state = torch.empty((n, num_envs, critic_state_dim), device=self.device)
                self.next_critic = torch.empty((n, num_envs, critic_state_dim), device=self.device)
            self.head = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
            self._all_env_ids = torch.arange(num_envs, dtype=torch.long, device=self.device)
            self._uniform_count: int | None = 0
            self.gammas = torch.pow(
                torch.tensor(self.gamma, dtype=torch.float32, device=self.device),
                torch.arange(n, dtype=torch.float32, device=self.device),
            )

    def _emit(self, env_ids: torch.Tensor, *, uniform_k: int | None = None):
        """Emit the oldest pending transition for each lane in `env_ids` using its current `count`."""
        if env_ids.numel() == 0:
            return
        h = self.head[env_ids]  # (m,)
        if uniform_k is not None:
            max_k = int(uniform_k)
            offsets = torch.arange(max_k, device=self.device).unsqueeze(0)
            slots = (h.unsqueeze(1) + offsets) % self.n_steps
            rews = self.reward[slots, env_ids.unsqueeze(1)]
            returns = (rews * self.gammas[:max_k].unsqueeze(0)).sum(dim=1)
            last = (h + max_k - 1) % self.n_steps
            bootstrap = (~self.terminated[last, env_ids]).float()
            discount = torch.full_like(returns, self.gamma ** max_k)
        else:
            k = self.count[env_ids]  # (m,) in 1..n_steps
            max_k = int(k.max().item())
            offsets = torch.arange(max_k, device=self.device).unsqueeze(0)  # (1, max_k)
            slots = (h.unsqueeze(1) + offsets) % self.n_steps  # (m, max_k)
            valid = offsets < k.unsqueeze(1)  # (m, max_k)
            rews = self.reward[slots, env_ids.unsqueeze(1)]
            returns = (rews * self.gammas[:max_k].unsqueeze(0) * valid.float()).sum(dim=1)
            last = (h + k - 1) % self.n_steps
            bootstrap = (~self.terminated[last, env_ids]).float()
            discount = torch.pow(torch.full_like(returns, self.gamma), k.float())
        kwargs = {}
        if self.asymmetric:
            kwargs["critic_state"] = self.critic_state[h, env_ids]
            kwargs["next_critic"] = self.next_critic[last, env_ids]
        self.replay.add(
            self.state[h, env_ids],
            self.action[h, env_ids],
            returns,
            self.next_state[last, env_ids],
            bootstrap,
            discount,
            **kwargs,
        )
        self.head[env_ids] = (h + 1) % self.n_steps
        self.count[env_ids] -= 1

    def push(self, state, action, reward, next_state, terminated, truncated, *,
             critic_state=None, next_critic=None, active=None, any_done: bool | None = None):
        reward = reward.reshape(-1)
        terminated = terminated.reshape(-1).bool()
        truncated = truncated.reshape(-1).bool()
        if self.n_steps == 1:
            if active is not None:
                active = active.reshape(-1).bool()
                if not bool(active.any()):
                    return
                state, action, reward = state[active], action[active], reward[active]
                next_state, terminated = next_state[active], terminated[active]
                if self.asymmetric:
                    critic_state, next_critic = critic_state[active], next_critic[active]
            kwargs = {"critic_state": critic_state, "next_critic": next_critic} if self.asymmetric else {}
            self.replay.add(
                state, action, reward, next_state, (~terminated).float(),
                torch.full_like(reward, self.gamma), **kwargs,
            )
            return

        if active is None and self._uniform_count is not None and any_done is False:
            env_ids = self._all_env_ids
            tail = (self.head + self._uniform_count) % self.n_steps
            self.state[tail, env_ids] = state
            self.next_state[tail, env_ids] = next_state
            self.action[tail, env_ids] = action
            self.reward[tail, env_ids] = reward
            self.terminated[tail, env_ids] = terminated
            if self.asymmetric:
                self.critic_state[tail, env_ids] = critic_state
                self.next_critic[tail, env_ids] = next_critic
            self._uniform_count += 1
            self.count.fill_(self._uniform_count)
            if self._uniform_count == self.n_steps:
                self._emit(env_ids, uniform_k=self.n_steps)
                self._uniform_count -= 1
            return

        if active is None:
            env_ids = self._all_env_ids
            done = terminated | truncated
        else:
            self._uniform_count = None
            active = active.reshape(-1).bool()
            env_ids = torch.nonzero(active, as_tuple=False).reshape(-1)
            if env_ids.numel() == 0:
                return
            done = (terminated | truncated)[env_ids]

        tail = (self.head[env_ids] + self.count[env_ids]) % self.n_steps
        self.state[tail, env_ids] = state[env_ids]
        self.next_state[tail, env_ids] = next_state[env_ids]
        self.action[tail, env_ids] = action[env_ids]
        self.reward[tail, env_ids] = reward[env_ids]
        self.terminated[tail, env_ids] = terminated[env_ids]
        if self.asymmetric:
            self.critic_state[tail, env_ids] = critic_state[env_ids]
            self.next_critic[tail, env_ids] = next_critic[env_ids]
        self.count[env_ids] += 1
        if self._uniform_count is not None:
            self._uniform_count += 1

        if self._uniform_count is not None:
            if self._uniform_count == self.n_steps:
                self._emit(env_ids, uniform_k=self.n_steps)
                self._uniform_count -= 1
        else:
            full = env_ids[self.count[env_ids] == self.n_steps]
            if full.numel() > 0:
                self._emit(full)

        if any_done is not False:
            done_ids = env_ids[done]
            if done_ids.numel() > 0:
                if done_ids.numel() == self.num_envs and self._uniform_count is not None:
                    while self._uniform_count > 0:
                        self._emit(env_ids, uniform_k=self._uniform_count)
                        self._uniform_count -= 1
                else:
                    self._uniform_count = None
                    while True:
                        pending = done_ids[self.count[done_ids] > 0]
                        if pending.numel() == 0:
                            break
                        self._emit(pending)

    def flush_all(self):
        """Drain any open horizons (e.g. when a training phase ends mid-episode)."""
        if self.n_steps == 1:
            return
        if self._uniform_count is not None:
            while self._uniform_count > 0:
                self._emit(self._all_env_ids, uniform_k=self._uniform_count)
                self._uniform_count -= 1
            return
        while True:
            pending = torch.nonzero(self.count > 0, as_tuple=False).reshape(-1)
            if pending.numel() == 0:
                break
            self._emit(pending)
        self._uniform_count = 0


def build_parser():
    defaults = BaseArgs()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-id", default="BallCatcher-Pitch")
    parser.add_argument("--env-kwargs", default="{}",
        help="JSON dict of extra keyword arguments forwarded to make_baseline_vec (both training and eval)")
    parser.add_argument("--mesh-bound-m", type=float,
        help="Override MeshSpaceCodec / CBDSAMeshSpaceCodec perturbation bound in metres (default: 0.0003 raw, 0.005 CBD-SA)")
    parser.add_argument("--use-cbd", action=argparse.BooleanOptionalAction, default=False,
        help="Use CBD-SA (+ SVD) cage-based deformation codec for MeshSpace environments (default: False)")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=0,
        help="FastTD3 vector steps trained on the base design before CMA-ES generation 0 (default: 0)")
    parser.add_argument("--search-frequency", type=int,
        help="FastTD3 vector steps (each followed by the FastTD3 gradient updates) before each "
             "CMA-ES generation. LOKI's flag of the same name counts PPO updates instead. "
             "Default: train until every world has finished at least one episode")
    parser.add_argument("--population-size", type=int, default=128,
        help="CMA candidates per generation; remaining simulator design slots repeat candidates")
    parser.add_argument("--covariance", choices=("diagonal", "full"), default="diagonal")
    parser.add_argument("--cma-scale", type=float, default=1.0,
        help="CMA-ES initial step-size multiplier (scales sigma = range / 6; note Nevergrad multiplies by 0.15 for bounded dim >= 18)")
    parser.add_argument("--cma-min-std", type=float, default=0.0,
        help="Optional minimum CMA-ES standardized std floor (pycma inopts.minstd, 1.0 = range/6; 0 disables)")
    parser.add_argument("--cma-elitist", action=argparse.BooleanOptionalAction, default=False,
        help="Enable elitist CMA-ES selection")
    parser.add_argument("--cma-init-mid", action=argparse.BooleanOptionalAction, default=False,
        help="Initialize continuous CMA-ES mean at the midpoint of design bounds instead of base (default: False)")
    parser.add_argument("--warmup-diverse", action=argparse.BooleanOptionalAction, default=False,
        help="Initialize warmup training slots with population_size diverse designs (1 base + P-1 sampled) instead of repeating base (default: False)")
    parser.add_argument("--explore-design-frac", type=float, default=0.0,
        help="Fraction of training design slots replaced each phase with fresh random designs in [0, 1) (default: 0.0)")
    parser.add_argument("--explore-design-scale", type=float, default=1.0,
        help="Gaussian std scale for mesh exploration designs (> 1.0 draws per-design radial scales in [0.5, scale]; default: 1.0)")
    parser.add_argument("--noise-corr", type=float, default=0.0,
        help="Max AR(1) temporal correlation rho for per-episode exploration noise in [0, 0.99] (default: 0.0 = white noise)")
    parser.add_argument("--noise-burst-frac", type=float, default=0.0,
        help="Fraction of training episodes using delayed-onset correlated exploration bursts in [0, 1] (default: 0.0)")
    parser.add_argument("--noise-dim-prob", type=float, default=1.0,
        help="Per-dimension noise mask probability during delayed exploration bursts in (0, 1] (default: 1.0)")
    parser.add_argument("--noise-burst-override", action=argparse.BooleanOptionalAction, default=False,
        help="Zero-center masked action dimensions during delayed bursts instead of adding noise onto saturated policy outputs (default: False)")
    parser.add_argument("--score-into-replay", action=argparse.BooleanOptionalAction, default=True,
        help="Push deterministic CMA-ES scoring rollouts into the RL replay buffer (default: True)")
    parser.add_argument("--design-smoothness-weight", type=float, default=0.0,
        help="Weight on heightfield pad spatial smoothness penalty (mean squared neighbor diff) in CMA-ES fitness (default: 0.0)")
    parser.add_argument("--design-condition", choices=("cmaes", "mid", "random"), default="cmaes",
        help="Embodiment condition: 'cmaes' (co-design search), 'mid' (fixed midpoint of bounds, "
             "e.g. 0.5 pads / 0.0 links), or 'random' (fixed per-seed uniform random design)")
    parser.add_argument("--evaluation-worlds", type=int, default=128)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--rejection-reward", type=float)
    replay_args = parser.add_mutually_exclusive_group()
    replay_args.add_argument("--buffer-size", type=int, help="Legacy units: each unit is 128 transitions, independent of num-envs")
    replay_args.add_argument("--replay-transitions", type=int, help="Requested transition capacity; default 6,553,600, capped by the memory budget")
    parser.add_argument("--replay-memory-gb", type=float, default=8.0,
        help="Replay storage cap in GiB, further limited to half the currently free CUDA memory")

    # FastTD3 RL hyperparameter overrides (defaulting to BaseArgs)
    parser.add_argument("--actor-hidden-dim", type=int, default=defaults.actor_hidden_dim)
    parser.add_argument("--critic-hidden-dim", type=int, default=defaults.critic_hidden_dim)
    parser.add_argument("--actor-lr", type=float, default=defaults.actor_learning_rate)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--tau", type=float, default=defaults.tau)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps,
        help="N-step return horizon (default: 1)")
    parser.add_argument("--num-updates", type=int, default=defaults.num_updates,
        help="Critic gradient updates per vector step (UTD ratio; default: 2)")
    parser.add_argument("--policy-frequency", type=int, default=defaults.policy_frequency,
        help="Actor update period in critic updates (default: 2)")
    parser.add_argument("--learning-starts", type=int, default=defaults.learning_starts)
    parser.add_argument("--v-min", type=float, default=defaults.v_min)
    parser.add_argument("--v-max", type=float, default=defaults.v_max)
    parser.add_argument("--num-atoms", type=int, default=defaults.num_atoms)
    parser.add_argument("--std-min", type=float, default=defaults.std_min)
    parser.add_argument("--std-max", type=float, default=defaults.std_max)
    parser.add_argument("--policy-noise", type=float, default=defaults.policy_noise)
    parser.add_argument("--noise-clip", type=float, default=defaults.noise_clip)
    parser.add_argument("--obs-normalization", action=argparse.BooleanOptionalAction, default=defaults.obs_normalization)
    parser.add_argument("--normalize-design", action=argparse.BooleanOptionalAction, default=True,
        help="Include trailing embodiment features in empirical observation normalization (default: True; "
             "pass --no-normalize-design to normalize only task observations and leave [-1, 1] design features untouched)")
    parser.add_argument("--max-grad-norm", type=float, default=0.0,
        help="Optional gradient norm clipping threshold for actor and critic (0 disables; default: 0.0)")
    parser.add_argument("--use-cdq", action=argparse.BooleanOptionalAction, default=defaults.use_cdq)
    parser.add_argument("--asymmetric-critic", action=argparse.BooleanOptionalAction, default=True,
        help="Feed privileged critic observations to the critic when exposed by the environment (default: True)")

    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--run-name")
    return parser


def resolve_hp(args):
    hp = asdict(BaseArgs())
    hp.update(
        batch_size=args.batch_size,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        tau=args.tau,
        num_steps=args.num_steps,
        num_updates=args.num_updates,
        policy_frequency=args.policy_frequency,
        learning_starts=args.learning_starts,
        v_min=args.v_min,
        v_max=args.v_max,
        num_atoms=args.num_atoms,
        std_min=args.std_min,
        std_max=args.std_max,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        obs_normalization=args.obs_normalization,
        normalize_design=args.normalize_design,
        use_grad_norm_clipping=args.max_grad_norm > 0.0,
        max_grad_norm=args.max_grad_norm,
        use_cdq=args.use_cdq,
    )
    return hp


def main():
    parser = build_parser()
    args = parser.parse_args()
    capacity = replay_capacity(args.buffer_size, args.replay_transitions)
    if min(args.num_envs, args.batch_size, args.population_size, args.evaluation_worlds,
           args.video_stride, args.video_fps, args.actor_hidden_dim, args.critic_hidden_dim,
           args.num_steps, args.num_updates, args.policy_frequency) < 1 or capacity < args.num_envs:
        parser.error("positive environment/batch/population/network/update sizes and replay capacity >= num-envs are required")
    if args.search_frequency is not None and args.search_frequency < 1:
        parser.error("search-frequency must be positive")
    if args.warmup_steps < 0 or args.learning_starts < 0:
        parser.error("warmup-steps and learning-starts must be non-negative")
    if args.num_atoms < 2 or not (math.isfinite(args.v_min) and math.isfinite(args.v_max) and args.v_min < args.v_max):
        parser.error("num-atoms >= 2 and finite v-min < v-max are required")
    if not (0.0 <= args.std_min <= args.std_max) or args.policy_noise < 0 or args.noise_clip < 0 or args.weight_decay < 0 or args.max_grad_norm < 0:
        parser.error("invalid noise, weight-decay, or max-grad-norm parameters")
    if not (0.0 < args.gamma <= 1.0) or not (0.0 < args.tau <= 1.0) or args.actor_lr <= 0 or args.critic_lr <= 0:
        parser.error("invalid gamma, tau, or learning rate")
    if not math.isfinite(args.cma_scale) or args.cma_scale <= 0:
        parser.error("cma-scale must be finite and positive")
    if not math.isfinite(args.cma_min_std) or args.cma_min_std < 0:
        parser.error("cma-min-std must be finite and non-negative")
    if not math.isfinite(args.explore_design_frac) or not (0.0 <= args.explore_design_frac < 1.0):
        parser.error("explore-design-frac must be in [0.0, 1.0)")
    if not math.isfinite(args.explore_design_scale) or args.explore_design_scale <= 0.0:
        parser.error("explore-design-scale must be finite and positive")
    if not math.isfinite(args.noise_corr) or not (0.0 <= args.noise_corr <= 0.99):
        parser.error("noise-corr must be in [0.0, 0.99]")
    if not math.isfinite(args.noise_burst_frac) or not (0.0 <= args.noise_burst_frac <= 1.0):
        parser.error("noise-burst-frac must be in [0.0, 1.0]")
    if not math.isfinite(args.noise_dim_prob) or not (0.0 < args.noise_dim_prob <= 1.0):
        parser.error("noise-dim-prob must be in (0.0, 1.0]")
    if not math.isfinite(args.design_smoothness_weight) or args.design_smoothness_weight < 0.0:
        parser.error("design-smoothness-weight must be finite and non-negative")
    if args.mesh_bound_m is not None and (not math.isfinite(args.mesh_bound_m) or args.mesh_bound_m <= 0):
        parser.error("mesh-bound-m must be finite and positive")
    if not math.isfinite(args.replay_memory_gb) or args.replay_memory_gb <= 0:
        parser.error("replay-memory-gb must be finite and positive")
    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --env-kwargs JSON: {exc}")
    if not isinstance(env_kwargs, dict):
        parser.error("--env-kwargs must decode to a JSON object")

    budget = WallClockBudget.start(args.hours)
    hp = resolve_hp(args)
    amp_enabled = args.amp and torch.device(args.device).type == "cuda"
    design_search_desc = (
        f"{args.covariance} CMA-ES over the embodiment only"
        if args.design_condition == "cmaes"
        else f"fixed embodiment ({args.design_condition}; CMA-ES disabled)"
    )
    log = RunLog(args.output_dir, budget, {**hp, **vars(args), "method": METHOD,
        "amp_effective": amp_enabled, "source_preset": "FastTD3 BaseArgs (HumanoidBench)",
        "controller": "FastTD3 actor over embodiment-conditioned observations",
        "design_search": design_search_desc,
        "search_frequency_unit": "FastTD3 vector steps" if args.search_frequency else "one episode in every world",
        "replay_capacity_requested": capacity, "replay_reference_envs": REPLAY_REFERENCE_ENVS,
        "schedule_source": (
            "FastTD3 training phase followed by one frozen deterministic-policy CMA-ES "
            "generation in the training environment; scoring transitions enter replay"
            if args.design_condition == "cmaes"
            else f"FastTD3 training only on fixed '{args.design_condition}' embodiment"
        ),
        "evaluation_execution": "independent_process",
        "evaluation_protocol": (
            "best accepted design of the latest CMA-ES generation under the current "
            "deterministic FastTD3 policy in every evaluation world; base design before "
            "the first generation"
            if args.design_condition == "cmaes"
            else f"fixed '{args.design_condition}' design under the current deterministic FastTD3 policy"
        )},
        project=args.project, entity=args.entity, group=args.group, name=args.run_name)
    env = evaluator = optimizer = None
    train_steps = scoring_steps = updates = generation = 0
    train_frames = search_frames = candidates_told = rejections = 0
    try:
        evaluator = AsyncEvaluator(log, args, METHOD)
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = hp["torch_deterministic"]
        env, base = make_baseline_vec(args.env_id, args.num_envs, device=args.device, seed=args.seed,
            control_mode="logits", rejection_reward=args.rejection_reward,
            mesh_bound_m=args.mesh_bound_m, **env_kwargs)
        if args.use_cbd:
            from .cbdsa_codec import apply_cbdsa_codec
            env, base = apply_cbdsa_codec(env, bound_m=args.mesh_bound_m)
        if capacity < env.num_envs:
            raise ValueError(f"replay capacity {capacity} is smaller than the {env.num_envs} simulator worlds")
        base = np.asarray(base, dtype=np.float32)
        design_dim = len(base)
        obs_dim = env.single_observation_space.shape[0]
        action_dim = env.single_action_space.shape[0]
        state_dim = obs_dim + design_dim
        critic_obs_space = getattr(env, "single_critic_observation_space", None)
        asymmetric = bool(args.asymmetric_critic and critic_obs_space is not None)
        critic_state_dim = (critic_obs_space.shape[0] + design_dim) if asymmetric else state_dim

        codec = env.design_codec
        pad_grids = extract_pad_grids(env)
        binary_design = bool(codec.discrete.all()) and np.all(codec.low == 0) and np.all(codec.high == 1)
        population_size = min(args.population_size, env.num_designs)
        assignment = population_assignment(env.num_designs, population_size)
        snapshot_design, training_designs, snapshot_kind = initial_condition_designs(
            args.design_condition, env.embodiment_space.low, env.embodiment_space.high,
            base, env.num_designs, seed=args.seed, cma_init_mid=args.cma_init_mid, binary=binary_design)
        is_mesh_env = hasattr(getattr(getattr(env, "env", None), "embodiment_space", None), "base_meshes")
        explore_max_unique = max(1, int(round(population_size * args.explore_design_frac))) if is_mesh_env else None
        explore_rng = np.random.default_rng(args.seed + TRAINING_SEED_OFFSET + 13_337)
        if args.design_condition == "cmaes" and args.warmup_diverse and population_size > 1:
            warmup_pop = np.repeat(snapshot_design[None], population_size, axis=0)
            warmup_pop = mix_exploration_designs(
                warmup_pop, (population_size - 1) / population_size,
                env.embodiment_space.low, env.embodiment_space.high, base,
                explore_rng, binary=binary_design,
                gaussian=is_mesh_env, max_unique=population_size - 1,
                gaussian_scale=args.explore_design_scale)
            training_designs = warmup_pop[assignment]
        elif args.design_condition == "cmaes" and args.explore_design_frac > 0.0:
            training_designs = mix_exploration_designs(
                training_designs, args.explore_design_frac,
                env.embodiment_space.low, env.embodiment_space.high, base,
                explore_rng, binary=binary_design,
                gaussian=is_mesh_env, max_unique=explore_max_unique,
                gaussian_scale=args.explore_design_scale)
        optimizer = build_design_optimiser(env.embodiment_space.low, env.embodiment_space.high,
            snapshot_design if args.design_condition == "cmaes" else base,
            population_size=population_size, seed=args.seed, binary=binary_design,
            diagonal=args.covariance == "diagonal", scale=args.cma_scale,
            elitist=args.cma_elitist, min_std=args.cma_min_std)
        log.log(cma_population_size=population_size, cma_design_slots=env.num_designs,
                cma_repeats_min=int(np.bincount(assignment).min()),
                cma_repeats_max=int(np.bincount(assignment).max()),
                cma_binary_logits=bool(binary_design),
                asymmetric_critic_effective=asymmetric,
                critic_state_dim=critic_state_dim)

        actor = Actor(state_dim, action_dim, env.num_envs, hp["init_scale"], hp["actor_hidden_dim"],
            std_min=hp["std_min"], std_max=hp["std_max"], device=args.device)
        critic = Critic(critic_state_dim, action_dim, hp["num_atoms"], hp["v_min"], hp["v_max"],
            hp["critic_hidden_dim"], "", 64, 8, device=args.device)
        target = deepcopy(critic)
        def make_normalizer(total_dim, task_dim):
            if not hp["obs_normalization"]:
                return None
            if hp["normalize_design"]:
                return EmpiricalNormalization(total_dim, args.device)
            return PrefixEmpiricalNormalization(task_dim, args.device)

        normalization = make_normalizer(state_dim, obs_dim)
        critic_obs_dim = critic_obs_space.shape[0] if asymmetric else obs_dim
        critic_normalization = (
            make_normalizer(critic_state_dim, critic_obs_dim)
            if (hp["obs_normalization"] and asymmetric)
            else normalization
        )
        inference = RolloutInference(actor, normalization, compile=args.compile, amp=amp_enabled)
        normalize_critic = (
            (lambda x, update=False: x)
            if critic_normalization is None
            else (inference.normalize if critic_normalization is normalization else critic_normalization.forward)
        )
        if args.compile and critic_normalization is not None and critic_normalization is not normalization:
            normalize_critic = torch.compile(normalize_critic, mode=None)
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=hp["actor_learning_rate"], weight_decay=hp["weight_decay"])
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=hp["critic_learning_rate"], weight_decay=hp["weight_decay"])
        requested_capacity = capacity
        max_replay_bytes = args.replay_memory_gb * 2**30
        if torch.device(args.device).type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(args.device)
            max_replay_bytes = min(max_replay_bytes, free_bytes // 2)
        replay_critic_dim = critic_state_dim if asymmetric else None
        capacity = fit_replay_capacity(capacity, state_dim, action_dim, env.num_envs, max_replay_bytes,
                                       critic_state_dim=replay_critic_dim)
        replay = AsymmetricReplay(capacity, state_dim, action_dim, args.device, critic_state_dim=replay_critic_dim)
        accumulator = NStepAccumulator(replay, env.num_envs, state_dim, action_dim,
                                       hp["num_steps"], hp["gamma"], args.device, critic_state_dim=replay_critic_dim)
        log.log(replay_capacity_requested=requested_capacity, replay_memory_budget_bytes=int(max_replay_bytes),
                replay_capacity_transitions=capacity,
                replay_bytes=sum(value.numel() * value.element_size() for value in replay.data.values()))

        base_t = torch.as_tensor(base, device=args.device)
        episode_returns = torch.zeros(env.num_envs, device=args.device)
        train_reward = None
        snapshot_score = None
        ar_explorer = (
            CorrelatedExploration(
                env.num_envs, action_dim, args.noise_corr, args.device,
                burst_frac=args.noise_burst_frac, dim_prob=args.noise_dim_prob,
                override=args.noise_burst_override,
                std_min=hp["std_min"], std_max=hp["std_max"],
            )
            if (args.noise_corr > 0.0 or args.noise_burst_frac > 0.0)
            else None
        )

        def frames():
            return train_frames + search_frames

        def condition_critic(installed):
            if not asymmetric:
                return None
            return condition_observation(env, env.last_critic_obs, installed)

        def publish_snapshot(force=False):
            design = snapshot_design.copy()
            evaluator.publish(lambda: {**policy_snapshot(actor, normalization, env_id=args.env_id,
                    seed=args.seed, state_dim=state_dim, action_dim=action_dim, design_dim=design_dim,
                    hp=hp, amp=amp_enabled),
                    "design": design, "embodiment_conditioned": True, "generation": generation,
                    "mesh_bound_m": args.mesh_bound_m, "use_cbd": args.use_cbd},
                force=force, kind=snapshot_kind, frames=frames(), train_frames=train_frames,
                search_frames=search_frames, vector_steps=train_steps, updates=updates,
                generation=generation,
                **{"eval/snapshot_generation_score": snapshot_score, "train/reward": train_reward,
                   "train/fps": frames() / max(budget.elapsed, 1e-9)})

        def checkpoint_payload():
            return {"format_version": 1, "method": METHOD, "env_id": args.env_id, "seed": args.seed,
                "state_dim": state_dim, "critic_state_dim": critic_state_dim, "asymmetric_critic": asymmetric,
                "action_dim": action_dim, "design_dim": design_dim, "mesh_bound_m": args.mesh_bound_m,
                "use_cbd": args.use_cbd,
                "num_envs": env.num_envs, "num_designs": env.num_designs, "hp": hp, "amp": amp_enabled,
                "actor": actor.state_dict(), "critic": critic.state_dict(), "target": target.state_dict(),
                "normalization": None if normalization is None else normalization.state_dict(),
                "critic_normalization": None if critic_normalization is None else critic_normalization.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(), "design": snapshot_design.copy(),
                "design_score": snapshot_score, "training_designs": training_designs.copy(),
                "design_condition": args.design_condition,
                "generation": generation, "embodiment_conditioned": True, "control_mode": "logits"}

        def install(designs, seed):
            """Full reset with one design per slot. Rejected slots run the base design."""
            observation, rejected, _ = env.reset(seed=seed, embodiment=designs)
            rejected = torch.as_tensor(rejected, dtype=torch.bool, device=args.device).reshape(-1)
            installed = torch.as_tensor(designs, dtype=torch.float32, device=args.device).clone()
            installed[env.collapse_rejections(rejected)] = base_t
            return observation, rejected, installed

        use_cdq = bool(hp["use_cdq"])
        policy_noise = float(hp["policy_noise"])
        noise_clip = float(hp["noise_clip"])
        max_grad_norm = float(hp["max_grad_norm"])

        def update_critic(data, c_obs, next_obs, next_c_obs):
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=amp_enabled):
                with torch.no_grad():
                    if policy_noise > 0.0:
                        noise = (torch.randn_like(data["action"]) * policy_noise).clamp(-noise_clip, noise_clip)
                        next_action = env.project_control((actor(next_obs) + noise).clamp(-1, 1))
                    else:
                        next_action = env.project_control(actor(next_obs).clamp(-1, 1))
                    p1, p2 = target.projection(next_c_obs, next_action, data["reward"][:, 0], data["bootstrap"][:, 0], data["discount"][:, 0])
                    if use_cdq:
                        target_dist = torch.where((target.get_value(p1) < target.get_value(p2))[:, None], p1, p2)
                    else:
                        target_dist = 0.5 * (p1 + p2)
                q1, q2 = critic(c_obs, data["action"])
                loss = -(target_dist * (F.log_softmax(q1, dim=1) + F.log_softmax(q2, dim=1))).sum(dim=1).mean()
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optimizer.step()
            return loss.detach()

        def update_actor(obs, c_obs):
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=amp_enabled):
                action = env.project_control(actor(obs).clamp(-1, 1))
                q1, q2 = critic(c_obs, action)
                v1, v2 = critic.get_value(q1.softmax(1)), critic.get_value(q2.softmax(1))
                actor_loss = -torch.minimum(v1, v2).mean() if use_cdq else -0.5 * (v1 + v2).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optimizer.step()
            return actor_loss.detach()

        if args.compile:
            update_critic = torch.compile(update_critic, mode=hp["compile_mode"])
            update_actor = torch.compile(update_actor, mode=hp["compile_mode"])

        def train_phase(designs, phase_target_steps):
            """FastTD3 on fixed designs. Finished worlds restart with the same design."""
            nonlocal train_steps, updates, train_frames, train_reward
            # Scoring reseeds the simulator with --seed, so give each training
            # phase its own seed; otherwise every phase would replay the same
            # sequence of initial states.
            observation, rejected, installed = install(designs, args.seed + TRAINING_SEED_OFFSET + generation)
            installed_np = installed.cpu().numpy()
            state = condition_observation(env, observation, installed)
            critic_state = condition_critic(installed)
            starts = torch.ones(env.num_envs, dtype=torch.bool, device=args.device)
            finished_once = torch.zeros_like(starts)
            episode_returns.zero_()
            phase_steps = 0
            phase_rejections = int(env.collapse_rejections(rejected).sum())
            policy_freq = hp["policy_frequency"]
            last_critic_loss = None
            last_actor_loss = None
            window_max_reward = None
            is_lockstep_mesh = (
                hasattr(getattr(getattr(env, "env", None), "embodiment_space", None), "base_meshes")
                and hasattr(env.env, "_episode_over")
            )
            while not budget.expired and not phase_complete(phase_steps, phase_target_steps, finished_once):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    if asymmetric and critic_normalization is not None:
                        normalize_critic(critic_state, update=True)
                    if ar_explorer is not None:
                        det_action = inference(state, dones=starts, deterministic=True)
                        action = env.project_control(ar_explorer.apply(det_action, starts, actor.noise_scales).clamp(-1, 1))
                    else:
                        action = env.project_control(inference(state, dones=starts).clamp(-1, 1))
                    observation, reward, terminated, truncated, _ = env.step(action)
                    # Terminal observations are kept for timeout bootstrapping.
                    next_state = condition_observation(env, observation, installed)
                    next_critic_state = condition_critic(installed)
                    any_done_known = bool(env.env._episode_over) if is_lockstep_mesh else None
                    accumulator.push(
                        state, action, reward, next_state, terminated, truncated,
                        critic_state=critic_state, next_critic=next_critic_state,
                        any_done=any_done_known,
                    )
                    episode_returns.add_(reward.reshape(-1))
                    done = (terminated | truncated).reshape(-1)
                    # A design slot restarts as a group, as in the joint FastTD3 rollout.
                    done = env.expand_design_values(env.collapse_rejections(done))
                    if any_done_known if is_lockstep_mesh else bool(done.any()):
                        done_returns = episode_returns[done]
                        train_reward = float(done_returns.mean())
                        done_max = float(done_returns.max())
                        window_max_reward = done_max if window_max_reward is None else max(window_max_reward, done_max)
                        episode_returns[done] = 0
                        finished_once |= done
                        observation, rejected, _ = env.reset(
                            embodiment=None if (is_lockstep_mesh and bool(done.all())) else installed_np,
                            options={"reset_mask": done},
                        )
                        rejected = torch.as_tensor(rejected, dtype=torch.bool, device=args.device).reshape(-1) & done
                        if bool(rejected.any()):  # Accepted designs should not be rejected again.
                            phase_rejections += int(env.collapse_rejections(rejected).sum())
                            installed[env.collapse_rejections(rejected)] = base_t
                            installed_np = installed.cpu().numpy()
                        next_state = condition_observation(env, observation, installed)
                        next_critic_state = condition_critic(installed)
                    state, critic_state, starts = next_state, next_critic_state, done
                    train_frames += env.num_envs
                if train_steps > hp["learning_starts"] and replay.count > 0:
                    for update_index in range(hp["num_updates"]):
                        if budget.expired:
                            break
                        data = replay.sample(args.batch_size)
                        obs = inference.normalize(data["state"], update=False)
                        next_obs = inference.normalize(data["next"], update=False)
                        if asymmetric:
                            c_obs = normalize_critic(data["critic_state"], update=False)
                            next_c_obs = normalize_critic(data["next_critic"], update=False)
                        else:
                            c_obs, next_c_obs = obs, next_obs
                        torch.compiler.cudagraph_mark_step_begin()
                        last_critic_loss = update_critic(data, c_obs, next_obs, next_c_obs)
                        if update_index % policy_freq == policy_freq - 1:
                            last_actor_loss = update_actor(obs, c_obs)
                        with torch.no_grad():
                            for source, destination in zip(critic.parameters(), target.parameters()):
                                destination.lerp_(source, hp["tau"])
                        updates += 1
                train_steps += 1
                phase_steps += 1
                publish_snapshot()
                if train_steps % 100 == 0:
                    evaluator.check()
                    metrics = {"vector_steps": train_steps, "updates": updates, "frames": frames(),
                        "train_frames": train_frames, "search_frames": search_frames,
                        "fps": frames() / max(budget.elapsed, 1e-9), "generation": generation}
                    if train_reward is not None:
                        metrics["train/reward"] = train_reward
                    if window_max_reward is not None:
                        metrics["train/reward_max"] = window_max_reward
                        window_max_reward = None
                    if last_critic_loss is not None:
                        metrics["train/critic_loss"] = float(last_critic_loss.item())
                    if last_actor_loss is not None:
                        metrics["train/actor_q"] = float(-last_actor_loss.item())
                    log.log(**metrics)
            accumulator.flush_all()
            return phase_steps, phase_rejections

        def score_generation(candidates):
            """One deterministic frozen-policy episode per world. None if the budget ran out."""
            nonlocal scoring_steps, search_frames
            designs = np.stack([decode_design(candidate.value, binary_design) for candidate in candidates]).astype(np.float32)
            observation, rejected, _ = env.reset(seed=args.seed, embodiment=designs[assignment])
            rejected = torch.as_tensor(rejected, dtype=torch.bool, device=args.device).reshape(-1)
            installed = torch.as_tensor(designs[assignment], dtype=torch.float32, device=args.device)
            installed[env.collapse_rejections(rejected)] = base_t
            result = EvaluationAccumulator(env)
            result.reject(rejected, env.rejection_reward)
            state = condition_observation(env, observation, installed)
            critic_state = condition_critic(installed)
            steps = 0
            while not result.complete and not budget.expired:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    active = ~result.finished
                    action = env.project_control(inference(state, deterministic=True, update=False).clamp(-1, 1))
                    observation, reward, terminated, truncated, _ = env.step(action)
                    next_state = condition_observation(env, observation, installed)
                    next_critic_state = condition_critic(installed)
                    # Off-policy data: optionally keep each world's single scoring episode.
                    if args.score_into_replay:
                        accumulator.push(
                            state, action, reward, next_state, terminated, truncated,
                            critic_state=critic_state, next_critic=next_critic_state, active=active,
                        )
                    result.add(reward, terminated, truncated)
                    state, critic_state = next_state, next_critic_state
                steps += 1
                scoring_steps += 1
                search_frames += env.num_envs
                if steps % 100 == 0:
                    log.log(**{"search/status": "running", "search/generation": generation, "search/steps": steps})
                    publish_snapshot()
            if args.score_into_replay:
                accumulator.flush_all()
            if not result.complete:
                return None  # Never give CMA-ES a partial-episode fitness.
            scores, invalid = population_scores(
                env.collapse_returns(result.returns).cpu().numpy(),
                env.collapse_rejections(rejected).cpu().numpy(), assignment, population_size)
            return designs, scores, invalid, steps, result.summary()

        publish_snapshot(force=True)
        while not budget.expired:
            evaluator.check()
            if args.design_condition != "cmaes":
                train_phase(training_designs, float("inf"))
                break
            phase_target = args.warmup_steps if (generation == 0 and args.warmup_steps > 0) else args.search_frequency
            phase_steps, phase_rejections = train_phase(training_designs, phase_target)
            if budget.expired:
                break
            candidates = [optimizer.ask() for _ in range(population_size)]
            scored = score_generation(candidates)
            if scored is None:
                break
            designs, scores, invalid, steps, summary = scored
            penalties = design_smoothness_penalty(designs, pad_grids)
            selection_scores = (
                scores - args.design_smoothness_weight * penalties
                if args.design_smoothness_weight > 0.0 and pad_grids
                else scores
            )
            for candidate, sel_score in zip(candidates, selection_scores):
                optimizer.tell(candidate, -float(sel_score))
            candidates_told += len(candidates)
            rejections += int(invalid.sum())
            best_design, best_score, best_idx = generation_best(
                designs, scores, invalid, selection_scores=selection_scores)
            best_smoothness = float(penalties[best_idx]) if (best_idx is not None and pad_grids) else None
            if best_design is not None:
                snapshot_design, snapshot_score, snapshot_kind = best_design, best_score, "generation_best"
            # If every candidate was rejected, the previous snapshot design remains.
            training_designs = slot_designs(designs, invalid, snapshot_design, assignment)
            if args.explore_design_frac > 0.0:
                training_designs = mix_exploration_designs(
                    training_designs, args.explore_design_frac,
                    env.embodiment_space.low, env.embodiment_space.high, base,
                    explore_rng, binary=binary_design,
                    gaussian=is_mesh_env, max_unique=explore_max_unique,
                    gaussian_scale=args.explore_design_scale)
            generation += 1
            publish_snapshot(force=True)
            np.savez(log.output / "generation_best.npz", design=snapshot_design, score=np.nan if snapshot_score is None else snapshot_score,
                     generation=generation, observation_size=obs_dim, action_size=action_dim, embodiment_conditioned=True)
            np.save(log.output / "embodiments.npy", training_designs)
            valid_designs = designs[~invalid]
            pop_std_mean = float(valid_designs.std(axis=0).mean()) if len(valid_designs) > 1 else 0.0
            best_delta_mean = float(np.abs(snapshot_design - base).mean())
            metrics = dict(generation=generation, candidates=candidates_told, rejections=rejections,
                embodiment_rejections=int(invalid.sum()), embodiment_rejection_rate=float(invalid.mean()),
                score=float(scores.mean()), score_std=float(scores.std()), best_score=best_score,
                generation_all_rejected=best_design is None, search_steps=steps,
                search_length_mean=summary["length_mean"], search_success_rate=summary["success_rate"],
                train_phase_steps=phase_steps, train_phase_rejections=phase_rejections,
                design_pop_std=pop_std_mean, design_best_delta=best_delta_mean,
                vector_steps=train_steps, scoring_steps=scoring_steps, updates=updates, frames=frames(),
                train_frames=train_frames, search_frames=search_frames,
                fps=frames() / max(budget.elapsed, 1e-9), **{"train/reward": train_reward})
            if pad_grids:
                metrics["search/smoothness_penalty"] = float(penalties.mean())
                metrics["search/best_smoothness_penalty"] = best_smoothness
                if args.design_smoothness_weight > 0.0:
                    metrics["search/penalized_score"] = float(selection_scores.mean())
            log.log(**{key: value for key, value in metrics.items() if value is not None})
        evaluator.close()
        torch.save(checkpoint_payload(), log.output / "checkpoint.pt")
        optimizer.dump(log.output / "optimizer.pkl")
        if not (log.output / "best_checkpoint.pt").exists():
            torch.save(checkpoint_payload(), log.output / "best_checkpoint.pt")
        log.finish("complete", generations=generation, vector_steps=train_steps, scoring_steps=scoring_steps,
            updates=updates, frames=frames(), train_frames=train_frames, search_frames=search_frames,
            candidates=candidates_told, rejections=rejections, best_eval_reward=evaluator.best_eval_reward)
    except Exception:
        if evaluator is not None:
            evaluator.close(raise_errors=False)
        log.finish("failed", error=traceback.format_exc(), generations=generation, frames=train_frames + search_frames)
        raise
    finally:
        if env is not None:
            env.close()
        if evaluator is not None:
            evaluator.close(raise_errors=False)


if __name__ == "__main__":
    main()
