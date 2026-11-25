"""Flat tensor transport for structured Codesign Gym environments."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from codesign_gym import make_vec


@dataclass(frozen=True)
class _Leaf:
    path: tuple[str, ...]
    space: spaces.Space
    start: int
    stop: int


class SpaceCodec:
    """Flatten Box/Dict/discrete designs without one-hot expanding the design."""

    def __init__(self, space: spaces.Space):
        self.space = space
        leaves = []

        def visit(current, path):
            if isinstance(current, spaces.Dict):
                for key, child in current.spaces.items():
                    visit(child, (*path, key))
                return
            if isinstance(current, spaces.Box):
                size = int(np.prod(current.shape, dtype=int))
            elif isinstance(current, spaces.MultiBinary):
                size = int(np.prod(current.shape, dtype=int))
            elif isinstance(current, spaces.Discrete):
                size = 1
            else:
                raise TypeError(f"unsupported embodiment space {type(current).__name__}")
            start = leaves[-1].stop if leaves else 0
            leaves.append(_Leaf(path, current, start, start + size))

        visit(space, ())
        self.leaves = tuple(leaves)
        low, high, discrete = [], [], []
        for leaf in self.leaves:
            current = leaf.space
            if isinstance(current, spaces.Box):
                low.extend(np.asarray(current.low).reshape(-1))
                high.extend(np.asarray(current.high).reshape(-1))
                discrete.extend([np.issubdtype(current.dtype, np.integer)] * (leaf.stop - leaf.start))
            elif isinstance(current, spaces.MultiBinary):
                low.extend([0] * (leaf.stop - leaf.start))
                high.extend([1] * (leaf.stop - leaf.start))
                discrete.extend([True] * (leaf.stop - leaf.start))
            else:
                low.append(current.start)
                high.append(current.start + current.n - 1)
                discrete.append(True)
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.discrete = np.asarray(discrete, dtype=bool)
        self.space_flat = spaces.Box(self.low, self.high, dtype=np.float32)

    def encode(self, value: Any) -> np.ndarray:
        pieces = []
        for leaf in self.leaves:
            current = value
            for key in leaf.path:
                current = current[key]
            pieces.append(np.asarray(current, dtype=np.float32).reshape(-1))
        return np.concatenate(pieces) if pieces else np.zeros(0, np.float32)

    def decode(self, value: Any) -> Any:
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim == 1:
            return self._decode_one(array)
        if array.ndim != 2:
            raise ValueError(f"flat embodiment must have rank one or two, got {array.shape}")
        return [self._decode_one(row) for row in array]

    def _decode_one(self, value: np.ndarray) -> Any:
        if value.shape != self.low.shape:
            raise ValueError(f"flat embodiment has shape {value.shape}, expected {self.low.shape}")
        root = {} if isinstance(self.space, spaces.Dict) else None
        for leaf in self.leaves:
            current, chunk = leaf.space, np.asarray(value[leaf.start:leaf.stop])
            if isinstance(current, spaces.Box):
                if np.issubdtype(current.dtype, np.integer):
                    chunk = np.rint(chunk)
                item = chunk.astype(current.dtype).reshape(current.shape)
            elif isinstance(current, spaces.MultiBinary):
                # clip to {0, 1} so an out-of-range proposal decodes to a valid occupancy
                # vector (a stray 2 would otherwise fail structural validation for the
                # wrong reason -- non-binary rather than the actual design constraint).
                item = np.clip(np.rint(chunk), 0, 1).astype(current.dtype).reshape(current.shape)
            else:
                item = int(np.rint(chunk[0]))
            if not leaf.path:
                root = item
                continue
            target = root
            for key in leaf.path[:-1]:
                target = target.setdefault(key, {})
            target[leaf.path[-1]] = item
        return root

    def project_normalized(self, value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(-1, 1)
        if not self.discrete.any():
            return value
        low = torch.as_tensor(self.low, device=value.device, dtype=value.dtype)
        high = torch.as_tensor(self.high, device=value.device, dtype=value.dtype)
        physical = low + (value + 1) * 0.5 * (high - low)
        quantized = physical.round()
        normalized = torch.where(high > low, 2 * (quantized - low) / (high - low) - 1, torch.zeros_like(quantized))
        mask = torch.as_tensor(self.discrete, device=value.device)
        projected = torch.where(mask, normalized, value)
        return value + (projected - value).detach()


class IndexedSpaceCodec:
    """Flat integer view exposed by structured spaces such as Pokémon teams."""

    def __init__(self, space):
        self.space = space
        flat = space.flat_space
        if not isinstance(flat, spaces.MultiDiscrete):
            raise TypeError(f"indexed embodiment flat space must be MultiDiscrete, got {flat}")
        self.low = np.asarray(flat.start, dtype=np.float32).reshape(-1)
        self.high = (np.asarray(flat.start) + np.asarray(flat.nvec) - 1).astype(
            np.float32).reshape(-1)
        self.discrete = np.ones(len(self.low), dtype=bool)
        self.space_flat = spaces.Box(self.low, self.high, dtype=np.float32)

    def encode(self, value):
        return np.asarray(self.space.to_index(value), dtype=np.float32).reshape(-1)

    def decode(self, value):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim == 1:
            return self._decode_one(array)
        if array.ndim != 2:
            raise ValueError(f"flat embodiment must have rank one or two, got {array.shape}")
        return [self._decode_one(row) for row in array]

    def _decode_one(self, value):
        if value.shape != self.low.shape:
            raise ValueError(f"flat embodiment has shape {value.shape}, expected {self.low.shape}")
        indices = np.clip(np.rint(value), self.low, self.high).astype(np.int64)
        return self.space.from_index(indices)

    def project_normalized(self, value):
        value = value.clamp(-1, 1)
        low = torch.as_tensor(self.low, device=value.device, dtype=value.dtype)
        high = torch.as_tensor(self.high, device=value.device, dtype=value.dtype)
        physical = low + (value + 1) * 0.5 * (high - low)
        quantized = physical.round()
        projected = torch.where(
            high > low, 2 * (quantized - low) / (high - low) - 1,
            torch.zeros_like(quantized))
        return value + (projected - value).detach()


class MeshSpaceCodec:
    """Fixed-topology mutable-vertex view of a NeroGrasp MeshSpace."""

    def __init__(self, space, bound_m=0.0003):
        self.space = space
        self.base_meshes = space.base_meshes
        self.names = tuple(self.base_meshes)
        self.mutable = {name: slice(space.mounting_vertex_counts[name], None) for name in self.names}
        base = np.concatenate([self.base_meshes[name].vertices[self.mutable[name]].reshape(-1) for name in self.names]).astype(np.float32)
        self.low = base - float(bound_m)
        self.high = base + float(bound_m)
        self.discrete = np.zeros(len(base), dtype=bool)
        self.space_flat = spaces.Box(self.low, self.high, dtype=np.float32)

    def encode(self, value):
        return np.concatenate([np.asarray(value[name].vertices[self.mutable[name]], np.float32).reshape(-1) for name in self.names])

    def decode(self, value):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim == 1:
            return self._decode_one(array)
        if array.ndim != 2:
            raise ValueError(f"flat embodiment must have rank one or two, got {array.shape}")
        return [self._decode_one(row) for row in array]

    def _decode_one(self, value):
        if value.shape != self.low.shape:
            raise ValueError(f"flat embodiment has shape {value.shape}, expected {self.low.shape}")
        result, start = {}, 0
        for name in self.names:
            mesh = self.base_meshes[name]
            vertices = np.array(mesh.vertices, copy=True)
            count = vertices[self.mutable[name]].size
            vertices[self.mutable[name]] = value[start:start + count].reshape(-1, 3)
            result[name] = type(mesh)(vertices, np.array(mesh.faces, copy=True))
            start += count
        return result

    @staticmethod
    def project_normalized(value):
        return value.clamp(-1, 1)


class FlatCodesignVecEnv:
    """Torch-facing adapter with flat designs and optional categorical logits."""

    def __init__(self, env, *, device: str, control_mode: str, seed: int, rejection_reward: float,
                 mesh_bound_m: float | None = None):
        if control_mode not in ("native", "logits"):
            raise ValueError("control_mode must be 'native' or 'logits'")
        self.env = env
        self.num_envs = env.num_envs
        self.num_designs = getattr(env, "num_embodiments", self.num_envs)
        self.worlds_per_design = self.num_envs // self.num_designs
        self.device = device
        self.seed = int(seed)
        self.rejection_reward = float(rejection_reward)
        if hasattr(env.embodiment_space, "base_meshes"):
            codec_kwargs = {} if mesh_bound_m is None else {"bound_m": mesh_bound_m}
            self.design_codec = MeshSpaceCodec(env.embodiment_space, **codec_kwargs)
        elif all(hasattr(env.embodiment_space, name)
                 for name in ("flat_space", "to_index", "from_index")):
            self.design_codec = IndexedSpaceCodec(env.embodiment_space)
        else:
            self.design_codec = SpaceCodec(env.embodiment_space)
        self.embodiment_space = self.design_codec.space_flat
        base_embodiment = env.base_embodiment[0] if getattr(env, "base_embodiment_is_batched", False) else env.base_embodiment
        self.base_embodiment = self.design_codec.encode(base_embodiment)
        self.control_mode = control_mode
        self._native_action_space = env.single_action_space
        self._categorical = isinstance(self._native_action_space, spaces.MultiDiscrete)
        self._discrete = isinstance(self._native_action_space, spaces.Discrete)
        if self._categorical and control_mode == "logits":
            self._nvec = np.asarray(self._native_action_space.nvec, dtype=np.int64).reshape(-1)
            self.single_action_space = spaces.Box(-1.0, 1.0, shape=(int(self._nvec.sum()),), dtype=np.float32)
        elif isinstance(self._native_action_space, spaces.Box):
            low, high = self._native_action_space.low, self._native_action_space.high
            if not np.isfinite(low).all() or not np.isfinite(high).all():
                low, high = np.full(self._native_action_space.shape, -1.0), np.full(self._native_action_space.shape, 1.0)
            self.single_action_space = spaces.Box(low, high, dtype=np.float32)
        else:
            self.single_action_space = self._native_action_space
        original_obs = env.single_observation_space
        self.action_mask_slice = None
        self.supports_selective_reset = True
        autoreset_mode = getattr(env, "metadata", {}).get("autoreset_mode")
        self.autoresets_same_step = (
            getattr(autoreset_mode, "name", None) == "SAME_STEP" or
            str(autoreset_mode).endswith("SAME_STEP"))
        self.single_critic_observation_space = None
        if isinstance(original_obs, spaces.Dict) and "actor" in original_obs.spaces:
            if "critic" in original_obs.spaces and isinstance(original_obs["critic"], spaces.Box) and len(original_obs["critic"].shape) == 1:
                critic_box = original_obs["critic"]
                self.single_critic_observation_space = spaces.Box(
                    np.asarray(critic_box.low, np.float32), np.asarray(critic_box.high, np.float32), dtype=np.float32)
            original_obs = original_obs["actor"]
        elif isinstance(original_obs, spaces.Dict):
            keys = tuple(sorted(key for key in original_obs.spaces
                                if key != "action_mask"))
            if "action_mask" not in original_obs.spaces:
                raise TypeError("structured observations require an action_mask")
            low, high, width = [], [], 0
            for key in keys:
                space = original_obs[key]
                if isinstance(space, spaces.Box):
                    low.append(np.asarray(space.low).reshape(-1))
                    high.append(np.asarray(space.high).reshape(-1))
                elif isinstance(space, spaces.MultiBinary):
                    size = int(np.prod(space.shape))
                    low.append(np.zeros(size)); high.append(np.ones(size))
                elif isinstance(space, spaces.MultiDiscrete):
                    low.append(np.asarray(space.start).reshape(-1))
                    high.append((np.asarray(space.start) + np.asarray(space.nvec) - 1).reshape(-1))
                else:
                    raise TypeError(f"unsupported observation field {key}: {space}")
                width += int(np.prod(space.shape))
            mask_width = int(np.prod(original_obs["action_mask"].shape))
            low.append(np.zeros(mask_width)); high.append(np.ones(mask_width))
            self._observation_keys = keys
            self.action_mask_slice = (width, width + mask_width)
            self.supports_selective_reset = False
            original_obs = spaces.Box(
                np.concatenate(low).astype(np.float32),
                np.concatenate(high).astype(np.float32), dtype=np.float32)
        if not isinstance(original_obs, spaces.Box) or len(original_obs.shape) != 1:
            raise TypeError(f"baseline adapter requires a flat Box observation, got {original_obs}")
        self.single_observation_space = spaces.Box(
            np.asarray(original_obs.low, np.float32), np.asarray(original_obs.high, np.float32), dtype=np.float32)
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space
        self._last_obs = None
        self.last_critic_obs = None
        self.last_held_at_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.last_diverged = torch.zeros_like(self.last_held_at_goal)
        if hasattr(env, "last_success"):
            self.last_success = torch.zeros_like(self.last_held_at_goal)

    def project_design(self, value: torch.Tensor) -> torch.Tensor:
        return self.design_codec.project_normalized(value)

    def project_control(self, value: torch.Tensor) -> torch.Tensor:
        if self._discrete:
            return value
        value = value.clamp(-1, 1)
        if not self._categorical or self.control_mode != "logits":
            return value
        pieces, start = [], 0
        for count in self._nvec:
            logits = value[..., start:start + int(count)]
            hard = torch.full_like(logits, -1.0)
            hard.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
            pieces.append(hard + (logits - logits.detach()))
            start += int(count)
        return torch.cat(pieces, dim=-1)

    def validate_embodiment(self, embodiment):
        return self.env.validate_embodiment(self._decode_embodiment(embodiment))

    def reset(self, *, seed=None, options=None, embodiment=None):
        decoded = None if embodiment is None else self._decode_embodiment(embodiment)
        mask = np.ones(self.num_envs, dtype=bool)
        if options and "reset_mask" in options:
            mask = self._numpy(options["reset_mask"]).astype(bool)
            if mask.shape != (self.num_envs,):
                raise ValueError("reset_mask must have shape (num_envs,)")
        if mask.all():
            obs, rejected, info = self.env.reset(seed=seed, embodiment=decoded)
            rejected = self._numpy(rejected).astype(bool).reshape(-1)
            if rejected.any():
                # Multi-stage rejections (e.g. geometric rejector on pass 1, then convex
                # decomposition failure on pass 2) require iterating fallback until all
                # remaining non-base designs succeed.
                for _ in range(8):
                    fallback = self._fallback_designs(decoded, rejected)
                    obs, retry_rejected, info = self.env.reset(seed=seed, embodiment=fallback)
                    retry_rejected = self._numpy(retry_rejected).astype(bool).reshape(-1)
                    if not retry_rejected.any():
                        break
                    if np.all(rejected | ~retry_rejected):
                        raise RuntimeError("fallback reset rejected an already-fallback (base) embodiment")
                    rejected |= retry_rejected
                else:
                    raise RuntimeError("fallback reset did not converge after repeated embodiment rejections")
            self._last_obs = self._float_value(obs)
        else:
            obs, rejected, info = self._partial_reset(mask, decoded, seed)
        if hasattr(self, "last_success"):
            self.last_success.copy_(self._bool_tensor(self.env.last_success).reshape(-1))
        return self._tensor(self._last_obs), self._bool_tensor(rejected), info

    def step(self, actions, embodiment=None):
        native = self._decode_actions(actions)
        decoded = None if embodiment is None else self._decode_embodiment(embodiment)
        obs, reward, terminated, truncated, info = self.env.step(native, embodiment=decoded)
        if hasattr(self.env.embodiment_space, "base_meshes"):
            if hasattr(self.env, "_episode_over"):
                episode_over = bool(self.env._episode_over)
            else:
                finished = info.get("finished") if isinstance(info, dict) else None
                episode_over = finished is not None and bool(self._bool_tensor(finished).all())
            if episode_over:
                term_mask = getattr(self.env, "terminated_mask", None)
                if term_mask is not None:
                    terminated = self._bool_tensor(term_mask).reshape(-1)
                    truncated = ~terminated
                else:
                    terminated = self._bool_tensor(terminated).reshape(-1)
                    truncated = ~terminated
            else:
                terminated = torch.zeros_like(terminated)
                truncated = torch.zeros_like(truncated)
        self._last_obs = self._float_value(obs)
        held = getattr(self.env, "last_held_at_goal", getattr(self.env, "last_all_clean", None))
        diverged = getattr(self.env, "last_diverged", None)
        self.last_held_at_goal.zero_() if held is None else self.last_held_at_goal.copy_(self._bool_tensor(held).reshape(-1))
        self.last_diverged.zero_() if diverged is None else self.last_diverged.copy_(self._bool_tensor(diverged).reshape(-1))
        if hasattr(self, "last_success"):
            self.last_success.copy_(self._bool_tensor(self.env.last_success).reshape(-1))
        return (self._tensor(self._last_obs), self._tensor(reward), self._bool_tensor(terminated),
                self._bool_tensor(truncated), info)

    def render(self, env_idx=None):
        if env_idx is not None and "env_idx" in inspect.signature(self.env.render).parameters:
            return self.env.render(env_idx=int(env_idx))
        if env_idx is not None and hasattr(self.env, "envs"):
            return self.env.envs[int(env_idx)].render()
        frames = self.env.render()
        if isinstance(frames, (tuple, list)):
            return frames[0 if env_idx is None else int(env_idx)]
        return frames

    def close(self):
        self.env.close()

    def _decode_actions(self, actions):
        if hasattr(self.env, "device") and not self._categorical:
            value = actions if torch.is_tensor(actions) else torch.as_tensor(actions)
            return value.reshape((self.num_envs,) + self._native_action_space.shape).to(
                device=self.env.device, dtype=torch.float32)
        array = self._numpy(actions)
        if self._categorical:
            if self.control_mode == "native":
                return np.rint(array).astype(np.int32).reshape(self.num_envs, -1)
            result, start = [], 0
            for count in self._nvec:
                result.append(array[:, start:start + int(count)].argmax(axis=1))
                start += int(count)
            return np.stack(result, axis=1).astype(np.int32)
        return array.reshape((self.num_envs,) + self._native_action_space.shape).astype(self._native_action_space.dtype)

    def _fallback_designs(self, decoded, rejected):
        designs = list(decoded) if isinstance(decoded, list) else [decoded] * self.num_designs
        rejected = self.collapse_rejections(rejected)
        base = self.env.embodiment_space.base_meshes if hasattr(self.env.embodiment_space, "base_meshes") else self.env.base_embodiment
        return [base if rejected[i] else designs[i] for i in range(self.num_designs)]

    def _partial_reset(self, mask, decoded, seed):
        if self._last_obs is None:
            raise RuntimeError("partial reset requires an earlier full reset")
        if hasattr(self.env, "envs"):
            designs = list(decoded) if isinstance(decoded, list) else [decoded] * self.num_envs
            rejected = np.zeros(self.num_envs, dtype=bool)
            infos = []
            for i, selected in enumerate(mask):
                if not selected:
                    infos.append({})
                    continue
                item_seed = None if seed is None else int(seed) + i
                obs, rejected_i, info = self.env.envs[i].reset(seed=item_seed, embodiment=designs[i])
                if rejected_i:
                    obs, _, info = self.env.envs[i].reset(seed=item_seed, embodiment=self.env.base_embodiment)
                self._last_obs[i] = obs
                rejected[i] = rejected_i
                self.env._autoreset[i] = False
                if hasattr(self.env, "_current_embodiments"):
                    self.env._current_embodiments[i] = self.env.envs[i]._embodiment
                infos.append(info)
            return self._last_obs, rejected, {"per_env": infos}
        if (self.env.metadata.get("supports_partial_reset", False) or
                hasattr(self.env, "reset_mask")):
            obs, rejected, info = self.env.reset(
                seed=seed, options={"reset_mask": mask}, embodiment=decoded)
            retry = self._numpy(rejected).astype(bool).reshape(-1) & mask
            if retry.any():
                # As for full resets: a rejected world (e.g. a SoftWalker design that fails to settle) restarts from
                # the base design so it can be stepped; it is still reported as rejected.
                obs, again, info = self.env.reset(
                    seed=seed, options={"reset_mask": retry}, embodiment=self._fallback_designs(decoded, retry))
                if (self._numpy(again).astype(bool).reshape(-1) & retry).any():
                    raise RuntimeError("the base embodiment was rejected on partial-reset fallback")
            self._last_obs = self._float_value(obs)
            return self._last_obs, rejected, info
        if all(hasattr(self.env, name) for name in
               ("_prepare", "_build_batch", "_next_keys", "_states",
                "_coords", "_vobs", "_to_obs")):
            return self._partial_reset_jax(mask, decoded, seed)
        raise NotImplementedError(
            f"{type(self.env).__name__} does not support selective reset")

    def _partial_reset_jax(self, mask, decoded, seed):
        import jax
        import jax.numpy as jnp

        designs = list(decoded) if isinstance(decoded, list) else [decoded] * self.num_envs
        coords, rejected = self.env._prepare(designs)
        new_states, _, _ = self.env._build_batch(self.env._next_keys(), coords)
        choose = jnp.asarray(mask)
        self.env._states = jax.tree_util.tree_map(
            lambda new, old: jnp.where(choose.reshape((self.num_envs,) + (1,) * (new.ndim - 1)), new, old),
            new_states, self.env._states)
        self.env._coords = jnp.where(choose[:, None, None], coords, self.env._coords)
        self.env._autoreset[mask] = False
        self.env._returns[mask] = 0.0
        view = self.env._vobs(self.env._states)
        self._last_obs = self.env._to_obs(view)
        return self._last_obs, rejected & mask, {}

    def _tensor(self, value):
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32, device=self.device)

    def _bool_tensor(self, value):
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=torch.bool)
        return torch.as_tensor(np.array(value, copy=True), dtype=torch.bool, device=self.device)

    def _float_value(self, value):
        if isinstance(value, dict):
            if "actor" in value:
                if "critic" in value and self.single_critic_observation_space is not None:
                    self.last_critic_obs = self._tensor(value["critic"])
                else:
                    self.last_critic_obs = None
                value = value["actor"]
            else:
                self.last_critic_obs = None
                fields = [np.asarray(value[key], dtype=np.float32).reshape(self.num_envs, -1)
                          for key in self._observation_keys]
                fields.append(np.asarray(value["action_mask"], dtype=np.float32).reshape(self.num_envs, -1))
                value = np.concatenate(fields, axis=1)
        else:
            self.last_critic_obs = None
        if torch.is_tensor(value):
            return value.detach().to(device=self.device, dtype=torch.float32)
        return np.asarray(value, dtype=np.float32)

    def _decode_embodiment(self, value):
        decoded = self.design_codec.decode(value)
        if hasattr(self.env.embodiment_space, "base_meshes") and isinstance(decoded, dict):
            decoded = [{name: mesh.copy() for name, mesh in decoded.items()} for _ in range(self.num_designs)]
        return decoded

    def expand_design_values(self, value):
        if self.worlds_per_design == 1:
            return value
        if torch.is_tensor(value):
            return torch.repeat_interleave(value, self.worlds_per_design, dim=0)
        return np.repeat(np.asarray(value), self.worlds_per_design, axis=0)

    def collapse_returns(self, value):
        if self.worlds_per_design == 1:
            return value
        if torch.is_tensor(value):
            return value.reshape(self.num_designs, self.worlds_per_design, *value.shape[1:]).mean(dim=1)
        array = np.asarray(value)
        return array.reshape(self.num_designs, self.worlds_per_design, *array.shape[1:]).mean(axis=1)

    def collapse_rejections(self, value):
        if self.worlds_per_design == 1:
            return value
        if torch.is_tensor(value):
            return value.reshape(self.num_designs, self.worlds_per_design, *value.shape[1:]).any(dim=1)
        array = np.asarray(value)
        return array.reshape(self.num_designs, self.worlds_per_design, *array.shape[1:]).any(axis=1)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class SoftWalkerSpaceCodec(SpaceCodec):
    """Per-cell SoftWalker vector whose decode repairs every design into a structurally valid one."""

    def __init__(self, env):
        super().__init__(env.embodiment_space)
        from envs.soft_walker import embodiment as E
        self.grid, self.channels, self.dtype = tuple(env.grid), int(env.channels), env.embodiment_space.dtype
        slices, width = E._slices(env.dim), E.width(env.dim)
        for name in ("kind", "channel"):
            self.discrete[slices[name].start::width] = True

    def _decode_one(self, value):
        from .soft_walker_design import repair
        if value.shape != self.low.shape:
            raise ValueError(f"flat embodiment has shape {value.shape}, expected {self.low.shape}")
        return repair(value, self.grid, self.channels).astype(self.dtype)


class SoftWalkerFlatVecEnv(FlatCodesignVecEnv):
    """SoftWalker takes host NumPy actions within [-1, 1] and repaired per-cell designs."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.design_codec = SoftWalkerSpaceCodec(env)
        self.embodiment_space = self.design_codec.space_flat

    def _decode_actions(self, actions):
        array = np.clip(self._numpy(actions), -1.0, 1.0)
        return array.reshape((self.num_envs,) + self._native_action_space.shape).astype(self._native_action_space.dtype)


class NativeFlatCodesignVecEnv:
    """Zero-copy baseline interface for an already-flat native Torch environment."""

    def __init__(self, env, *, control_mode, rejection_reward):
        if control_mode not in ("native", "logits"):
            raise ValueError("control_mode must be 'native' or 'logits'")
        self.env = env
        self.num_envs = env.num_envs
        self.num_designs = env.num_envs
        self.worlds_per_design = 1
        self.device = env.device
        self.rejection_reward = float(rejection_reward)
        self.design_codec = SpaceCodec(env.embodiment_space)
        self.embodiment_space = self.design_codec.space_flat
        self.base_embodiment = self.design_codec.encode(env.base_embodiment)
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation, rejected, info = self.env.reset(**kwargs)
        return observation, rejected.reshape(-1), info

    def step(self, actions, embodiment=None):
        return self.env.step(actions, embodiment=embodiment)

    def project_design(self, value):
        return self.design_codec.project_normalized(value)

    @staticmethod
    def project_control(value):
        return value.clamp(-1, 1) if torch.is_tensor(value) else np.clip(value, -1, 1)

    @staticmethod
    def expand_design_values(value):
        return value

    @staticmethod
    def collapse_returns(value):
        return value

    @staticmethod
    def collapse_rejections(value):
        return value.reshape(-1)


def default_rejection_reward(env_id: str) -> float:
    if env_id.startswith("Microgrid"):
        return -1e9
    if env_id.startswith("LQStructure-"):
        return -1e6
    if env_id.startswith("SoftWalker"):
        return -10.0  # episode returns are O(1) body lengths of travel minus work
    return -250.0


def _pokemon_showdown_path():
    configured = os.environ.get("POKEMON_SHOWDOWN_PATH")
    sibling = Path(__file__).resolve().parents[2] / "pokemon-codesign-env" / "pokemon-showdown"
    showdown = Path(configured).expanduser() if configured else sibling
    if not (showdown / "pokemon-showdown").is_file():
        raise FileNotFoundError(
            "Pokémon Showdown checkout not found; set POKEMON_SHOWDOWN_PATH "
            f"or pass showdown_path (checked {showdown})")
    return showdown.resolve()


def make_baseline_vec(env_id: str, num_envs: int, *, device="cuda:0", seed=0,
                      is_eval_env=False, render_mode=None, control_mode="logits",
                      rejection_reward=None, mesh_bound_m=None, **cfg):
    if env_id.startswith(("BallCatcher-", "RotateInHand-Cube", "HandPickUp-")):
        raw, _ = make_vec(env_id, num_envs, device=device, seed=seed,
                             is_eval_env=is_eval_env, render_mode=render_mode, **cfg)
        reward = default_rejection_reward(env_id) if rejection_reward is None else rejection_reward
        wrapped = NativeFlatCodesignVecEnv(raw, control_mode=control_mode,
                                           rejection_reward=reward)
        return wrapped, wrapped.base_embodiment.copy()
    native = ({"device": device, "seed": seed}
        if env_id.startswith(("SolarCleaner", "TruckUnload", "NeroGraspAlt"))
        else {"seed": seed} if env_id.startswith("NeroGrasp")
        # SoftWalker trainers step their worlds concurrently (one CUDA stream per world). That is only stable when the
        # process is the only Warp process on its GPU, so evaluation runs serially on separate, shared eval GPUs.
        else {"device": device, "execution": "serial" if is_eval_env else "auto"} if env_id.startswith("SoftWalker") else {})
    if env_id.startswith("Pokenv") and "showdown_path" not in cfg:
        native["showdown_path"] = _pokemon_showdown_path()
    if env_id.startswith("Pokenv") and "embodiment_config" not in cfg:
        from pokemon_common import search_config
        path = cfg.get("showdown_path", native["showdown_path"])
        native["embodiment_config"] = search_config(env_id, path)
    actual_num_envs = num_envs * _nerograsp_object_count(env_id) if env_id.startswith("NeroGrasp") else num_envs
    build_cfg = {**cfg, **native}
    if render_mode is not None and (
        env_id.startswith(("NeroGraspAlt", "NeroGrasp-Tools")) or not env_id.startswith("NeroGrasp")
    ):
        build_cfg["render_mode"] = render_mode
    raw, _ = make_vec(env_id, actual_num_envs, is_eval_env=is_eval_env, **build_cfg)
    if env_id.startswith("NeroGrasp") and is_eval_env:
        # Evaluation renders the best and worst physical worlds, whose indices
        # are only known after rollout. Retain every compiled variant so those
        # worlds can be rendered with their actual fingertip geometry.
        raw.retain_all_render_models = True
    reward = default_rejection_reward(env_id) if rejection_reward is None else rejection_reward
    adapter = SoftWalkerFlatVecEnv if env_id.startswith("SoftWalker") else FlatCodesignVecEnv
    adapter_kwargs = {"mesh_bound_m": mesh_bound_m} if adapter is FlatCodesignVecEnv and mesh_bound_m is not None else {}
    wrapped = adapter(raw, device=device, control_mode=control_mode, seed=seed,
                      rejection_reward=reward, **adapter_kwargs)
    return wrapped, wrapped.base_embodiment.copy()


def _nerograsp_object_count(env_id):
    if env_id == "NeroGrasp-All":
        return 10
    return {"NeroGrasp-Small": 5, "NeroGrasp-Tiny": 4}.get(env_id, 1)
