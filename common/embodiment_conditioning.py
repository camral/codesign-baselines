"""Controller input construction; the Codesign Gym observation stays unchanged."""

import numpy as np
import torch
from gymnasium import spaces


def conditioned_observation_space(env):
    observation = env.single_observation_space
    design_width = len(env.base_embodiment)
    return spaces.Box(
        low=np.concatenate((np.asarray(observation.low, np.float32).reshape(-1),
                            -np.ones(design_width, np.float32))),
        high=np.concatenate((np.asarray(observation.high, np.float32).reshape(-1),
                             np.ones(design_width, np.float32))),
        dtype=np.float32,
    )


def condition_observation(env, observation, embodiment):
    """Append the installed design, scaled to [-1, 1], to each world's observation.

    Accept one design, one per design slot, or one per world. Design slots are
    expanded with the adapter's existing world-to-design assignment.
    """
    obs = torch.as_tensor(observation, dtype=torch.float32, device=env.device)
    design = torch.as_tensor(embodiment, dtype=torch.float32, device=env.device)
    if design.ndim == 1:
        design = design.unsqueeze(0)
    if design.shape[1:] != (len(env.base_embodiment),):
        raise ValueError("embodiment width does not match the controller design space")
    if design.shape[0] == 1:
        design = design.expand(env.num_envs, -1)
    elif design.shape[0] == env.num_envs:
        pass
    elif design.shape[0] == getattr(env, "num_designs", env.num_envs):
        design = env.expand_design_values(design)
    else:
        raise ValueError("embodiment count must match designs or worlds")
    space = env.embodiment_space
    cache_key = (id(space), obs.device, obs.dtype)
    cached = getattr(env, "_cond_bounds_cache", None)
    if cached is not None and cached[0] == cache_key:
        _, low, span, pos_span = cached
    else:
        low = torch.as_tensor(space.low, dtype=obs.dtype, device=obs.device)
        high = torch.as_tensor(space.high, dtype=obs.dtype, device=obs.device)
        span = high - low
        pos_span = span > 0
        span = span.clamp_min(1e-12)
        try:
            env._cond_bounds_cache = (cache_key, low, span, pos_span)
        except AttributeError:
            pass
    normalized = torch.where(pos_span, 2 * (design - low) / span - 1, torch.zeros_like(design))
    return torch.cat((obs, normalized.clamp(-1, 1)), dim=-1)
