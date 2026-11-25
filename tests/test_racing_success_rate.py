"""Racing lap completion flows through the shared evaluation success metric."""

import pytest
import torch

pytest.importorskip("f1tenth_gym_jax")

from common.codesign_adapter import make_baseline_vec
from common.gym_batch import EvaluationAccumulator


def test_racing_lap_success_reaches_evaluation_accumulator(monkeypatch):
    import jax.numpy as jnp

    env, _ = make_baseline_vec("Racing-Spielberg", 3, device="cuda:0", seed=3)
    try:
        env.reset(seed=3)
        raw = env.env
        states = raw._states.replace(num_laps=jnp.asarray([[1], [1], [0]], dtype=raw._states.num_laps.dtype))
        obs = jnp.asarray(raw._last_obs)
        def fake_advance(*args):
            return states, obs, jnp.zeros(3, jnp.float32), jnp.array([False, True, True]), jnp.array([True, False, False]), jnp.array([False, False, True])
        monkeypatch.setattr(raw, "_advance", fake_advance)
        _, reward, terminated, truncated, info = env.step(torch.zeros((3, 2), device=env.device))
        assert info["success"].tolist() == [True, False, False]
        assert info["diverged"].tolist() == [False, False, True]
        assert env.last_success.tolist() == [True, False, False]
        assert env.last_diverged.tolist() == [False, False, True]
        assert not env.last_held_at_goal.any()
        result = EvaluationAccumulator(env)
        result.add(reward, terminated, truncated)
        assert result.complete
        summary = result.summary()
        assert summary["success_rate"] == pytest.approx(1 / 3)
        assert summary["divergence_rate"] == pytest.approx(1 / 3)
    finally:
        env.close()
