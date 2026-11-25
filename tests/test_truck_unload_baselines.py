"""TruckUnload-Single integration shared by the non-LLM baselines and LOKI."""

import numpy as np
import pytest
import torch

from common.codesign_adapter import make_baseline_vec
from launch_environment_suite import DEFERRED_ENV_IDS, FAMILIES
from loki.collectors import make_collector


def test_only_single_preset_is_in_the_suite():
    assert FAMILIES["truck_unload"] == ("TruckUnload-Single",)
    assert "TruckUnload-Multi" in DEFERRED_ENV_IDS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TruckUnload production path needs CUDA")
def test_adapter_reset_step_conditioning_and_render():
    env, base = make_baseline_vec(
        "TruckUnload-Single", 2, device="cuda:0", seed=7,
        render_mode="rgb_array")
    try:
        assert base.shape == (6,)
        assert env.single_observation_space.shape == (115,)
        assert env.single_action_space.shape == (16,)
        assert str(env.env.device) == "cuda:0"
        designs = np.stack([env.embodiment_space.low,
                            env.embodiment_space.high])
        observation, rejected, _ = env.reset(embodiment=designs)
        assert observation.shape == (2, 115)
        assert not bool(rejected.any())
        actions = torch.zeros((2, 16), device="cuda:0")
        observation, reward, terminated, truncated, info = env.step(actions)
        assert observation.shape == (2, 115)
        assert reward.shape == terminated.shape == truncated.shape == (2,)
        assert info["reward_per_robot"].shape == (2, 2)
        frame = env.render(env_idx=0)
        assert frame.shape == (480, 480, 3) and frame.dtype == np.uint8

        from common.embodiment_conditioning import condition_observation
        conditioned = condition_observation(env, observation, designs)
        assert conditioned.shape == (2, 121)
        torch.testing.assert_close(
            conditioned[:, -6:],
            torch.tensor([[-1.] * 6, [1.] * 6], device="cuda:0"))
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TruckUnload production path needs CUDA")
def test_loki_collector_uses_native_bounds_and_rejection_contract():
    with make_collector("TruckUnload-Single", seed=11, device="cuda:0") as collector:
        for _ in range(32):
            design = collector.sample_embodiment()
            assert collector.space.contains(design)
            assert collector.validate_embodiment(design) == (True, "")
        metadata = collector.sampler_metadata()
        assert metadata["max_offset_m"] == 1.0
