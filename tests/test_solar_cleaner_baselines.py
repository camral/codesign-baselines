"""SolarCleaner integration with the three common baseline runners."""

import numpy as np
import pytest
import torch

from common.codesign_adapter import make_baseline_vec
from launch_environment_suite import FAMILIES


SMALL = dict(
    use_graph=False, panel_rows=1, panel_cols=1,
    n_sensor_rows=1, n_sensor_cols=1,
)


def test_solar_cleaner_family_contains_all_presets():
    assert FAMILIES["solar_cleaner"] == (
        "SolarCleaner-Reach", "SolarCleaner-Stability", "SolarCleaner-Traverse")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SolarCleaner production path needs CUDA")
def test_solar_cleaner_adapter_stays_on_device_and_resets_per_world():
    env, base = make_baseline_vec(
        "SolarCleaner-Reach", 2, device="cuda:0", seed=7, **SMALL)
    try:
        designs = np.stack([base, base])
        obs, rejected, _ = env.reset(embodiment=designs)
        assert obs.is_cuda and not rejected.any()
        lo = env.embodiment_space.low
        changed = designs.copy()
        changed[0] = lo
        obs, rejected, _ = env.reset(
            embodiment=changed,
            options={"reset_mask": torch.tensor([True, False], device="cuda:0")})
        assert obs.is_cuda and not rejected.any()
        assert torch.allclose(env.env._current_embodiment[0], torch.as_tensor(lo, device="cuda:0"))
        assert torch.allclose(env.env._current_embodiment[1], torch.as_tensor(base, device="cuda:0"))
        actions = np.zeros((2, env.single_action_space.shape[0]), np.float32)
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert all(value.is_cuda for value in (obs, reward, terminated, truncated))
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SolarCleaner production path needs CUDA")
def test_solar_cleaner_adapter_forwards_render_index():
    env, _ = make_baseline_vec("SolarCleaner-Reach", 2, device="cuda:0", seed=7, **SMALL)
    try:
        env.env.render = lambda env_idx=None: env_idx
        assert env.render(env_idx=1) == 1
    finally:
        env.close()
