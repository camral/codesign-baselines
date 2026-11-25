"""SoftWalker-Beam3D integration shared by the non-LLM baselines and LOKI."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("warp")

from envs.soft_walker import config as C
from envs.soft_walker import embodiment as E

from common.codesign_adapter import SoftWalkerFlatVecEnv, make_baseline_vec
from common.soft_walker_design import repair
from launch_comparison import command_for
from launch_environment_suite import DEFERRED_ENV_IDS, FAMILIES
from loki.collectors import make_collector

GRID, CHANNELS = C.grid("beam"), C.CHANNELS[3]


def _uniform(rng, n):
    low, high = E.bounds(3, int(np.prod(GRID)), CHANNELS)
    return low + rng.random((n, len(low))) * (high - low)


def test_only_beam3d_is_in_the_suite():
    assert FAMILIES["soft_walker"] == ("SoftWalker-Beam3D",)
    assert "SoftWalker-Beam3D" not in DEFERRED_ENV_IDS
    assert {"SoftWalker-MPM3D", "SoftWalker-HexFEM3D", "SoftWalker-Beam3DStairs", "SoftWalker-Spring2DGaps"} <= DEFERRED_ENV_IDS


def test_repair_makes_every_design_structurally_valid():
    rng = np.random.default_rng(0)
    designs = _uniform(rng, 600)
    designs[200:400, 0::10] = rng.random((200, 288)) * 0.6   # sparse, mostly fragmented bodies
    designs[400:450, 0::10] = 0.0                           # empty
    designs[450:500, 4:7] = 0.0                              # zero fibres
    designs[500:] *= 3.0                                     # far out of bounds
    low, high = E.bounds(3, int(np.prod(GRID)), CHANNELS)
    for design in designs:
        repaired = repair(design, GRID, CHANNELS)
        assert E.structural_errors(repaired, GRID, CHANNELS) == []
        assert np.all(repaired >= low) and np.all(repaired <= high)
        np.testing.assert_array_equal(repair(repaired, GRID, CHANNELS), repaired)


def test_repair_keeps_valid_designs():
    base = E.from_cells(C.base_props("beam"), 3, np.float64)
    np.testing.assert_array_equal(repair(base, GRID, CHANNELS), base)


def test_adapter_routes_device_codec_actions_and_rejection_reward():
    env, base = make_baseline_vec("SoftWalker-Beam3D", 2, device="cpu")
    try:
        assert isinstance(env, SoftWalkerFlatVecEnv)
        assert env.env.device == "cpu"
        assert env.rejection_reward == -10.0
        assert env.single_observation_space.shape == (5220,)
        assert env.single_action_space.shape == (16,)
        assert base.shape == env.embodiment_space.shape == (2880,)
        assert int(env.design_codec.discrete.sum()) == 2 * 288          # kind and channel of every cell
        assert env.env.metadata["supports_partial_reset"]
        decoded = env.design_codec.decode(_uniform(np.random.default_rng(1), 4))
        assert len(decoded) == 4 and all(bool(np.all(env.env.validate_embodiment(d))) for d in decoded)
        actions = env._decode_actions(torch.full((2, 16), 3.0))
        assert isinstance(actions, np.ndarray) and actions.dtype == np.float64 and actions.max() == 1.0
    finally:
        env.close()


def test_loki_collector_samples_valid_designs():
    with make_collector("SoftWalker-Beam3D", seed=11, device="cpu") as collector:
        for _ in range(16):
            design = collector.sample_embodiment()
            assert collector.space.contains(design)
            assert collector.validate_embodiment(design) == (True, "")
        assert collector.sampler_metadata()["grid"] == [8, 6, 6]


def test_soft_walker_evaluates_on_fewer_worlds_by_default(tmp_path):
    def worlds(env_id, method, evaluation_worlds=None):
        args = SimpleNamespace(hours=1, num_envs=16, env_id=env_id, project=None, group=None, entity=None,
                               evaluation_worlds=evaluation_worlds, loki_rollout_steps=640)
        command = command_for(args, method, 0, 0, tmp_path)
        return int(command[command.index("--evaluation-worlds") + 1])
    for method in ("fasttd3", "ppo_ngopt", "cmaes", "loki"):
        assert worlds("SoftWalker-Beam3D", method) == 8
        assert worlds("BallCatcher-Pitch", method) == 128
        assert worlds("SoftWalker-Beam3D", method, 4) == 4


def test_soft_walker_loki_scores_fewer_candidates_by_default(tmp_path):
    def candidates(env_id, loki_candidates=None):
        args = SimpleNamespace(hours=1, num_envs=16, env_id=env_id, project=None, group=None, entity=None,
                               evaluation_worlds=None, loki_rollout_steps=640, loki_candidates=loki_candidates)
        command = command_for(args, "loki", 0, 0, tmp_path)
        return int(command[command.index("--candidates") + 1]) if "--candidates" in command else None
    assert candidates("SoftWalker-Beam3D") == 16
    assert candidates("SoftWalker-Beam3D", 8) == 8
    assert candidates("BallCatcher-Pitch") is None



def test_soft_walker_fasttd3_uses_a_smaller_replay_by_default(tmp_path):
    def replay(env_id, replay_memory_gb=None):
        args = SimpleNamespace(hours=1, num_envs=16, env_id=env_id, project=None, group=None, entity=None,
                               evaluation_worlds=None, replay_memory_gb=replay_memory_gb)
        command = command_for(args, "fasttd3", 0, 0, tmp_path)
        return float(command[command.index("--replay-memory-gb") + 1])
    assert replay("SoftWalker-Beam3D") == 4.0
    assert replay("BallCatcher-Pitch") == 8.0
    assert replay("SoftWalker-Beam3D", 2.0) == 2.0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="SoftWalker physics takes minutes per lane on CPU")
def test_adapter_reset_step_partial_reset_and_render():
    env, base = make_baseline_vec("SoftWalker-Beam3D", 2, device="cuda:0", render_mode="rgb_array")
    try:
        observation, rejected, _ = env.reset(embodiment=np.stack([base, base]))
        assert observation.shape == (2, 5220) and not bool(rejected.any())
        observation, reward, terminated, truncated, _ = env.step(torch.zeros((2, 16), device="cuda:0"))
        assert observation.shape == (2, 5220) and reward.shape == terminated.shape == truncated.shape == (2,)
        observation, rejected, _ = env.reset(embodiment=np.stack([base, base]), options={"reset_mask": np.array([True, False])})
        assert observation.shape == (2, 5220) and not bool(rejected.any())
        frame = env.render(env_idx=0)
        assert frame.shape == (420, 720, 3) and frame.dtype == np.uint8
    finally:
        env.close()


def test_partial_reset_falls_back_to_base_for_rejected_worlds():
    from common.codesign_adapter import FlatCodesignVecEnv
    from gymnasium import spaces

    class Fake:
        metadata = {"supports_partial_reset": True}
        num_envs = 3
        embodiment_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        base_embodiment = np.zeros(2, np.float32)
        single_action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        single_observation_space = spaces.Box(-np.inf, np.inf, (1,), np.float32)

        def __init__(self):
            self.calls = []

        def reset(self, *, seed=None, options=None, embodiment=None):
            mask = np.ones(3, bool) if not options else np.asarray(options["reset_mask"])
            designs = embodiment if isinstance(embodiment, list) else [embodiment] * 3
            self.calls.append((mask.copy(), [None if d is None else np.asarray(d).copy() for d in designs]))
            # A design with first coordinate 0.9 "fails to settle".
            rejected = np.array([bool(m) and d is not None and float(np.asarray(d)[0]) == np.float32(0.9) for m, d in zip(mask, designs)])
            return np.zeros((3, 1), np.float32), rejected, {}

    fake = Fake()
    env = FlatCodesignVecEnv(fake, device="cpu", control_mode="native", seed=0, rejection_reward=-10.0)
    env.reset(embodiment=np.zeros((3, 2), np.float32))
    designs = np.array([[0.1, 0.0], [0.9, 0.0], [0.2, 0.0]], np.float32)
    _, rejected, _ = env.reset(embodiment=designs, options={"reset_mask": np.array([True, True, False])})
    assert rejected.tolist() == [False, True, False]            # still reported, so the method is penalised
    mask, fallback = fake.calls[-1]
    assert mask.tolist() == [False, True, False]                # only the rejected world is reset again ...
    np.testing.assert_array_equal(fallback[1], fake.base_embodiment)  # ... with the base design
