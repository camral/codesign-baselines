import numpy as np
import pytest
import torch

from codesign_gym import make_vec
from common.codesign_adapter import (_pokemon_showdown_path,
                                      default_rejection_reward,
                                      make_baseline_vec)
from cmaes_evolutionary.gym_train import control, policy_size


ENV_IDS = (
    "LQStructure-LQR",
    "Microgrid-Random",
    "Network-Motorway",
    "Warehouse-Small",
)


def test_default_rejection_reward_penalizes_invalid_episode():
    assert default_rejection_reward("Network-Urban") == -250.0
    assert default_rejection_reward("Warehouse-Small") == -250.0


def test_pokemon_showdown_path_honors_environment_override(tmp_path, monkeypatch):
    showdown = tmp_path / "pokemon-showdown"
    showdown.mkdir()
    (showdown / "pokemon-showdown").write_text("#!/bin/sh\n")
    monkeypatch.setenv("POKEMON_SHOWDOWN_PATH", str(showdown))
    assert _pokemon_showdown_path() == showdown.resolve()


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("control_mode", ("native", "logits"))
def test_structured_environment_reset_and_step(env_id, control_mode):
    env, base = make_baseline_vec(env_id, 2, device="cpu", seed=7, control_mode=control_mode)
    try:
        observation, rejected, _ = env.reset(seed=7, embodiment=np.stack([base, base]))
        assert observation.shape == (2, env.single_observation_space.shape[0])
        assert rejected.shape == (2,)
        assert not bool(rejected.any())
        if control_mode == "logits":
            actions = env.project_control(torch.zeros((2, env.single_action_space.shape[0])))
        else:
            actions = torch.as_tensor(np.stack([env.single_action_space.sample()] * 2))
        next_observation, reward, terminated, truncated, _ = env.step(actions)
        assert next_observation.shape == observation.shape
        assert reward.shape == terminated.shape == truncated.shape == (2,)
        partial, partial_rejected, _ = env.reset(embodiment=np.stack([base, base]),
            options={"reset_mask": torch.tensor([True, False])})
        assert partial.shape == observation.shape
        assert not bool(partial_rejected.any())
    finally:
        env.close()


def test_categorical_logits_project_to_executed_one_hot():
    env, _ = make_baseline_vec("Network-Motorway", 2, device="cpu", control_mode="logits")
    try:
        logits = torch.linspace(-0.9, 0.9, env.single_action_space.shape[0]).repeat(2, 1).requires_grad_()
        projected = env.project_control(logits)
        start = 0
        for count in env._nvec:
            piece = projected[:, start:start + count]
            assert torch.all((piece == -1).sum(1) == count - 1)
            assert torch.all((piece == 1).sum(1) == 1)
            start += count
        projected.sum().backward()
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))
    finally:
        env.close()


def test_invalid_warehouse_layout_is_rejected_and_falls_back_to_base():
    env, base = make_baseline_vec("Warehouse-Small", 2, device="cpu", control_mode="logits")
    try:
        invalid = base.copy()
        invalid[:] = 0  # No shelves violates the minimum shelf count.
        _, rejected, _ = env.reset(embodiment=np.stack([invalid, invalid]))
        assert rejected.tolist() == [True, True]
        assert np.array_equal(np.asarray(env.env._design[0]), env.env.base_embodiment)
    finally:
        env.close()


def test_lq_single_dictionary_broadcasts_to_all_worlds():
    env, base = make_vec("LQStructure-LQR", 3)
    try:
        observation, rejected, _ = env.reset(seed=2, embodiment=base)
        assert observation.shape[0] == 3
        assert rejected.tolist() == [False, False, False]
    finally:
        env.close()


def test_cma_controller_pairs_one_weight_vector_with_each_world():
    observations = torch.randn(3, 7)
    weights = torch.randn(3, policy_size(7, 5))
    batched = control(observations, weights, 5)
    expected = torch.stack([control(observations[i:i + 1], weights[i], 5)[0] for i in range(3)])
    torch.testing.assert_close(batched, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BallCatcher requires CUDA")
def test_ball_catcher_native_adapter_supports_all_baseline_batching():
    env, base = make_baseline_vec("BallCatcher-Pitch", 2, device="cuda:0", seed=4,
        control_mode="logits")
    try:
        assert env.num_designs == env.num_envs == 2
        assert env.worlds_per_design == 1
        assert env.design_codec.encode(base).shape == base.shape
        observation, rejected, _ = env.reset(seed=4, embodiment=np.stack([base, base]))
        assert observation.shape[0] == 2 and rejected.shape == (2,)
        assert not bool(rejected.any())
        weights = np.zeros((2, policy_size(observation.shape[1], env.single_action_space.shape[0])), np.float32)
        expanded = env.expand_design_values(weights)
        assert expanded is weights
        weights_t = torch.as_tensor(expanded, device=env.device)
        actions = control(observation, weights_t, env.single_action_space.shape[0])
        observation, reward, terminated, truncated, _ = env.step(actions)
        assert observation.shape[0] == reward.shape[0] == 2
        assert env.collapse_returns(reward) is reward
        collapsed = env.collapse_rejections(torch.tensor([[False], [True]], device=env.device))
        assert collapsed.shape == (2,) and collapsed.tolist() == [False, True]
        from common.gym_batch import GymBatch
        batch = GymBatch(np.stack([base, base]), env=env)
        batch.reset()
        batch.step(torch.zeros((2, env.single_action_space.shape[0]), device=env.device))
        from fast_td3.codesign.joint import JointRollout
        rollout = JointRollout(env, seed=5)
        output = torch.zeros((2, len(base) + env.single_action_space.shape[0]), device=env.device)
        selected, rejected, _, _, _ = rollout.select_design(output)
        assert selected.all() and not rejected.any()
        _, _, reward, _, terminated, truncated, _ = rollout.step(output)
        assert reward.shape == terminated.shape == truncated.shape == (2,)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_rgb_render_after_step(env_id):
    env, base = make_baseline_vec(env_id, 2, device="cpu", seed=9, control_mode="logits",
        render_mode="rgb_array")
    try:
        env.reset(seed=9, embodiment=np.stack([base, base]))
        actions = env.project_control(torch.zeros((2, env.single_action_space.shape[0])))
        env.step(actions)
        frame = env.render(env_idx=0)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8
    finally:
        env.close()


def test_evaluation_ignores_completed_and_rejected_worlds():
    from types import SimpleNamespace
    from common.gym_batch import EvaluationAccumulator

    env = SimpleNamespace(num_envs=3, device="cpu")
    result = EvaluationAccumulator(env)
    result.reject(torch.tensor([False, False, True]), -250)
    result.add(torch.tensor([2., 3., 100.]), torch.tensor([True, False, False]),
               torch.zeros(3, dtype=torch.bool))
    result.add(torch.tensor([100., 4., 100.]), torch.zeros(3, dtype=torch.bool),
               torch.tensor([False, True, False]))
    assert result.complete
    assert result.summary()["episode_returns"] == [2., 7., -250.]
    assert result.summary()["episode_lengths"] == [1, 2, 0]


def test_controller_input_tracks_design_without_changing_environment_observation():
    from common.embodiment_conditioning import condition_observation, conditioned_observation_space
    from common.gym_batch import GymBatch

    env, base = make_baseline_vec("Warehouse-Small", 2, device="cpu", control_mode="native")
    try:
        denser = base.copy()
        denser[np.flatnonzero(denser == 0)[0]] = 1
        designs = np.stack([base, denser])
        raw, rejected, _ = env.reset(seed=7, embodiment=designs)
        assert not rejected.any()
        original_width = env.single_observation_space.shape[0]
        conditioned = condition_observation(env, raw, designs)
        assert raw.shape == (2, original_width)
        assert conditioned.shape == (2, original_width + len(base))
        torch.testing.assert_close(conditioned[:, :original_width], raw)
        torch.testing.assert_close(conditioned[:, original_width:],
                                   torch.as_tensor(2 * designs - 1, dtype=torch.float32))
        assert all(conditioned_observation_space(env).contains(row) for row in conditioned.numpy())
        batch = GymBatch(designs, env=env)
        torch.testing.assert_close(batch.reset()[:, original_width:],
                                   conditioned[:, original_width:])
    finally:
        env.close()
