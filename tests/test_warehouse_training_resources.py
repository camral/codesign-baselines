"""Memory and population behavior that previously broke large Warehouse runs."""
import numpy as np
import pytest

from fast_td3.codesign.gym_train import fit_replay_capacity, DEFAULT_REPLAY_TRANSITIONS
from cmaes_evolutionary.main import CMAESConfig, build_optimizer
from cmaes_evolutionary.gym_train import decode_design, population_assignment, population_scores


def test_congested_replay_fits_budget_without_reducing_worlds():
    state_dim, action_dim, worlds = 528 + 438 + 1, 438 + 40, 16384
    budget = 8 * 2**30
    capacity = fit_replay_capacity(DEFAULT_REPLAY_TRANSITIONS, state_dim, action_dim, worlds, budget)
    assert worlds <= capacity < DEFAULT_REPLAY_TRANSITIONS
    assert capacity * 4 * (2 * state_dim + action_dim + 3) <= budget
    assert fit_replay_capacity(20000, state_dim, action_dim, worlds, budget) == 20000
    with pytest.raises(ValueError, match="one simulator batch"):
        fit_replay_capacity(DEFAULT_REPLAY_TRANSITIONS, state_dim, action_dim, worlds, 1024)


def test_population_repeats_cover_every_world_and_average_fitness():
    assignment = population_assignment(16384, 128)
    assert len(assignment) == 16384
    np.testing.assert_array_equal(np.bincount(assignment), np.full(128, 128))
    assignment = population_assignment(10, 3)
    scores, invalid = population_scores(np.arange(10), [False] * 9 + [True], assignment, 3)
    np.testing.assert_allclose(scores, [4.5, 4, 5])
    assert invalid.tolist() == [True, False, False]


def test_binary_cma_explores_executable_layouts():
    base = np.tile([0., 1.], 49)
    optimizer = build_optimizer(CMAESConfig(98, 16, 1000, num_workers=32,
        population_size=32, diagonal=True, binary_design=True, seed=0,
        design_initial=base, design_lower_bound=0, design_upper_bound=1))
    candidates = [optimizer.ask() for _ in range(32)]
    layouts = np.stack([decode_design(candidate.value["design"], True) for candidate in candidates])
    assert np.isin(layouts, [0, 1]).all()
    assert len(np.unique(layouts, axis=0)) > 1
    assert (layouts != base).any(axis=1).sum() >= 24
    for candidate, layout in zip(candidates, layouts):
        optimizer.tell(candidate, -float(layout.sum()))
    assert optimizer.num_tell == 32


def test_standalone_binary_cma_evaluates_and_returns_decoded_designs():
    from cmaes_evolutionary.main import run_cmaes

    seen = []
    def evaluate(design, policy):
        assert np.isin(design, [0, 1]).all()
        seen.append(design.copy())
        return float(design.sum())

    result = run_cmaes(evaluate, CMAESConfig(8, 2, 8, num_workers=4,
        population_size=4, diagonal=True, binary_design=True,
        design_initial=np.tile([0., 1.], 4), design_lower_bound=0, design_upper_bound=1))
    assert len(seen) == 9
    assert np.isin(result.design, [0, 1]).all()
