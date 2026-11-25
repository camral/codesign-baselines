"""Design-only CMA-ES and the generation bookkeeping of the FastTD3 + CMA-ES trainer."""

import numpy as np
import pytest
import torch

from fasttd3_cmaes.design_optimiser import build_design_optimiser, decode_design
from fasttd3_cmaes.gym_train import (
    AsymmetricReplay,
    NStepAccumulator,
    build_parser,
    fit_replay_capacity,
    generation_best,
    phase_complete,
    resolve_hp,
    slot_designs,
)


def test_continuous_candidates_are_designs_within_bounds():
    lower, upper = np.array([-2.0, 0.0, 1.0]), np.array([2.0, 1.0, 3.0])
    optimiser = build_design_optimiser(lower, upper, [0.0, 0.5, 2.0], population_size=8, seed=3)
    values = np.stack([optimiser.ask().value for _ in range(8)])
    assert values.shape == (8, 3)
    assert np.all(values >= lower) and np.all(values <= upper)


def test_same_seed_gives_same_population():
    def population():
        optimiser = build_design_optimiser(-1, 1, [0.0, 0.0], population_size=4, seed=11)
        return np.stack([optimiser.ask().value for _ in range(4)])
    np.testing.assert_array_equal(population(), population())


def test_design_search_improves_a_quadratic():
    target = np.array([0.6, -0.4, 0.2])
    optimiser = build_design_optimiser(-1, 1, np.zeros(3), population_size=8, seed=0)
    for _ in range(40):
        candidates = [optimiser.ask() for _ in range(8)]
        for candidate in candidates:
            optimiser.tell(candidate, float(np.sum((candidate.value - target) ** 2)))
    best = optimiser.provide_recommendation().value
    assert np.sum((best - target) ** 2) < 0.01


@pytest.mark.parametrize("diagonal", [True, False])
def test_binary_initial_logits_decode_to_the_reference(diagonal):
    reference = np.array([1, 0, 1, 1, 0, 0], dtype=float)
    optimiser = build_design_optimiser(0, 1, reference, population_size=4, binary=True, diagonal=diagonal)
    np.testing.assert_array_equal(decode_design(optimiser.parametrization.value, True), reference)
    assert set(np.unique(decode_design(optimiser.ask().value, True))) <= {0.0, 1.0}


@pytest.mark.parametrize("kwargs, message", [
    (dict(lower=[0, 0], upper=[1, 0], initial=[0, 0]), "below its upper"),
    (dict(lower=0, upper=1, initial=[0.5, 2.0]), "within the design bounds"),
    (dict(lower=[0, 0, 0], upper=1, initial=[0.5, 0.5]), "shape"),
    (dict(lower=0, upper=2, initial=[1, 0], binary=True), "binary"),
])
def test_invalid_design_spaces_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_design_optimiser(population_size=4, **kwargs)


def test_generation_best_ignores_rejected_candidates():
    designs = np.array([[0.0], [1.0], [2.0]])
    design, score = generation_best(designs, np.array([5.0, 9.0, 7.0]), np.array([False, True, False]))
    np.testing.assert_array_equal(design, [2.0])
    assert score == 7.0
    assert generation_best(designs, np.zeros(3), np.ones(3, dtype=bool)) == (None, None)


def test_slot_designs_repeat_candidates_and_replace_rejections():
    designs = np.array([[0.0, 0.0], [1.0, 1.0]])
    slots = slot_designs(designs, [False, True], np.array([9.0, 9.0]), np.arange(5) % 2)
    np.testing.assert_array_equal(slots[:, 0], [0, 9, 0, 9, 0])
    np.testing.assert_array_equal(designs[1], [1.0, 1.0])  # Inputs are not modified.


def test_training_phase_length():
    finished = torch.tensor([True, False])
    assert not phase_complete(10_000, None, finished)
    assert phase_complete(1, None, torch.tensor([True, True]))
    assert not phase_complete(4, 5, finished)
    assert phase_complete(5, 5, finished)


def test_nstep_accumulator_returns_discounts_and_bootstraps():
    gamma = 0.5
    replay = AsymmetricReplay(32, state_dim=2, action_dim=1, device="cpu", critic_state_dim=4)
    acc = NStepAccumulator(replay, num_envs=2, state_dim=2, action_dim=1, n_steps=3, gamma=gamma,
                           device="cpu", critic_state_dim=4)
    # Env 0 terminates at step 1 (after 2 steps); env 1 runs 4 steps and truncates at step 3.
    rewards = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([4.0, 4.0]),
        torch.tensor([99.0, 8.0]),
        torch.tensor([99.0, 16.0]),
    ]
    terms = [
        torch.tensor([False, False]),
        torch.tensor([True, False]),
        torch.tensor([False, False]),
        torch.tensor([False, False]),
    ]
    truncs = [
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        torch.tensor([False, True]),
    ]
    actives = [
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        torch.tensor([False, True]),
        torch.tensor([False, True]),
    ]
    for t in range(4):
        s = torch.full((2, 2), float(t))
        ns = torch.full((2, 2), float(t + 1))
        cs = torch.full((2, 4), float(t + 10))
        ncs = torch.full((2, 4), float(t + 11))
        a = torch.full((2, 1), float(t))
        acc.push(s, a, rewards[t], ns, terms[t], truncs[t], critic_state=cs, next_critic=ncs, active=actives[t])
    acc.flush_all()

    # Env 0 emits 2 transitions (t=0..1, t=1..1); env 1 emits 4 transitions (t=0..2, t=1..3, t=2..3, t=3..3)
    assert replay.count == 6
    rows = [(float(replay.data["state"][i, 0]), float(replay.data["next"][i, 0]),
             float(replay.data["critic_state"][i, 0]), float(replay.data["next_critic"][i, 0]),
             float(replay.data["reward"][i, 0]), float(replay.data["bootstrap"][i, 0]),
             float(replay.data["discount"][i, 0])) for i in range(replay.count)]
    # Env 0: t=0 -> next=2, R = 1 + 0.5*4 = 3, bootstrap=0, discount=0.25
    assert (0.0, 2.0, 10.0, 12.0, 3.0, 0.0, 0.25) in rows
    # Env 0: t=1 -> next=2, R = 4, bootstrap=0, discount=0.5
    assert (1.0, 2.0, 11.0, 12.0, 4.0, 0.0, 0.5) in rows
    # Env 1: t=0 -> next=3, R = 2 + 0.5*4 + 0.25*8 = 6, bootstrap=1, discount=0.125
    assert (0.0, 3.0, 10.0, 13.0, 6.0, 1.0, 0.125) in rows
    # Env 1: t=1 -> next=4, R = 4 + 0.5*8 + 0.25*16 = 12, bootstrap=1 (truncated), discount=0.125
    assert (1.0, 4.0, 11.0, 14.0, 12.0, 1.0, 0.125) in rows
    # Env 1: t=2 -> next=4, R = 8 + 0.5*16 = 16, bootstrap=1, discount=0.25
    assert (2.0, 4.0, 12.0, 14.0, 16.0, 1.0, 0.25) in rows
    # Env 1: t=3 -> next=4, R = 16, bootstrap=1, discount=0.5
    assert (3.0, 4.0, 13.0, 14.0, 16.0, 1.0, 0.5) in rows


def test_policy_snapshot_roundtrip_without_obs_normalization(tmp_path):
    from fast_td3.codesign.evaluation import load_policy, policy_snapshot
    from fast_td3.fast_td3 import Actor

    parser = build_parser()
    args = parser.parse_args([
        "--hours", "1", "--output-dir", str(tmp_path),
        "--no-obs-normalization", "--actor-hidden-dim", "32", "--critic-hidden-dim", "64",
        "--num-steps", "8", "--num-updates", "4", "--v-min", "0", "--v-max", "60",
        "--weight-decay", "0", "--cma-scale", "0.5", "--mesh-bound-m", "0.003",
        "--warmup-steps", "3000", "--no-use-cdq",
    ])
    hp = resolve_hp(args)
    assert hp["obs_normalization"] is False
    assert hp["num_steps"] == 8 and hp["num_updates"] == 4
    assert hp["v_min"] == 0.0 and hp["v_max"] == 60.0
    assert hp["weight_decay"] == 0.0 and hp["use_cdq"] is False
    assert args.cma_scale == 0.5 and args.mesh_bound_m == 0.003 and args.warmup_steps == 3000

    actor = Actor(10, 4, 2, hp["init_scale"], hp["actor_hidden_dim"],
                  std_min=hp["std_min"], std_max=hp["std_max"], device="cpu")
    payload = policy_snapshot(actor, None, env_id="Test-v0", seed=0,
                              state_dim=10, action_dim=4, design_dim=3, hp=hp, amp=False)
    path = tmp_path / "snap.pt"
    torch.save(payload, path)
    loaded, inference = load_policy(path, "cpu", compile=False)
    assert loaded["normalization"] is None
    x = torch.randn(2, 10)
    torch.testing.assert_close(inference(x, deterministic=True, update=False), actor(x))


def test_fit_replay_capacity_accounts_for_asymmetric_critic():
    sym = fit_replay_capacity(1_000_000, state_dim=10, action_dim=4, num_envs=4, max_bytes=10_000)
    asym = fit_replay_capacity(1_000_000, state_dim=10, action_dim=4, num_envs=4, max_bytes=10_000, critic_state_dim=20)
    assert asym < sym


def test_prefix_obs_normalization_leaves_design_tail_unchanged_and_roundtrips(tmp_path):
    from fast_td3.codesign.evaluation import policy_snapshot
    from fast_td3.fast_td3 import Actor
    from fasttd3_cmaes.normalization import PrefixEmpiricalNormalization, load_policy

    parser = build_parser()
    args = parser.parse_args([
        "--hours", "1", "--output-dir", str(tmp_path),
        "--obs-normalization", "--no-normalize-design", "--max-grad-norm", "10.0",
        "--actor-hidden-dim", "32", "--critic-hidden-dim", "32",
    ])
    hp = resolve_hp(args)
    assert hp["obs_normalization"] is True
    assert hp["normalize_design"] is False
    assert hp["max_grad_norm"] == 10.0
    assert hp["use_grad_norm_clipping"] is True

    obs_dim, design_dim = 6, 4
    state_dim = obs_dim + design_dim
    norm = PrefixEmpiricalNormalization(obs_dim, "cpu")
    # Constant design tail during warmup must not affect tail output after perturbation.
    warmup = torch.cat((torch.randn(64, obs_dim) * 0.05 + 1.0, torch.full((64, design_dim), -1.0)), dim=-1)
    norm(warmup, update=True)
    perturbed = torch.cat((torch.randn(8, obs_dim) * 0.05 + 1.0, torch.full((8, design_dim), 0.75)), dim=-1)
    out = norm(perturbed, update=False)
    torch.testing.assert_close(out[:, obs_dim:], perturbed[:, obs_dim:])

    actor = Actor(state_dim, 3, 2, hp["init_scale"], hp["actor_hidden_dim"],
                  std_min=hp["std_min"], std_max=hp["std_max"], device="cpu")
    payload = policy_snapshot(actor, norm, env_id="Test-v0", seed=0,
                              state_dim=state_dim, action_dim=3, design_dim=design_dim, hp=hp, amp=False)
    path = tmp_path / "prefix_snap.pt"
    torch.save(payload, path)
    loaded, inference = load_policy(path, "cpu", compile=False)
    assert isinstance(inference.normalization, PrefixEmpiricalNormalization)
    assert inference.normalization.obs_dim == obs_dim
    torch.testing.assert_close(
        inference(perturbed, deterministic=True, update=False),
        actor(norm(perturbed, update=False)),
    )




def test_initial_condition_designs():
    from fasttd3_cmaes.gym_train import initial_condition_designs

    # Mimic HandPickUp: 5 pad cells in [0, 1] and 2 link offsets in [-1, 1], base all 0.
    low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    base = np.zeros(7, dtype=np.float32)

    d_cma, t_cma, k_cma = initial_condition_designs("cmaes", low, high, base, 4, seed=7)
    np.testing.assert_allclose(d_cma, base)
    assert t_cma.shape == (4, 7) and k_cma == "initial_candidate"

    d_mid, t_mid, k_mid = initial_condition_designs("mid", low, high, base, 4, seed=7)
    np.testing.assert_allclose(d_mid, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(t_mid, np.repeat(d_mid[None], 4, axis=0))
    assert k_mid == "fixed_mid"

    d_r1, t_r1, k_r1 = initial_condition_designs("random", low, high, base, 4, seed=7)
    d_r2, _, _ = initial_condition_designs("random", low, high, base, 4, seed=7)
    d_r3, _, _ = initial_condition_designs("random", low, high, base, 4, seed=8)
    np.testing.assert_allclose(d_r1, d_r2)
    assert not np.allclose(d_r1, d_r3)
    assert np.all(d_r1 >= low) and np.all(d_r1 <= high)
    assert t_r1.shape == (4, 7) and k_r1 == "fixed_random"

    d_cmid, t_cmid, k_cmid = initial_condition_designs("cmaes", low, high, base, 4, seed=7, cma_init_mid=True)
    np.testing.assert_allclose(d_cmid, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32))
    assert t_cmid.shape == (4, 7) and k_cmid == "initial_candidate"


def test_mix_exploration_designs_and_correlated_exploration():
    from fasttd3_cmaes.gym_train import CorrelatedExploration, mix_exploration_designs

    low = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    base = np.zeros(3, dtype=np.float32)
    designs = np.repeat(base[None], 10, axis=0)
    rng = np.random.default_rng(42)
    mixed = mix_exploration_designs(designs, 0.5, low, high, base, rng)
    assert mixed.shape == (10, 3)
    assert np.all(mixed >= low) and np.all(mixed <= high)
    # Exactly 5 slots replaced, 5 untouched at base.
    unchanged = np.all(mixed == base, axis=1)
    assert unchanged.sum() == 5
    np.testing.assert_array_equal(designs, 0.0)

    # Zero frac returns untouched copy.
    np.testing.assert_array_equal(mix_exploration_designs(designs, 0.0, low, high, base, rng), designs)

    # CorrelatedExploration produces stationary ~N(0, 1) with positive lag-1 autocorrelation.
    torch.manual_seed(0)
    ar = CorrelatedExploration(num_envs=256, action_dim=8, max_rho=0.9, device="cpu")
    dones = torch.ones(256, dtype=torch.bool)
    e0 = ar.sample(dones).clone()
    e1 = ar.sample(torch.zeros(256, dtype=torch.bool)).clone()
    assert abs(float(e0.std()) - 1.0) < 0.1
    assert abs(float(e1.std()) - 1.0) < 0.1
    corr = float((e0 * e1).mean())
    assert 0.25 < corr < 0.65  # E[rho] = 0.45

    # Reset clears correlation with previous step.
    e2 = ar.sample(torch.ones(256, dtype=torch.bool)).clone()
    assert abs(float((e1 * e2).mean())) < 0.15


def test_design_smoothness_penalty_and_selection():
    from fasttd3_cmaes.gym_train import design_smoothness_penalty, extract_pad_grids

    class DummyEnv:
        _pads = {"col0": [0, 4], "nrow": [2, 2], "ncol": [2, 2]}

    grids = extract_pad_grids(DummyEnv())
    assert grids == [(0, 2, 2), (4, 2, 2)]
    assert extract_pad_grids(object()) == []

    # 8 pad cells (two 2x2 pads) + 2 trailing link dims (must be ignored by smoothness).
    flat = np.array([0.5] * 8 + [-1.0, 1.0], dtype=np.float32)
    checker = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    ramp = np.array([0.0, 0.5, 0.0, 0.5, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0], dtype=np.float32)
    designs = np.stack([flat, checker, ramp])

    penalties = design_smoothness_penalty(designs, grids)
    assert penalties[0] == pytest.approx(0.0, abs=1e-7)
    assert penalties[1] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < penalties[2] < 0.2

    # When task scores are tied, selection_scores picks the smooth design while returning its raw score.
    scores = np.array([9.0, 9.05, 9.02], dtype=np.float32)
    sel = scores - 2.0 * penalties
    best_d, best_raw, best_idx = generation_best(designs, scores, [False, False, False], selection_scores=sel)
    assert best_idx == 0
    assert best_raw == pytest.approx(9.0)
    np.testing.assert_array_equal(best_d, flat)


def test_cma_min_std_and_multiscale_gaussian_and_burst_exploration(tmp_path):
    from fasttd3_cmaes.gym_train import CorrelatedExploration, mix_exploration_designs

    # 1. CMA-ES min_std prevents step-size collapse on a quadratic bowl.
    opt = build_design_optimiser(
        -0.01, 0.01, np.zeros(24), population_size=16, seed=0, scale=6.0, min_std=0.5,
    )
    for _ in range(15):
        cands = [opt.ask() for _ in range(16)]
        for c in cands:
            opt.tell(c, float(np.sum(c.value ** 2)))
    final_std = float(np.std([c.value for c in cands], axis=0).mean())
    assert final_std > 0.0012  # >= ~0.5 * (0.02 / 6) = 0.00167

    # 2. Multi-scale Gaussian mesh exploration respects bounds and max_unique.
    lo, hi, base = np.full(24, -0.01, np.float32), np.full(24, 0.01, np.float32), np.zeros(24, np.float32)
    mixed = mix_exploration_designs(
        np.repeat(base[None], 32, axis=0), 0.5, lo, hi, base,
        np.random.default_rng(7), gaussian=True, max_unique=8, gaussian_scale=1.5,
    )
    assert np.all(mixed >= lo) and np.all(mixed <= hi)
    changed = mixed[~np.all(mixed == base, axis=1)]
    assert len(changed) == 16
    assert len(np.unique(changed, axis=0)) <= 8

    # 3. Delayed-onset masked CorrelatedExploration runs low noise before t_switch and masked bursts after.
    torch.manual_seed(0)
    ar = CorrelatedExploration(
        num_envs=128, action_dim=8, max_rho=0.9, device="cpu",
        burst_frac=1.0, dim_prob=0.5, std_min=0.01, std_max=0.8,
    )
    scales = torch.full((128, 1), 0.5)
    e0 = ar.sample(torch.ones(128, dtype=torch.bool), scales)
    # At t=0 every burst env is pre-burst (std_min=0.01).
    assert float(e0.std()) < 0.03
    d0 = torch.zeros(128, dtype=torch.bool)
    for _ in range(70):
        e_late = ar.sample(d0, scales)
    # By t=70 all burst envs are in-burst with partial dim masking.
    assert float(e_late.std()) > 0.15
    assert int((e_late == 0.0).sum()) > 0
    assert bool((ar.dim_mask.sum(dim=1) >= 1.0).all())

    # 3b. Override burst zero-centers masked dims instead of adding onto saturated actions.
    torch.manual_seed(1)
    ar_ov = CorrelatedExploration(
        num_envs=128, action_dim=8, max_rho=0.92, device="cpu",
        burst_frac=1.0, dim_prob=0.5, override=True, std_min=0.01, std_max=0.8,
    )
    det_sat = torch.ones((128, 8))
    ar_ov.apply(det_sat, torch.ones(128, dtype=torch.bool), scales)
    for _ in range(55):
        a_ov = ar_ov.apply(det_sat, d0, scales)
    # Unmasked dims remain ~1.0 while masked dims escape the +1.0 boundary into negative values.
    assert int((a_ov < 0.0).sum()) > 50

    # 4. Parser accepts all new co-design & exploration flags.
    args = build_parser().parse_args([
        "--hours", "0.1", "--output-dir", str(tmp_path),
        "--cma-scale", "6.0", "--cma-min-std", "0.50",
        "--warmup-diverse", "--explore-design-scale", "1.25",
        "--noise-burst-frac", "0.35", "--noise-dim-prob", "0.6",
        "--noise-burst-override",
    ])
    assert args.cma_min_std == 0.50 and args.warmup_diverse is True
    assert args.explore_design_scale == 1.25
    assert args.noise_burst_frac == 0.35 and args.noise_dim_prob == 0.6
    assert args.noise_burst_override is True
