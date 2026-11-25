"""One-cluster LOKI PPO, scoring, replacement, and resume contracts."""

import json
import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from common.codesign_adapter import SpaceCodec
from loki.collectors.storage import file_digest, write_json
from loki.controller import (MorphologyTransformerExtractor,
                             controller_policy_kwargs)
from loki.single_cluster import (ElitePool, SingleClusterConfig,
                                 TokenConditioner, UpstreamLearningRate,
                                 create_cluster_runtime,
                                 train_single_cluster)
from loki.vae import CorpusTokenSchema

from test_loki_vae import TypedEnv, make_corpus


def make_clusters(corpus, root, count=40):
    (root / "clusters").mkdir(parents=True)
    np.save(root / "clusters" / "0.npy", np.arange(count, dtype=np.int64))
    write_json(root / "manifest.json", {
        "status": "complete",
        "design_count": count,
        "clustering_config": {"n_clusters": 1},
        "corpus_identity": {
            "manifest_sha256": file_digest(corpus / "manifest.json"),
            "constants_sha256": file_digest(corpus / "constants.npz"),
        },
    })


def tiny_policy_kwargs(schema, observation_dim):
    value = controller_policy_kwargs(schema, observation_dim)
    value["features_extractor_kwargs"].update(
        hidden_dim=8, heads=2, layers=1, ffn_dim=16)
    return value


class ToyCodesignVec:
    """Torch vector environment with the adapter contract used by LOKI."""

    def __init__(self, n):
        source = TypedEnv()
        self.num_envs = self.num_designs = n
        self.worlds_per_design = 1
        self.device = "cpu"
        self.rejection_reward = -250.0
        self.design_codec = SpaceCodec(source.embodiment_space)
        self.base_embodiment = self.design_codec.encode(source.base_embodiment)
        self.single_observation_space = gym.spaces.Box(
            -10, 10, (2,), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(
            -1, 1, (1,), dtype=np.float32)
        self.steps = torch.zeros(n, dtype=torch.long)
        self.designs = torch.zeros((n, len(self.base_embodiment)))
        self.closed = False

    def expand_design_values(self, values):
        return values

    def reset(self, embodiment, seed=None, options=None):
        mask = (torch.ones(self.num_envs, dtype=torch.bool) if options is None
                else torch.as_tensor(options["reset_mask"]).reshape(-1).bool())
        designs = torch.as_tensor(embodiment, dtype=torch.float32)
        if designs.ndim == 1:
            designs = designs.expand(self.num_envs, -1)
        self.designs[mask] = designs[mask]
        self.steps[mask] = 0
        obs = torch.stack((self.designs[:, -2], self.steps.float()), dim=-1)
        return obs, torch.zeros((self.num_envs, 1), dtype=torch.bool), {}

    def step(self, action):
        self.steps += 1
        obs = torch.stack((self.designs[:, -2], self.steps.float()), dim=-1)
        reward = self.designs[:, -2] + 0.01 * torch.as_tensor(action).reshape(-1)
        terminated = self.steps >= 2
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.closed = True


def test_released_single_cluster_defaults_and_schedule():
    config = SingleClusterConfig()
    assert (config.elite_size, config.search_frequency,
            config.candidates_per_search, config.replacements_per_search,
            config.candidate_horizon) == (20, 2, 128, 2, 200)
    assert (config.rollout_steps, config.ppo_epochs,
            config.ppo_batch_size) == (2560, 8, 5120)
    schedule = UpstreamLearningRate(3e-4, 20)
    assert math.isclose(schedule(0), 3e-5)
    schedule.iteration = 5
    assert math.isclose(schedule(0), 3e-4 * 0.5 * (1 + math.cos(math.pi / 4)))


def test_literal_replacement_uses_best_candidates_and_worst_training_slots():
    pool = ElitePool([10, 11, 12])
    pool.record(0, 5)
    pool.record(1, 1)
    pool.record(2, 3)
    events = pool.replace_from_candidates(
        [20, 21, 22], [0.0, -4.0, -3.0], [False, False, True], 2)
    # Literal LOKI replaces even though both valid candidates are worse.
    assert [event["slot"] for event in events] == [1, 2]
    assert pool.design_ids.tolist() == [10, 20, 21]
    assert pool.training_scores().tolist() == [5.0, 1e6, 1e6]


def test_corpus_normalization_and_transformer_features(tmp_path):
    corpus = tmp_path / "corpus"
    make_corpus(corpus)
    schema = CorpusTokenSchema.from_corpus(corpus)
    env = ToyCodesignVec(2)
    conditioner = TokenConditioner(env, schema)
    designs = torch.stack((
        torch.as_tensor(env.design_codec.low),
        torch.as_tensor(env.design_codec.high)))
    normalized = conditioner.normalize_designs(designs)
    torch.testing.assert_close(normalized[0], -torch.ones(6))
    torch.testing.assert_close(normalized[1], torch.ones(6))
    policy = controller_policy_kwargs(schema, 2)
    assert policy["share_features_extractor"] is False
    extractor = MorphologyTransformerExtractor(
        conditioner.observation_space, observation_dim=2,
        token_layout=[(g.kind, g.count, g.width) for g in schema.groups],
        hidden_dim=8, heads=2, layers=1, ffn_dim=16)
    conditioned = torch.cat((torch.zeros((2, 2)), normalized), dim=-1)
    features = extractor(conditioned)
    assert features.shape == (2, 8)
    assert not torch.equal(features[0], features[1])


def test_pool_replacement_resets_only_worlds_using_changed_elite_slots(
        tmp_path, monkeypatch):
    from loki import single_cluster as module

    corpus, clusters = tmp_path / "corpus", tmp_path / "clusters"
    make_corpus(corpus)
    make_clusters(corpus, clusters)
    instances = []

    def make_env(_env_id, n, **_kwargs):
        env = ToyCodesignVec(n)
        instances.append(env)
        return env, env.base_embodiment

    monkeypatch.setattr(module, "make_baseline_vec", make_env)
    monkeypatch.setattr(module, "controller_policy_kwargs", tiny_policy_kwargs)
    runtime = create_cluster_runtime(
        "typed", corpus, clusters, 0, num_envs=2, device="cpu", seed=4,
        config=SingleClusterConfig(
            elite_size=2, search_frequency=1, candidates_per_search=4,
            replacements_per_search=1, candidate_horizon=3,
            rollout_steps=4, ppo_epochs=1, ppo_batch_size=4))
    try:
        runtime.model._last_obs = runtime.normalizer.reset()
        runtime.model._last_episode_starts = np.zeros(2, dtype=bool)
        env = instances[0]
        env.steps[:] = torch.tensor([7, 9])
        runtime.vector.episode_returns[:] = torch.tensor([11.0, 22.0])
        runtime.vector.slot_returns[:] = torch.tensor([33.0, 44.0])
        runtime.normalizer.returns[:] = np.array([55.0, 66.0])

        old_ids = runtime.pool.design_ids.copy()
        runtime.pool.design_ids[0] = next(
            value for value in runtime.store.cluster_ids
            if value not in old_ids)
        runtime.install_pool([0])

        assert env.steps.tolist() == [0, 9]
        torch.testing.assert_close(
            runtime.vector.episode_returns, torch.tensor([0.0, 22.0]))
        torch.testing.assert_close(
            runtime.vector.slot_returns, torch.tensor([0.0, 44.0]))
        assert runtime.normalizer.returns.tolist() == [0.0, 66.0]
        assert runtime.model._last_episode_starts.tolist() == [True, False]
    finally:
        runtime.close()


def test_one_cluster_pipeline_scores_replaces_checkpoints_and_resumes(
        tmp_path, monkeypatch):
    from loki import single_cluster as module

    corpus, clusters, output = (tmp_path / "corpus", tmp_path / "clusters",
                                tmp_path / "run")
    make_corpus(corpus)
    make_clusters(corpus, clusters)
    instances = []

    def make_env(_env_id, n, **_kwargs):
        env = ToyCodesignVec(n)
        instances.append(env)
        return env, env.base_embodiment

    monkeypatch.setattr(module, "make_baseline_vec", make_env)
    monkeypatch.setattr(module, "controller_policy_kwargs", tiny_policy_kwargs)
    config = SingleClusterConfig(
        elite_size=2, search_frequency=1, candidates_per_search=4,
        replacements_per_search=1, candidate_horizon=3,
        rollout_steps=4, ppo_epochs=1, ppo_batch_size=4,
        max_iterations=1)
    first = train_single_cluster(
        "typed", corpus, clusters, 0, output, num_envs=2,
        device="cpu", seed=4, config=config)
    assert first["counters"] == {
        "ppo_frames": 8, "candidate_frames": 12, "search_rounds": 1,
        "incomplete_search_rounds": 0, "candidate_rejections": 0,
        "elite_replacements": 1}
    assert (output / "checkpoints/000001/policy.zip").is_file()
    assert (output / "checkpoints/000001/vecnormalize.pkl").is_file()
    runtime = torch.load(output / "checkpoints/000001/runtime.pt",
                         map_location="cpu", weights_only=False)
    assert runtime["iteration"] == 1
    assert len(runtime["pool"]["design_ids"]) == 2
    history = [json.loads(row) for row in
               (output / "history.jsonl").read_text().splitlines()]
    assert len(history) == 1 and len(history[0]["replacements"]) == 1

    resumed = train_single_cluster(
        "typed", corpus, clusters, 0, output, num_envs=2,
        device="cpu", seed=4,
        config=SingleClusterConfig(**{
            **config.__dict__, "max_iterations": 2}), resume=True)
    assert resumed["iterations"] == 2
    assert resumed["counters"]["search_rounds"] == 2
    assert resumed["counters"]["elite_replacements"] == 2
    assert resumed["logical_design_slots"] == 2
    assert resumed["physical_worlds"] == 2
    assert (output / "checkpoints/000002/runtime.pt").is_file()
    with pytest.raises(ValueError, match="Resume task, seed, environments"):
        train_single_cluster(
            "different-task", corpus, clusters, 0, output, num_envs=2,
            device="cpu", seed=4,
            config=SingleClusterConfig(**{
                **config.__dict__, "max_iterations": 3}), resume=True)
    assert all(env.closed for env in instances)


def test_evaluation_selection_uses_fresh_distinct_designs_and_stable_ties():
    from loki.single_cluster import EvaluationSelection

    rankings = EvaluationSelection()
    rankings.update(0, 3, [10, 10, 11], [9.0, 8.0, 7.0],
                    [False, False, False], elapsed_seconds=12)
    rankings.update(1, 3, [20, 21, 22], [8.0, 6.0, 0.0],
                    [False, False, True], elapsed_seconds=13)
    selected = rankings.select(2, cluster_ids=[0, 1])
    assert selected["cluster_id"] == 0
    assert selected["policy_version"] == 3
    assert selected["design_ids"].tolist() == [10, 11]
    assert selected["scores"].tolist() == [9.0, 7.0]
    assert selected["cluster_score"] == 8.0

    restored = EvaluationSelection.from_state_dict(rankings.state_dict())
    assert restored.select(2, cluster_ids=[0, 1])["cluster_id"] == 0
    rankings.update(1, 3, [20, 21], [9.0, 7.0], [False, False])
    # Equal means choose the smaller cluster ID deterministically.
    assert rankings.select(2, cluster_ids=[0, 1])["cluster_id"] == 0


def test_candidate_scoring_chunks_stop_at_deadline(tmp_path, monkeypatch):
    from loki import single_cluster as module
    from loki.single_cluster import CandidateScorer

    corpus = tmp_path / "corpus"
    make_corpus(corpus)
    schema = CorpusTokenSchema.from_corpus(corpus)
    instances = []

    def make_env(_env_id, n, **_kwargs):
        env = ToyCodesignVec(n)
        instances.append(env)
        return env, env.base_embodiment

    class Model:
        @staticmethod
        def predict(observation, deterministic=False):
            assert not deterministic
            return np.zeros((len(observation), 1), np.float32), None

    class Normalizer:
        @staticmethod
        def normalize_obs(observation):
            return observation

    class Deadline:
        expired = False

    deadline = Deadline()
    ticks = 0

    def progress():
        nonlocal ticks
        ticks += 1
        deadline.expired = ticks >= 4

    monkeypatch.setattr(module, "make_baseline_vec", make_env)
    scorer = CandidateScorer("typed", schema, device="cpu", seed=3)
    designs = np.stack([ToyCodesignVec(1).base_embodiment] * 4)
    result = scorer.score(
        Model(), Normalizer(), designs, round_index=0, horizon=3,
        batch_size=2, deadline=deadline, progress=progress)
    assert not result.complete
    assert result.frames == 6
    assert np.isfinite(result.scores[:2]).all()
    assert np.isneginf(result.scores[2:]).all()
    assert result.rejected[2:].all()
    assert all(env.closed for env in instances)


def test_candidate_scoring_does_not_reset_same_step_autoreset_envs(
        tmp_path, monkeypatch):
    from loki import single_cluster as module
    from loki.single_cluster import CandidateScorer

    corpus = tmp_path / "corpus"
    make_corpus(corpus)
    schema = CorpusTokenSchema.from_corpus(corpus)
    instances = []

    class SameStepToyCodesignVec(ToyCodesignVec):
        autoresets_same_step = True

        def __init__(self, n):
            super().__init__(n)
            self.reset_calls = 0

        def reset(self, embodiment, seed=None, options=None):
            self.reset_calls += 1
            return super().reset(embodiment, seed=seed, options=options)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            self.steps[terminated | truncated] = 0
            return obs, reward, terminated, truncated, info

    def make_env(_env_id, n, **_kwargs):
        env = SameStepToyCodesignVec(n)
        instances.append(env)
        return env, env.base_embodiment

    class Model:
        @staticmethod
        def predict(observation, deterministic=False):
            return np.zeros((len(observation), 1), np.float32), None

    class Normalizer:
        @staticmethod
        def normalize_obs(observation):
            return observation

    monkeypatch.setattr(module, "make_baseline_vec", make_env)
    scorer = CandidateScorer("typed", schema, device="cpu", seed=3)
    designs = np.stack([ToyCodesignVec(1).base_embodiment] * 2)
    result = scorer.score(
        Model(), Normalizer(), designs, round_index=0, horizon=5)

    assert result.complete
    assert instances[0].reset_calls == 1
    assert all(env.closed for env in instances)


def test_incomplete_runtime_search_never_replaces_elites():
    from loki.single_cluster import (CandidateScoreResult, ClusterRuntime,
                                     SearchRoundResult)

    class Store:
        cluster_id = 2
        @staticmethod
        def sample_ids(_rng, count):
            return np.arange(20, 20 + count)
        @staticmethod
        def flat_designs(_env, ids):
            return np.zeros((len(ids), 1), np.float32)

    class Scorer:
        @staticmethod
        def score(*_args, **_kwargs):
            return CandidateScoreResult(
                np.array([5.0, -np.inf]), np.array([False, True]), 7, False)

    class Model:
        completed_ppo_iterations = 2

    pool = ElitePool([10, 11])
    runtime = ClusterRuntime(
        config=SingleClusterConfig(
            elite_size=2, candidates_per_search=2,
            replacements_per_search=1),
        store=Store(), pool=pool, rng=np.random.default_rng(1),
        training_env=object(), vector=object(), normalizer=object(),
        model=Model(), scorer=Scorer(), logical_slots=[0, 1])
    result = runtime.search(batch_size=1)
    assert isinstance(result, SearchRoundResult)
    assert not result.complete
    assert pool.design_ids.tolist() == [10, 11]
    assert runtime.counters["search_rounds"] == 0
    assert runtime.counters["incomplete_search_rounds"] == 1
    assert runtime.counters["candidate_frames"] == 7



def test_cluster_runtime_owns_independent_torch_rng_state():
    from loki.single_cluster import ClusterRuntime

    class Store:
        cluster_id = 0

    class Model:
        completed_ppo_iterations = 0

    def runtime(state):
        return ClusterRuntime(
            config=SingleClusterConfig(), store=Store(), pool=ElitePool([1]),
            rng=np.random.default_rng(), training_env=object(), vector=object(),
            normalizer=object(), model=Model(), scorer=object(),
            logical_slots=[0], torch_rng=state)

    torch.manual_seed(11)
    state_a = torch.get_rng_state()
    reference_a = torch.Generator().set_state(state_a)
    torch.manual_seed(22)
    state_b = torch.get_rng_state()
    reference_b = torch.Generator().set_state(state_b)
    torch.manual_seed(99)
    outer = torch.get_rng_state().clone()
    first, second = runtime(state_a), runtime(state_b)

    with first.rng_scope():
        first_draw = torch.rand(3)
    with second.rng_scope():
        second_draw = torch.rand(3)
    with first.rng_scope():
        next_first_draw = torch.rand(3)

    torch.testing.assert_close(first_draw, torch.rand(3, generator=reference_a))
    torch.testing.assert_close(next_first_draw,
                               torch.rand(3, generator=reference_a))
    torch.testing.assert_close(second_draw, torch.rand(3, generator=reference_b))
    torch.testing.assert_close(torch.get_rng_state(), outer)
