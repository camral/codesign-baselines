"""Six-cluster LOKI wall-clock coordinator contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from common.wall_clock import WallClockBudget
from loki.gym_train import (BudgetLearningRate, SynchronizedCoordinator,
                            _cluster_metrics, _live_training_metrics,
                            _preparation_provenance,
                            allocate_worlds,
                            save_training_checkpoint)
from loki.single_cluster import SearchRoundResult, SingleClusterConfig


class FakeArtifact:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.cluster_ids = np.arange(cluster_id * 100, cluster_id * 100 + 20)
        self.identity = {"cluster_id": cluster_id}


class FakeModel:
    def __init__(self, version=0):
        self.completed_ppo_iterations = version
        self.lr_schedule = BudgetLearningRate()

    def save(self, path):
        path.write_bytes(b"policy")


class FakeNormalizer:
    def save(self, path):
        path.write_bytes(b"normalizer")


class FakeRuntime:
    def __init__(self, cluster_id, version=0, fail_search=False):
        self.store = FakeArtifact(cluster_id)
        self.model = FakeModel(version)
        self.normalizer = FakeNormalizer()
        self.config = SingleClusterConfig(
            elite_size=2, search_frequency=2, candidates_per_search=3,
            replacements_per_search=1, candidate_horizon=2,
            rollout_steps=2, ppo_epochs=1, ppo_batch_size=2)
        self.vector = SimpleNamespace(
            training_reward=float(cluster_id),
            batch=SimpleNamespace(frames=cluster_id * 10))
        self.training_env = SimpleNamespace(
            num_designs=2, num_envs=4, worlds_per_design=2)
        self.counters = {
            "ppo_frames": 0, "candidate_frames": 0, "ranking_frames": 0,
            "search_rounds": version // 2 if version % 2 else 0,
            "incomplete_search_rounds": 0, "candidate_rejections": 0,
            "elite_replacements": 0,
        }
        self.fail_search = fail_search
        self.commits = []

    @property
    def cluster_id(self):
        return self.store.cluster_id

    @property
    def policy_version(self):
        return self.model.completed_ppo_iterations

    def train_iteration(self, budget=None, progress=None):
        if progress:
            progress()
        if budget.expired:
            return {"complete": False, "frames": 0,
                    "iteration": self.policy_version}
        self.model.completed_ppo_iterations += 1
        self.counters["ppo_frames"] += 10
        return {"complete": True, "frames": 10,
                "iteration": self.policy_version}

    def score_search(self, **_kwargs):
        ids = np.arange(self.cluster_id * 100 + 20,
                        self.cluster_id * 100 + 23)
        complete = not self.fail_search
        return SearchRoundResult(
            self.cluster_id, self.policy_version, ids,
            np.array([5.0, 4.0, 3.0]), np.zeros(3, bool), 6, complete)

    def score_elites(self, **_kwargs):
        ids = np.arange(self.cluster_id * 100, self.cluster_id * 100 + 2)
        return SearchRoundResult(
            self.cluster_id, self.policy_version, ids,
            np.array([2.0, 1.0]), np.zeros(2, bool), 4, True)

    def commit_search(self, value):
        self.commits.append(value.policy_version)
        self.counters["search_rounds"] += 1
        self.counters["elite_replacements"] += 1
        return SearchRoundResult(**{**value.__dict__, "replacements": ({
            "slot": 0, "new_design_id": int(value.candidate_ids[0])},)})

    def state_dict(self):
        return {"cluster_id": self.cluster_id,
                "policy_version": self.policy_version,
                "counters": dict(self.counters)}


def runtimes(version=0):
    return [FakeRuntime(index, version) for index in range(6)]


def test_total_world_allocation_is_exact_and_deterministic():
    assert allocate_worlds(16_384, 6) == [2731, 2731, 2731, 2731, 2730, 2730]
    assert sum(allocate_worlds(7, 6)) == 7
    with pytest.raises(ValueError, match="at least"):
        allocate_worlds(5, 6)


def test_round_robin_training_recovers_a_partly_advanced_cycle():
    values = runtimes()
    values[0].model.completed_ppo_iterations = 1
    coordinator = SynchronizedCoordinator(
        values, evaluation_worlds=2, candidate_batch_size=2)
    budget = WallClockBudget(100, 0, lambda: 1)
    assert coordinator.train_to_next_synchronization(budget)
    assert coordinator.policy_versions == [1] * 6
    assert values[0].counters["ppo_frames"] == 0
    assert all(value.counters["ppo_frames"] == 10 for value in values[1:])


def test_synchronized_search_selects_best_cluster_and_commits_all():
    values = runtimes(version=2)
    coordinator = SynchronizedCoordinator(
        values, evaluation_worlds=2, candidate_batch_size=2)
    result = coordinator.synchronized_search(
        WallClockBudget(100, 0, lambda: 1))
    assert len(result) == 6
    assert all(value.commits == [2] for value in values)
    # Scores tie, so deterministic cluster-ID tie breaking selects cluster zero.
    assert coordinator.selected["cluster_id"] == 0
    assert coordinator.selected["design_ids"].tolist() == [20, 21]
    assert coordinator.completed_search_cycles == 1
    metrics = _cluster_metrics(coordinator, elapsed_seconds=7)
    assert metrics["elite_replacements"] == 6
    assert metrics["cluster/0/elite_turnover_rate"] == pytest.approx(0.5)
    assert metrics["selected_design_ids"] == [20, 21]
    assert metrics["ranking_age_seconds"] == pytest.approx(6)
    assert metrics["training_logical_worlds"] == 12
    assert metrics["training_physical_worlds"] == 24
    assert metrics["candidate_peak_physical_worlds"] == 4
    assert metrics["evaluation_physical_worlds"] == 4
    assert metrics["phase/candidate_scoring_seconds"] >= 0
    assert metrics["cluster/0/phase/ranking_seconds"] >= 0
    assert metrics["train/reward"] == pytest.approx(2.5)
    live = _live_training_metrics(coordinator, elapsed_seconds=10)
    assert live["train/reward"] == pytest.approx(2.5)
    assert live["ppo_frames"] == 150
    assert live["train/fps"] > 0


def test_incomplete_global_search_commits_no_cluster():
    values = runtimes(version=2)
    values[3].fail_search = True
    coordinator = SynchronizedCoordinator(
        values, evaluation_worlds=2, candidate_batch_size=2)
    assert not coordinator.synchronized_search(
        WallClockBudget(100, 0, lambda: 1))
    assert all(not value.commits for value in values)
    assert coordinator.completed_search_cycles == 0


def test_six_controller_checkpoint_is_atomic_and_complete(tmp_path):
    coordinator = SynchronizedCoordinator(
        runtimes(version=2), evaluation_worlds=2, candidate_batch_size=2)
    coordinator.initial_selection()
    path = save_training_checkpoint(
        tmp_path, coordinator, {"method": "loki"},
        WallClockBudget(100, 0, lambda: 5), 1, status="running")
    assert path.name == "000001"
    for cluster_id in range(6):
        assert (path / f"cluster_{cluster_id}/policy.zip").is_file()
        assert (path / f"cluster_{cluster_id}/vecnormalize.pkl").is_file()
    assert (path / "runtime.pt").is_file()
    assert (path / "manifest.json").is_file()
    assert not list((tmp_path / "training_checkpoints").glob(".*.tmp"))
    runtime = __import__("torch").load(
        path / "runtime.pt", map_location="cpu", weights_only=False)
    assert "phase_seconds" in runtime
    assert "cluster_phase_seconds" in runtime


def test_budget_learning_rate_uses_elapsed_fraction_and_update_warmup():
    schedule = BudgetLearningRate(3e-4)
    assert schedule(0) == pytest.approx(3e-5)
    schedule.iteration = 5
    schedule.fraction = 0.5
    assert schedule(0) == pytest.approx(1.5e-4)


def test_preparation_provenance_reports_hashes_and_shared_cost(tmp_path):
    vae = tmp_path / "vae"
    vae.mkdir()
    (vae / "convergence.json").write_text(
        '{"elapsed_seconds": 20.0}\n')
    store = SimpleNamespace(
        identity={
            "cluster_manifest_sha256": "cluster-hash",
            "corpus_manifest_sha256": "corpus-hash"},
        manifest={
            "vae": str(vae), "vae_checkpoint_sha256": "vae-hash",
            "vae_epochs_completed": 200, "elapsed_seconds": 3.0,
            "extraction_seconds": 1.0, "design_count": 500_000,
            "diagnostics": {"cluster_sizes": [1, 2, 3, 4, 5, 6]}},
        reader=SimpleNamespace(
            manifest={"state": {"elapsed_seconds": 7.0}}))
    result = _preparation_provenance([store])
    assert result["cluster_manifest_sha256"] == "cluster-hash"
    assert result["vae_checkpoint_sha256"] == "vae-hash"
    assert result["total_preparation_seconds"] == pytest.approx(30)


def test_loki_snapshot_runs_in_standard_evaluation_worker(tmp_path, monkeypatch):
    import threading
    import time
    import torch
    from common import evaluation_worker
    from loki import single_cluster as single_module
    from loki.gym_train import build_evaluation_snapshot
    from loki.single_cluster import (SingleClusterConfig,
                                     create_cluster_runtime)
    from test_loki_single_cluster import (ToyCodesignVec, make_clusters,
                                          tiny_policy_kwargs)
    from test_loki_vae import make_corpus

    corpus, clusters = tmp_path / "corpus", tmp_path / "clusters"
    make_corpus(corpus)
    make_clusters(corpus, clusters)

    def make_env(_env_id, n, **_kwargs):
        env = ToyCodesignVec(n)
        return env, env.base_embodiment

    monkeypatch.setattr(single_module, "make_baseline_vec", make_env)
    monkeypatch.setattr(single_module, "controller_policy_kwargs",
                        tiny_policy_kwargs)
    monkeypatch.setattr(evaluation_worker, "make_baseline_vec", make_env)
    runtime = create_cluster_runtime(
        "typed", corpus, clusters, 0, num_envs=2, device="cpu", seed=4,
        config=SingleClusterConfig(
            elite_size=2, search_frequency=1, candidates_per_search=4,
            replacements_per_search=1, candidate_horizon=3,
            rollout_steps=4, ppo_epochs=1, ppo_batch_size=4))
    coordinator = SynchronizedCoordinator(
        [runtime], evaluation_worlds=2, candidate_batch_size=2)
    coordinator.initial_selection()
    payload = build_evaluation_snapshot(coordinator)
    payload["_evaluation"] = {
        "version": 1, "elapsed_seconds": 0.0, "kind": "current",
        "metrics": {"selected_cluster": 0}}
    path = tmp_path / "snapshot.pt"
    torch.save(payload, path)
    evaluator = evaluation_worker.SnapshotEvaluator({
        "method": "loki", "seed": 0, "device": "cpu",
        "output_dir": str(tmp_path), "env_id": "typed",
        "evaluation_worlds": 2, "video_stride": 1, "video_fps": 10,
    })
    published = []
    try:
        metrics, _ = evaluator.evaluate(
            path, deadline=time.monotonic() + 30, stop=threading.Event(),
            on_metrics=published.append)
        assert metrics["eval/status"] == "complete"
        assert metrics["selected_cluster"] == 0
        assert published and np.isfinite(published[0]["eval/reward"])
        evaluator.save_best(path, metrics["eval/reward"])
        assert (tmp_path / "best_checkpoint.pt").is_file()
        assert (tmp_path / "best_embodiments.npy").is_file()
    finally:
        evaluator.close()
        runtime.close()


def test_comparison_launcher_constructs_loki_command(tmp_path):
    import json
    from launch_comparison import command_for, manifest_config

    args = SimpleNamespace(
        hours=2, num_envs=60, env_id="BallCatcher-Pitch",
        project=None, group=None, entity=None, evaluation_worlds=12,
        loki_rollout_steps=19,
        loki_corpus_root=tmp_path / "corpora",
        loki_artifact_root=tmp_path / "offline")
    command = command_for(args, "loki", 7, 2, tmp_path / "run")
    assert command[command.index("-m") + 1] == "loki.gym_train"
    assert command[command.index("--num-envs") + 1] == "60"
    assert command[command.index("--evaluation-worlds") + 1] == "12"
    assert command[command.index("--rollout-steps") + 1] == "19"
    assert command[command.index("--corpus") + 1] == str(
        tmp_path / "corpora/BallCatcher-Pitch")
    assert command[command.index("--cluster-artifact") + 1] == str(
        tmp_path / "offline/BallCatcher-Pitch/clusters")
    encoded = json.dumps(manifest_config(args))
    assert str(tmp_path / "corpora") in encoded
    assert str(tmp_path / "offline") in encoded
