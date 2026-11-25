"""Six-cluster LOKI training under one wall-clock and evaluation schedule."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from common.async_evaluation import AsyncEvaluator
from common.run_log import RunLog
from common.wall_clock import WallClockBudget
from .collect_all import PRESET_TO_CORPUS, REPO, corpus_dir
from .collectors import COLLECTORS
from .collectors.storage import json_value, write_json
from .single_cluster import (ClusterDesignStore, EvaluationSelection,
                             SingleClusterConfig, create_cluster_runtime)


@dataclass(frozen=True)
class OnlineConfig:
    cluster_count: int = 6
    evaluation_worlds: int = 128
    candidate_batch_size: int = 128

    def validate(self, total_worlds):
        if min(self.cluster_count, self.evaluation_worlds,
               self.candidate_batch_size) < 1:
            raise ValueError("LOKI online counts must be positive")
        if total_worlds < self.cluster_count:
            raise ValueError(
                "--num-envs must provide at least one training world per cluster")


def allocate_worlds(total_worlds, cluster_count):
    """Deterministically split a total allocation without creating extra worlds."""
    if cluster_count < 1 or total_worlds < cluster_count:
        raise ValueError("total worlds must be at least the cluster count")
    quotient, remainder = divmod(int(total_worlds), int(cluster_count))
    return [quotient + (index < remainder) for index in range(cluster_count)]


class BudgetLearningRate:
    """Released cosine LR and five-update warmup, driven by run-time fraction."""

    def __init__(self, base_rate=3e-4):
        self.base_rate = float(base_rate)
        self.fraction = 0.0
        self.iteration = 0

    def __call__(self, _progress_remaining):
        fraction = min(max(float(self.fraction), 0.0), 1.0)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * fraction))
        if self.iteration < 5:
            alpha = self.iteration / 5.0
            multiplier *= 0.1 * (1.0 - alpha) + alpha
        return self.base_rate * multiplier


class SynchronizedCoordinator:
    """Round-robin PPO plus globally atomic six-cluster search/ranking."""

    def __init__(self, runtimes, *, evaluation_worlds, candidate_batch_size,
                 selection=None):
        self.runtimes = sorted(runtimes, key=lambda value: value.cluster_id)
        ids = [runtime.cluster_id for runtime in self.runtimes]
        if ids != list(range(len(self.runtimes))):
            raise ValueError("Cluster runtimes must have contiguous IDs from zero")
        if len({id(runtime.model) for runtime in self.runtimes}) != len(runtimes):
            raise ValueError("Clusters must not share a policy object")
        if len({id(runtime.normalizer) for runtime in self.runtimes}) != len(runtimes):
            raise ValueError("Clusters must not share normalization state")
        self.evaluation_worlds = int(evaluation_worlds)
        self.candidate_batch_size = int(candidate_batch_size)
        self.selection = selection or EvaluationSelection()
        self.selected = None
        self.completed_search_cycles = min(
            runtime.counters["search_rounds"] for runtime in self.runtimes)
        self.phase_seconds = {
            "initialization": 0.0,
            "ppo": 0.0,
            "candidate_scoring": 0.0,
            "ranking": 0.0,
            "checkpointing": 0.0,
        }
        self.cluster_phase_seconds = {
            runtime.cluster_id: {
                "ppo": 0.0, "candidate_scoring": 0.0, "ranking": 0.0}
            for runtime in self.runtimes
        }

    def record_phase(self, phase, seconds, *, cluster_id=None):
        seconds = max(0.0, float(seconds))
        self.phase_seconds[phase] = self.phase_seconds.get(phase, 0.0) + seconds
        if cluster_id is not None:
            phases = self.cluster_phase_seconds[int(cluster_id)]
            phases[phase] = phases.get(phase, 0.0) + seconds

    @property
    def policy_versions(self):
        return [runtime.policy_version for runtime in self.runtimes]

    @property
    def synchronized_version(self):
        versions = self.policy_versions
        return versions[0] if len(set(versions)) == 1 else None

    def initial_selection(self):
        runtime = self.runtimes[0]
        if self.evaluation_worlds > len(runtime.store.cluster_ids):
            raise ValueError("Cluster 0 cannot supply the initial evaluation batch")
        ids = runtime.store.cluster_ids[:self.evaluation_worlds].copy()
        self.selected = {
            "cluster_id": 0,
            "policy_version": runtime.policy_version,
            "design_ids": ids,
            "scores": None,
            "cluster_score": None,
            "elapsed_seconds": None,
            "ranking_status": "unranked",
        }
        return self.selected

    def _search_due(self):
        version = self.synchronized_version
        if version is None or version == 0:
            return False
        frequency = self.runtimes[0].config.search_frequency
        return (version % frequency == 0 and
                self.completed_search_cycles < version // frequency)

    def train_to_next_synchronization(self, budget, *, progress=None):
        """Bring every controller to min(version)+1, stopping coherently."""
        target = min(self.policy_versions) + 1
        for runtime in self.runtimes:
            if runtime.policy_version >= target:
                continue
            schedule = runtime.model.lr_schedule
            if isinstance(schedule, BudgetLearningRate):
                schedule.fraction = budget.elapsed / budget.seconds
                schedule.iteration = runtime.policy_version
            started = time.perf_counter()
            result = runtime.train_iteration(budget=budget, progress=progress)
            self.record_phase(
                "ppo", time.perf_counter() - started,
                cluster_id=runtime.cluster_id)
            if not result["complete"]:
                return False
        return self.synchronized_version == target

    def synchronized_search(self, budget, *, progress=None):
        """Score all clusters, then commit all six replacements together."""
        if not self._search_due():
            return False
        staged_search, staged_elites = [], []
        for runtime in self.runtimes:
            started = time.perf_counter()
            search = runtime.score_search(
                deadline=budget, batch_size=self.candidate_batch_size,
                progress=progress)
            self.record_phase(
                "candidate_scoring", time.perf_counter() - started,
                cluster_id=runtime.cluster_id)
            if not search.complete:
                return False
            started = time.perf_counter()
            elites = runtime.score_elites(
                deadline=budget, batch_size=self.candidate_batch_size,
                progress=progress)
            self.record_phase(
                "ranking", time.perf_counter() - started,
                cluster_id=runtime.cluster_id)
            if not elites.complete:
                return False
            staged_search.append(search)
            staged_elites.append(elites)
        # Validate every transaction before the first elite pool changes.
        for runtime, search in zip(self.runtimes, staged_search):
            valid = np.count_nonzero(
                ~search.rejected & np.isfinite(search.scores))
            if valid < runtime.config.replacements_per_search:
                raise ValueError(
                    f"Cluster {runtime.cluster_id} has only {valid} valid "
                    "candidates for synchronized replacement")
        committed = [runtime.commit_search(search)
                     for runtime, search in zip(self.runtimes, staged_search)]
        elapsed = budget.elapsed
        for search, elites in zip(staged_search, staged_elites):
            self.selection.update(
                search.cluster_id, search.policy_version,
                np.concatenate((search.candidate_ids, elites.candidate_ids)),
                np.concatenate((search.scores, elites.scores)),
                np.concatenate((search.rejected, elites.rejected)),
                elapsed_seconds=elapsed)
        self.selected = self.selection.select(
            self.evaluation_worlds,
            cluster_ids=[runtime.cluster_id for runtime in self.runtimes])
        self.selected["ranking_status"] = "fresh"
        self.completed_search_cycles += 1
        return committed

    def aggregate_counters(self):
        keys = set().union(*(runtime.counters for runtime in self.runtimes))
        return {key: sum(int(runtime.counters.get(key, 0))
                         for runtime in self.runtimes) for key in sorted(keys)}

    def state_dict(self):
        return {
            "completed_search_cycles": self.completed_search_cycles,
            "evaluation_worlds": self.evaluation_worlds,
            "candidate_batch_size": self.candidate_batch_size,
            "phase_seconds": dict(self.phase_seconds),
            "cluster_phase_seconds": self.cluster_phase_seconds,
            "selection": self.selection.state_dict(),
            "selected": (None if self.selected is None else
                         json_value(self.selected)),
            "runtime_states": [runtime.state_dict()
                               for runtime in self.runtimes],
        }


def _normalizer_state(normalizer):
    return {
        "mean": np.asarray(normalizer.obs_rms.mean).copy(),
        "var": np.asarray(normalizer.obs_rms.var).copy(),
        "count": float(normalizer.obs_rms.count),
        "epsilon": float(normalizer.epsilon),
        "clip_obs": float(normalizer.clip_obs),
    }


def _preparation_provenance(stores):
    store = stores[0]
    manifest = store.manifest
    corpus_manifest = store.reader.manifest
    vae_report_path = Path(manifest.get("vae", "")) / "convergence.json"
    vae_report = (json.loads(vae_report_path.read_text())
                  if vae_report_path.is_file() else {})
    corpus_seconds = corpus_manifest.get("state", {}).get("elapsed_seconds")
    vae_seconds = vae_report.get("elapsed_seconds")
    clustering_seconds = manifest.get("elapsed_seconds")
    durations = [corpus_seconds, vae_seconds, clustering_seconds]
    return {
        "cluster_manifest_sha256": store.identity[
            "cluster_manifest_sha256"],
        "corpus_manifest_sha256": store.identity[
            "corpus_manifest_sha256"],
        "vae_checkpoint_sha256": manifest.get("vae_checkpoint_sha256"),
        "vae_epochs_completed": manifest.get("vae_epochs_completed"),
        "vae_fixed_epoch_recipe": manifest.get("vae_fixed_epoch_recipe"),
        "vae_converged_diagnostic": manifest.get("vae_converged"),
        "corpus_collection_seconds": corpus_seconds,
        "vae_training_seconds": vae_seconds,
        "clustering_seconds": clustering_seconds,
        "latent_extraction_seconds": manifest.get("extraction_seconds"),
        "total_preparation_seconds": (
            sum(float(value) for value in durations
                if value is not None)),
        "upstream_loki_commit": manifest.get("upstream_loki_commit"),
        "design_count": manifest.get("design_count"),
        "cluster_sizes": manifest.get("diagnostics", {}).get("cluster_sizes"),
    }


def build_evaluation_snapshot(coordinator):
    selected = coordinator.selected or coordinator.initial_selection()
    runtime = coordinator.runtimes[int(selected["cluster_id"])]
    policy = io.BytesIO()
    runtime.model.save(policy)
    design_ids = np.asarray(selected["design_ids"], dtype=np.int64)
    designs = runtime.store.flat_designs(runtime.training_env, design_ids)
    return {
        "format_version": 1,
        "method": "loki",
        "ppo": policy.getvalue(),
        "normalizer": _normalizer_state(runtime.normalizer),
        "token_schema": runtime.store.schema,
        "design": designs,
        "design_ids": design_ids,
        "cluster_id": runtime.cluster_id,
        "policy_version": runtime.policy_version,
        "ranking_status": selected.get("ranking_status", "unranked"),
        "ranking_scores": selected.get("scores"),
        "ranking_score": selected.get("cluster_score"),
        "ranking_elapsed_seconds": selected.get("elapsed_seconds"),
        "cluster_identity": runtime.store.identity,
    }


def save_training_checkpoint(output, coordinator, config, budget, sequence,
                             *, status):
    """Atomically save all six policies, optimizers, normalizers, and pools."""
    root = Path(output) / "training_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    name = f"{int(sequence):06d}"
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Training checkpoint already exists: {destination}")
    staging = root / f".{name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for runtime in coordinator.runtimes:
        folder = staging / f"cluster_{runtime.cluster_id}"
        folder.mkdir()
        runtime.model.save(folder / "policy.zip")
        runtime.normalizer.save(folder / "vecnormalize.pkl")
    torch.save(coordinator.state_dict(), staging / "runtime.pt")
    write_json(staging / "manifest.json", {
        "format_version": 1,
        "status": status,
        "sequence": int(sequence),
        "elapsed_seconds": budget.elapsed,
        "config": config,
        "policy_versions": coordinator.policy_versions,
        "cluster_identities": [runtime.store.identity
                               for runtime in coordinator.runtimes],
        "counters": coordinator.aggregate_counters(),
    })
    staging.rename(destination)
    write_json(root / ".latest.tmp", {"checkpoint": name, "status": status})
    (root / ".latest.tmp").replace(root / "latest.json")
    return destination


def _cluster_metrics(coordinator, *, elapsed_seconds=None):
    metrics = {}
    training_rewards = []
    logical_worlds = 0
    physical_worlds = 0
    for runtime in coordinator.runtimes:
        prefix = f"cluster/{runtime.cluster_id}"
        metrics[f"{prefix}/policy_version"] = runtime.policy_version
        training_reward = runtime.vector.training_reward
        metrics[f"{prefix}/training_reward"] = training_reward
        if training_reward is not None:
            training_rewards.append(float(training_reward))
        logical = int(runtime.training_env.num_designs)
        physical = int(runtime.training_env.num_envs)
        worlds_per_design = int(getattr(
            runtime.training_env, "worlds_per_design",
            physical // max(logical, 1)))
        logical_worlds += logical
        physical_worlds += physical
        metrics[f"{prefix}/logical_worlds"] = logical
        metrics[f"{prefix}/physical_worlds"] = physical
        metrics[f"{prefix}/worlds_per_design"] = worlds_per_design
        for phase, seconds in coordinator.cluster_phase_seconds[
                runtime.cluster_id].items():
            metrics[f"{prefix}/phase/{phase}_seconds"] = seconds
        for key, value in runtime.counters.items():
            metrics[f"{prefix}/{key}"] = value
        searches = int(runtime.counters.get("search_rounds", 0))
        metrics[f"{prefix}/elite_turnover_rate"] = (
            float(runtime.counters.get("elite_replacements", 0)) /
            (runtime.config.elite_size * searches) if searches else 0.0)
    counters = coordinator.aggregate_counters()
    metrics.update({key: value for key, value in counters.items()})
    metrics["training_logical_worlds"] = logical_worlds
    metrics["training_physical_worlds"] = physical_worlds
    metrics["candidate_peak_logical_worlds"] = coordinator.candidate_batch_size
    worlds_per_design = max(
        int(getattr(runtime.training_env, "worlds_per_design",
                    runtime.training_env.num_envs //
                    max(runtime.training_env.num_designs, 1)))
        for runtime in coordinator.runtimes)
    metrics["candidate_peak_physical_worlds"] = (
        coordinator.candidate_batch_size * worlds_per_design)
    metrics["evaluation_logical_worlds"] = coordinator.evaluation_worlds
    metrics["evaluation_physical_worlds"] = (
        coordinator.evaluation_worlds * worlds_per_design)
    for phase, seconds in coordinator.phase_seconds.items():
        metrics[f"phase/{phase}_seconds"] = seconds
    metrics["frames"] = sum(
        counters.get(key, 0)
        for key in ("ppo_frames", "candidate_frames", "ranking_frames"))
    if training_rewards:
        metrics["train/reward"] = float(np.mean(training_rewards))
    if coordinator.selected is not None:
        metrics["selected_cluster"] = coordinator.selected["cluster_id"]
        metrics["selected_policy_version"] = coordinator.selected["policy_version"]
        metrics["selected_design_ids"] = [
            int(value) for value in coordinator.selected["design_ids"]]
        metrics["ranking_status"] = coordinator.selected["ranking_status"]
        metrics["ranking_score"] = coordinator.selected.get("cluster_score")
        ranked_at = coordinator.selected.get("elapsed_seconds")
        if elapsed_seconds is not None and ranked_at is not None:
            metrics["ranking_age_seconds"] = max(
                0.0, float(elapsed_seconds) - float(ranked_at))
    if torch.cuda.is_available():
        devices = {str(runtime.model.device)
                   for runtime in coordinator.runtimes
                   if hasattr(runtime.model, "device") and
                   str(runtime.model.device).startswith("cuda")}
        if devices:
            metrics["gpu_peak_memory_bytes"] = max(
                torch.cuda.max_memory_allocated(torch.device(device))
                for device in devices)
            metrics["gpu_peak_reserved_bytes"] = max(
                torch.cuda.max_memory_reserved(torch.device(device))
                for device in devices)
    return {key: value for key, value in metrics.items() if value is not None}


def _live_training_metrics(coordinator, elapsed_seconds):
    """Small progress record safe to emit while a PPO update is in flight."""
    metrics = {}
    rewards = []
    ppo_frames = 0
    for runtime in coordinator.runtimes:
        prefix = f"cluster/{runtime.cluster_id}"
        reward = runtime.vector.training_reward
        if reward is not None:
            reward = float(reward)
            rewards.append(reward)
            metrics[f"{prefix}/training_reward"] = reward
        ppo_frames += int(runtime.vector.batch.frames)
    counters = coordinator.aggregate_counters()
    frames = (ppo_frames + counters.get("candidate_frames", 0) +
              counters.get("ranking_frames", 0))
    metrics.update({
        "ppo_frames": ppo_frames,
        "frames": frames,
        "train/fps": frames / max(float(elapsed_seconds), 1e-9),
        "training_clusters_with_returns": len(rewards),
    })
    if rewards:
        metrics["train/reward"] = float(np.mean(rewards))
    return metrics


def run(args):
    online = OnlineConfig(
        cluster_count=args.clusters,
        evaluation_worlds=args.evaluation_worlds,
        candidate_batch_size=args.candidate_batch_size)
    online.validate(args.num_envs)
    if args.clusters != 6:
        raise ValueError("The agreed production LOKI configuration requires six clusters")
    if args.evaluation_worlds > args.candidates + args.elite_size:
        raise ValueError("evaluation worlds exceed candidates plus elite pool")
    canonical = PRESET_TO_CORPUS[args.env_id]
    corpus = args.corpus or REPO / "data/loki" / corpus_dir(canonical)
    clusters = args.cluster_artifact or REPO / "artifacts/loki" / corpus_dir(canonical) / "clusters"
    allocation = allocate_worlds(args.num_envs, args.clusters)
    budget = WallClockBudget.start(args.hours)
    cluster_config = SingleClusterConfig(
        elite_size=args.elite_size,
        search_frequency=args.search_frequency,
        candidates_per_search=args.candidates,
        replacements_per_search=args.replacements,
        candidate_horizon=args.candidate_horizon,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size,
        max_iterations=2**31 - 1)
    # Artifact validation is part of the measured online initialization.
    stores = [ClusterDesignStore(corpus, clusters, cluster_id)
              for cluster_id in range(args.clusters)]
    preparation = _preparation_provenance(stores)
    logged = {
        **vars(args), "method": "loki", "canonical_corpus": canonical,
        "corpus": str(corpus), "cluster_artifact": str(clusters),
        "world_allocation": allocation,
        "online_config": asdict(online),
        "cluster_config": asdict(cluster_config),
        "learning_rate_schedule": "wall-clock cosine with five-update warmup",
        "evaluation_execution": "independent_process",
        "evaluation_protocol": "one selected cluster controller over its top-N training-ranked designs",
        "preparation_provenance": preparation,
    }
    log = RunLog(
        args.output_dir, budget, logged, project=args.project,
        entity=args.entity, group=args.group, name=args.run_name)
    evaluator = None
    runtimes = []
    coordinator = None
    checkpoint_sequence = 0
    try:
        evaluator = AsyncEvaluator(log, args, "loki")
        for cluster_id, worlds in enumerate(allocation):
            runtime = create_cluster_runtime(
                args.env_id, corpus, clusters, cluster_id,
                num_envs=worlds, device=args.device,
                seed=args.seed + cluster_id * 100_003,
                env_kwargs=json.loads(args.env_kwargs),
                rejection_reward=args.rejection_reward,
                config=cluster_config, store=stores[cluster_id])
            runtime.model.lr_schedule = BudgetLearningRate(
                cluster_config.learning_rate)
            runtimes.append(runtime)
        coordinator = SynchronizedCoordinator(
            runtimes, evaluation_worlds=args.evaluation_worlds,
            candidate_batch_size=args.candidate_batch_size)
        coordinator.initial_selection()
        coordinator.record_phase("initialization", budget.elapsed)
        last_training_log = float("-inf")

        def training_progress():
            nonlocal last_training_log
            evaluator.check()
            now = budget.elapsed
            if now - last_training_log >= 30.0:
                log.log(**_live_training_metrics(coordinator, now))
                last_training_log = now

        def publish(force=False):
            metrics = _cluster_metrics(
                coordinator, elapsed_seconds=budget.elapsed)
            metrics["train/fps"] = metrics["frames"] / max(budget.elapsed, 1e-9)
            evaluator.publish(
                lambda: build_evaluation_snapshot(coordinator),
                force=force, **metrics)

        publish(force=True)
        while not budget.expired:
            complete = coordinator.train_to_next_synchronization(
                budget, progress=training_progress)
            if not complete:
                break
            if coordinator._search_due():
                replacements = coordinator.synchronized_search(
                    budget, progress=evaluator.check)
                if not replacements:
                    # Keep the last fully ranked controller/design snapshot.
                    # A partial global scoring transaction is never published.
                    break
                checkpoint_sequence += 1
                checkpoint_started = time.perf_counter()
                save_training_checkpoint(
                    log.output, coordinator, logged, budget,
                    checkpoint_sequence, status="running")
                coordinator.record_phase(
                    "checkpointing", time.perf_counter() - checkpoint_started)
                log.log(**_cluster_metrics(
                    coordinator, elapsed_seconds=budget.elapsed))
                publish(force=True)
            else:
                evaluator.check()
        checkpoint_sequence += 1
        checkpoint_started = time.perf_counter()
        final_checkpoint = save_training_checkpoint(
            log.output, coordinator, logged, budget,
            checkpoint_sequence, status="complete")
        coordinator.record_phase(
            "checkpointing", time.perf_counter() - checkpoint_started)
        evaluator.close()
        metrics = _cluster_metrics(
            coordinator, elapsed_seconds=budget.elapsed)
        log.finish(
            "complete", **metrics, policy_versions=coordinator.policy_versions,
            completed_search_cycles=coordinator.completed_search_cycles,
            final_checkpoint=str(final_checkpoint),
            preparation_provenance=preparation,
            cluster_identities=[runtime.store.identity
                                for runtime in coordinator.runtimes],
            best_eval_reward=evaluator.best_eval_reward)
        return coordinator
    except Exception:
        if evaluator is not None:
            evaluator.close(raise_errors=False)
        log.finish("failed", error=traceback.format_exc())
        raise
    finally:
        for runtime in runtimes:
            runtime.close()
        if evaluator is not None:
            evaluator.close(raise_errors=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=sorted(COLLECTORS),
                        default="BallCatcher-Pitch")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--evaluation-worlds", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--cluster-artifact", type=Path)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--elite-size", type=int, default=20)
    parser.add_argument("--search-frequency", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--replacements", type=int, default=2)
    parser.add_argument("--candidate-horizon", type=int, default=200)
    parser.add_argument("--candidate-batch-size", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=2560)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--ppo-batch-size", type=int, default=5120)
    parser.add_argument("--rejection-reward", type=float)
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--project")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as error:
        parser.error(f"env-kwargs must be a JSON object: {error}")
    if not isinstance(env_kwargs, dict):
        parser.error("env-kwargs must be a JSON object")
    if (not math.isfinite(args.hours) or args.hours <= 0 or
            min(args.num_envs, args.evaluation_worlds, args.video_stride,
                args.video_fps) < 1):
        parser.error("hours and world/video counts must be positive")
    return args


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
