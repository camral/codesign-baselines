"""Train one LOKI cluster controller with literal candidate replacement.

This is the first online vertical slice. It intentionally owns one controller
and one frozen cluster; six-controller scheduling and held-out evaluation are
separate later integration stages.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import bisect
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecNormalize

from common.codesign_adapter import make_baseline_vec
from common.sb3_env import SB3GymBatch
from ppo_ngopt_rlgo.timed_ppo import TimedPPO, TrainingDeadlineReached
from .collect_all import PRESET_TO_CORPUS, REPO, corpus_dir
from .collectors import COLLECTORS
from .collectors.storage import (CorpusReader, file_digest, json_value,
                                 write_json)
from .controller import FlatMaskedActorCriticPolicy, controller_policy_kwargs
from .vae import BINARY, CATEGORICAL, CorpusTokenSchema


UPSTREAM_LOKI_COMMIT = "24aaac7eb2ecd8f1aada1e1ed0e2d91577fb578a"


@dataclass(frozen=True)
class SingleClusterConfig:
    elite_size: int = 20
    search_frequency: int = 2
    candidates_per_search: int = 128
    replacements_per_search: int = 2
    candidate_horizon: int = 200
    rollout_steps: int = 2560
    ppo_epochs: int = 8
    ppo_batch_size: int = 5120
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    adam_epsilon: float = 1e-5
    target_kl: float = 0.2 / 1.5
    max_iterations: int = 4

    def validate(self):
        integer_values = (
            self.elite_size, self.search_frequency, self.candidates_per_search,
            self.replacements_per_search, self.candidate_horizon,
            self.rollout_steps, self.ppo_epochs, self.ppo_batch_size,
            self.max_iterations)
        if min(integer_values) < 1:
            raise ValueError("LOKI counts and PPO dimensions must be positive")
        if self.replacements_per_search > self.elite_size:
            raise ValueError("Cannot replace more slots than the elite pool contains")
        if self.replacements_per_search > self.candidates_per_search:
            raise ValueError("Replacement count exceeds candidate count")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid PPO discount or GAE lambda")
        if min(self.learning_rate, self.clip_range, self.value_coefficient,
               self.max_grad_norm, self.adam_epsilon, self.target_kl) <= 0:
            raise ValueError("PPO floating-point settings must be positive")
        if self.entropy_coefficient < 0:
            raise ValueError("Entropy coefficient must be nonnegative")


class UpstreamLearningRate:
    """Upstream cosine schedule with its five-iteration 0.1 warmup."""

    def __init__(self, base_rate, max_iterations, iteration=0):
        self.base_rate = float(base_rate)
        self.max_iterations = int(max_iterations)
        self.iteration = int(iteration)

    def __call__(self, _progress_remaining):
        progress = min(max(self.iteration / max(self.max_iterations, 1), 0.0), 1.0)
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.iteration < 5:
            alpha = self.iteration / 5.0
            multiplier *= 0.1 * (1.0 - alpha) + alpha
        return self.base_rate * multiplier


class ClusterDesignStore:
    """Validated frozen cluster IDs with bounded-cache random corpus access."""

    def __init__(self, corpus, clusters, cluster_id, *, cache_shards=8):
        self.corpus_path = Path(corpus).resolve()
        self.clusters_path = Path(clusters).resolve()
        self.reader = CorpusReader(self.corpus_path)
        self.schema = CorpusTokenSchema.from_corpus(self.corpus_path)
        manifest_path = self.clusters_path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing cluster manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("status") != "complete":
            raise ValueError("Single-cluster training requires completed clusters")
        n_clusters = int(self.manifest["clustering_config"]["n_clusters"])
        if not 0 <= cluster_id < n_clusters:
            raise ValueError(f"cluster-id must be in [0, {n_clusters})")
        identity = self.manifest.get("corpus_identity", {})
        if identity.get("manifest_sha256") != file_digest(
                self.corpus_path / "manifest.json"):
            raise ValueError("Cluster artifact belongs to a different corpus manifest")
        if identity.get("constants_sha256") != file_digest(
                self.corpus_path / "constants.npz"):
            raise ValueError("Cluster artifact belongs to different corpus constants")
        accepted = int(self.reader.manifest["state"]["stats"]["accepted"])
        if int(self.manifest.get("design_count", -1)) != accepted:
            raise ValueError("Cluster design count differs from the corpus")
        index_path = self.clusters_path / "clusters" / f"{cluster_id}.npy"
        self.cluster_ids = np.load(index_path, allow_pickle=False)
        if (self.cluster_ids.dtype.kind not in "iu" or self.cluster_ids.ndim != 1
                or len(self.cluster_ids) == 0
                or np.any(self.cluster_ids < 0)
                or np.any(self.cluster_ids >= accepted)
                or len(np.unique(self.cluster_ids)) != len(self.cluster_ids)):
            raise ValueError("Invalid cluster design index")
        self.cluster_ids = self.cluster_ids.astype(np.int64, copy=False)
        self.cluster_id = int(cluster_id)
        self.identity = {
            "cluster_manifest_sha256": file_digest(manifest_path),
            "corpus_manifest_sha256": identity["manifest_sha256"],
            "cluster_id": self.cluster_id,
        }
        self._ranges = None
        self._cache = OrderedDict()
        self._cache_shards = int(cache_shards)

    def sample_ids(self, rng, count):
        if count > len(self.cluster_ids):
            raise ValueError("Cluster contains fewer designs than requested")
        return np.asarray(rng.choice(self.cluster_ids, count, replace=False),
                          dtype=np.int64)

    def _index_shards(self):
        if self._ranges is not None:
            return
        ranges = []
        for folder in sorted((self.corpus_path / "shards").iterdir()):
            with np.load(folder / "data.npz", allow_pickle=False) as data:
                ids = data["ids"]
                if len(ids):
                    ranges.append((int(ids[0]), int(ids[-1]) + 1, folder))
        if not ranges or ranges[0][0] != 0:
            raise ValueError("Corpus shard index is empty or noncontiguous")
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] != current[0]:
                raise ValueError("Corpus shard index is noncontiguous")
        self._ranges = ranges
        self._starts = [item[0] for item in ranges]

    def _shard(self, folder):
        key = str(folder)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with np.load(folder / "data.npz", allow_pickle=False) as data:
            value = {name: data[name] for name in data.files}
        self._cache[key] = value
        if len(self._cache) > self._cache_shards:
            self._cache.popitem(last=False)
        return value

    def native_designs(self, design_ids):
        self._index_shards()
        result = []
        for design_id in np.asarray(design_ids, dtype=np.int64):
            location = bisect.bisect_right(self._starts, int(design_id)) - 1
            if location < 0 or int(design_id) >= self._ranges[location][1]:
                raise IndexError(f"Design ID {design_id} is outside the corpus")
            start, _, folder = self._ranges[location]
            shard = self._shard(folder)
            offset = int(design_id) - start
            if int(shard["ids"][offset]) != int(design_id):
                raise ValueError("Corpus shard ID differs from its index")
            row = {name: shard[name][offset] for name in self.reader.codec.fields}
            result.append(self.reader.codec.decode(row))
        return result

    def flat_designs(self, env, design_ids):
        values = [env.design_codec.encode(item)
                  for item in self.native_designs(design_ids)]
        result = np.stack(values).astype(np.float32, copy=False)
        expected = sum(group.count * group.width for group in self.schema.groups)
        if result.shape != (len(values), expected):
            raise ValueError(
                f"Environment flat design shape {result.shape} differs from corpus token width {expected}")
        return result


class ElitePool:
    """Fixed slots and upstream-style rolling per-walker training returns."""

    def __init__(self, design_ids, *, history_size=10):
        values = np.asarray(design_ids, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0 or len(np.unique(values)) != len(values):
            raise ValueError("Initial elite IDs must be a nonempty unique vector")
        self.design_ids = values.copy()
        self.histories = [deque(maxlen=history_size) for _ in values]
        self.installed_candidate_scores = [None] * len(values)

    def record(self, slot, episode_return):
        self.histories[int(slot)].append(float(episode_return))

    def training_scores(self):
        # Upstream assigns 1e6 to a walker without a completed episode, keeping
        # it out of the worst set unless every score is unavailable.
        return np.asarray([
            float(np.mean(history)) if history else 1e6
            for history in self.histories], dtype=np.float64)

    def assigned_ids(self, design_slots):
        return self.design_ids[np.asarray(design_slots, dtype=np.int64)]

    def replace_from_candidates(self, candidate_ids, candidate_scores,
                                rejected, count):
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        scores = np.asarray(candidate_scores, dtype=np.float64)
        rejected = np.asarray(rejected, dtype=bool)
        if not (candidate_ids.shape == scores.shape == rejected.shape):
            raise ValueError("Candidate IDs, scores, and rejection mask must align")
        valid = np.flatnonzero(~rejected & np.isfinite(scores))
        if len(valid) < count:
            raise ValueError("Too few valid scored candidates for an atomic replacement")
        best = valid[np.argsort(-scores[valid], kind="stable")[:count]]
        train_scores = self.training_scores()
        worst = np.argsort(train_scores, kind="stable")[:count]
        events = []
        for slot, candidate in zip(worst, best):
            events.append({
                "slot": int(slot),
                "old_design_id": int(self.design_ids[slot]),
                "old_training_score": float(train_scores[slot]),
                "new_design_id": int(candidate_ids[candidate]),
                "candidate_score": float(scores[candidate]),
            })
            # Literal LOKI: replacement does not require candidate improvement.
            self.design_ids[slot] = candidate_ids[candidate]
            self.installed_candidate_scores[slot] = float(scores[candidate])
            self.histories[slot].clear()
        return events

    def state_dict(self):
        return {
            "design_ids": self.design_ids.tolist(),
            "histories": [list(history) for history in self.histories],
            "installed_candidate_scores": self.installed_candidate_scores,
        }

    @classmethod
    def from_state_dict(cls, state):
        result = cls(state["design_ids"])
        for target, values in zip(result.histories, state["histories"]):
            target.extend(values)
        result.installed_candidate_scores = list(
            state["installed_candidate_scores"])
        return result


class TokenConditioner:
    """Append corpus-normalized flat values while retaining corpus token rows."""

    def __init__(self, env, schema):
        self.env, self.schema = env, schema
        self.state_space = env.single_observation_space
        self.design_width = sum(group.count * group.width
                                for group in schema.groups)
        if len(env.base_embodiment) != self.design_width:
            raise ValueError("Environment design width differs from corpus schema")
        self.observation_space = spaces.Box(
            low=np.concatenate((np.asarray(self.state_space.low, np.float32),
                                -np.ones(self.design_width, np.float32))),
            high=np.concatenate((np.asarray(self.state_space.high, np.float32),
                                 np.ones(self.design_width, np.float32))),
            dtype=np.float32)

    def normalize_designs(self, designs):
        value = torch.as_tensor(designs, dtype=torch.float32,
                                device=self.env.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        output, cursor = torch.zeros_like(value), 0
        for group in self.schema.groups:
            width = group.count * group.width
            stop = cursor + width
            low = np.asarray(group.low, dtype=np.float32)
            if group.kind == CATEGORICAL:
                low = low.reshape(group.count)
                high = low + np.asarray(group.high, dtype=np.float32) - 1
            elif group.kind == BINARY:
                low = np.zeros(group.count, np.float32)
                high = np.ones(group.count, np.float32)
            else:
                low = low.reshape(-1)
                high = np.asarray(group.high, dtype=np.float32).reshape(-1)
            low_t = torch.as_tensor(low, device=value.device).reshape(1, -1)
            high_t = torch.as_tensor(high, device=value.device).reshape(1, -1)
            span = high_t - low_t
            chunk = value[:, cursor:stop]
            output[:, cursor:stop] = torch.where(
                span > 0, 2 * (chunk - low_t) / span.clamp_min(1e-12) - 1,
                torch.zeros_like(chunk))
            cursor = stop
        return output.clamp(-1, 1)

    def __call__(self, observation, designs):
        obs = torch.as_tensor(observation, dtype=torch.float32,
                              device=self.env.device)
        design = torch.as_tensor(designs, dtype=torch.float32,
                                 device=self.env.device)
        if design.ndim == 1:
            design = design.unsqueeze(0)
        if design.shape[0] == 1:
            design = design.expand(self.env.num_envs, -1)
        elif design.shape[0] == getattr(self.env, "num_designs", self.env.num_envs):
            design = self.env.expand_design_values(design)
        elif design.shape[0] != self.env.num_envs:
            raise ValueError("Design count must match logical designs or worlds")
        return torch.cat((obs, self.normalize_designs(design)), dim=-1)


class TokenConditionedGymBatch:
    """GymBatch equivalent using corpus rather than environment design bounds."""

    def __init__(self, env, designs, schema):
        expected = getattr(env, "num_designs", env.num_envs)
        if len(designs) != expected:
            raise ValueError(f"received {len(designs)} designs for {expected} slots")
        self.env = env
        self.designs = torch.as_tensor(designs, dtype=torch.float32,
                                       device=env.device)
        self.conditioner = TokenConditioner(env, schema)
        self.single_observation_space = self.conditioner.observation_space
        self.frames = 0

    def reset(self, reset_mask=None):
        options = (None if reset_mask is None else
                   {"reset_mask": torch.as_tensor(
                       reset_mask, dtype=torch.bool, device=self.env.device)})
        obs, rejected, _ = self.env.reset(
            embodiment=self.designs, options=options)
        if bool(rejected.any()):
            raise ValueError("controller batch contains rejected corpus embodiments")
        return self.conditioner(obs, self.designs)

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        self.frames += self.env.num_envs
        done = terminated | truncated
        info = dict(info)
        info["time_outs"] = truncated & ~terminated
        if bool(done.any()):
            info["terminal_observation"] = self.conditioner(obs, self.designs)
            if not getattr(self.env, "autoresets_same_step", False):
                obs, rejected, _ = self.env.reset(
                    embodiment=self.designs, options={"reset_mask": done})
                if bool(rejected.any()):
                    raise ValueError("episode reset rejected a corpus embodiment")
        return self.conditioner(obs, self.designs), reward, done, info

    def close(self):
        self.env.close()


class TrackingSB3GymBatch(SB3GymBatch):
    """SB3 transport that attributes completed returns to stable elite slots."""

    def __init__(self, batch, pool, logical_slots):
        self.pool = pool
        super().__init__(batch)
        self.set_designs(batch.designs, logical_slots)

    def set_designs(self, designs, logical_slots):
        slots = np.asarray(logical_slots, dtype=np.int64)
        if len(slots) != getattr(self.batch.env, "num_designs", self.batch.env.num_envs):
            raise ValueError("Elite slot assignment must match logical design slots")
        self.batch.designs = torch.as_tensor(
            designs, dtype=torch.float32, device=self.batch.env.device)
        self.physical_slots = np.repeat(
            slots, getattr(self.batch.env, "worlds_per_design", 1))
        self.slot_returns = torch.zeros(
            self.batch.env.num_envs, device=self.batch.env.device)

    def replace_designs(self, designs, logical_slots, changed_slots):
        """Install replacements while preserving every unaffected trajectory."""
        slots = np.asarray(logical_slots, dtype=np.int64)
        if len(slots) != getattr(
                self.batch.env, "num_designs", self.batch.env.num_envs):
            raise ValueError("Elite slot assignment must match logical design slots")
        changed_slots = np.asarray(changed_slots, dtype=np.int64)
        if changed_slots.ndim != 1 or len(changed_slots) == 0:
            raise ValueError("Selective installation requires changed elite slots")
        logical_mask = np.isin(slots, changed_slots)
        physical_mask = np.repeat(
            logical_mask, getattr(self.batch.env, "worlds_per_design", 1))
        if not physical_mask.any():
            raise ValueError("No training worlds map to the changed elite slots")

        self.batch.designs = torch.as_tensor(
            designs, dtype=torch.float32, device=self.batch.env.device)
        self.physical_slots = np.repeat(
            slots, getattr(self.batch.env, "worlds_per_design", 1))
        observation = self.batch.reset(reset_mask=physical_mask)
        mask = torch.as_tensor(
            physical_mask, dtype=torch.bool, device=self.batch.env.device)
        self.episode_returns[mask] = 0
        self.slot_returns[mask] = 0
        return observation.cpu().numpy(), physical_mask

    def reset(self):
        value = super().reset()
        self.slot_returns.zero_()
        return value

    def step_wait(self):
        obs, rewards, dones, rows = super().step_wait()
        reward_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.batch.env.device)
        self.slot_returns += reward_tensor
        for index in np.flatnonzero(dones):
            self.pool.record(self.physical_slots[index],
                             float(self.slot_returns[index]))
            self.slot_returns[index] = 0
        return obs, rewards, dones, rows


@dataclass(frozen=True)
class CandidateScoreResult:
    """One candidate batch, including incomplete deadline-limited attempts."""

    scores: np.ndarray
    rejected: np.ndarray
    frames: int
    complete: bool


@dataclass(frozen=True)
class SearchRoundResult:
    """Atomic search outcome published by a cluster runtime."""

    cluster_id: int
    policy_version: int
    candidate_ids: np.ndarray
    scores: np.ndarray
    rejected: np.ndarray
    frames: int
    complete: bool
    replacements: tuple[dict, ...] = ()


class EvaluationSelection:
    """Fresh policy-versioned ranking batches used for cross-cluster selection."""

    def __init__(self):
        self.rankings = {}

    def update(self, cluster_id, policy_version, design_ids, scores, rejected,
               *, elapsed_seconds=None):
        design_ids = np.asarray(design_ids, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        rejected = np.asarray(rejected, dtype=bool)
        if design_ids.ndim != 1 or not (
                design_ids.shape == scores.shape == rejected.shape):
            raise ValueError("Ranking IDs, scores, and rejection mask must align")
        if len(design_ids) == 0:
            raise ValueError("Ranking batch must contain at least one design")
        self.rankings[int(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "policy_version": int(policy_version),
            "design_ids": design_ids.copy(),
            "scores": scores.copy(),
            "rejected": rejected.copy(),
            "elapsed_seconds": (None if elapsed_seconds is None
                                else float(elapsed_seconds)),
        }

    @staticmethod
    def _top_distinct(ranking, count):
        valid = np.flatnonzero(
            ~ranking["rejected"] & np.isfinite(ranking["scores"]))
        order = valid[np.argsort(-ranking["scores"][valid], kind="stable")]
        selected, seen = [], set()
        for index in order:
            design_id = int(ranking["design_ids"][index])
            if design_id in seen:
                continue
            seen.add(design_id)
            selected.append(int(index))
            if len(selected) == count:
                break
        if len(selected) < count:
            raise ValueError(
                f"Cluster {ranking['cluster_id']} has fewer than {count} "
                "distinct valid freshly scored designs")
        return np.asarray(selected, dtype=np.int64)

    def select(self, count, *, cluster_ids=None):
        if count < 1:
            raise ValueError("Evaluation design count must be positive")
        requested = (sorted(self.rankings) if cluster_ids is None
                     else sorted(int(value) for value in cluster_ids))
        if not requested:
            raise ValueError("No cluster rankings are available")
        missing = [value for value in requested if value not in self.rankings]
        if missing:
            raise ValueError(f"Missing fresh rankings for clusters {missing}")
        versions = {self.rankings[value]["policy_version"]
                    for value in requested}
        if len(versions) != 1:
            raise ValueError(
                "Cross-cluster rankings must use one synchronized policy version")
        candidates = []
        for cluster_id in requested:
            ranking = self.rankings[cluster_id]
            selected = self._top_distinct(ranking, count)
            selected_scores = ranking["scores"][selected]
            candidates.append({
                "cluster_id": cluster_id,
                "policy_version": ranking["policy_version"],
                "design_ids": ranking["design_ids"][selected].copy(),
                "scores": selected_scores.copy(),
                "cluster_score": float(selected_scores.mean()),
                "elapsed_seconds": ranking["elapsed_seconds"],
            })
        # Highest mean wins; stable cluster order supplies the deterministic tie.
        return max(candidates, key=lambda item: item["cluster_score"])

    def state_dict(self):
        return {str(cluster_id): {
            "cluster_id": ranking["cluster_id"],
            "policy_version": ranking["policy_version"],
            "design_ids": ranking["design_ids"].tolist(),
            "scores": ranking["scores"].tolist(),
            "rejected": ranking["rejected"].tolist(),
            "elapsed_seconds": ranking["elapsed_seconds"],
        } for cluster_id, ranking in self.rankings.items()}

    @classmethod
    def from_state_dict(cls, state):
        result = cls()
        for ranking in state.values():
            result.update(
                ranking["cluster_id"], ranking["policy_version"],
                ranking["design_ids"], ranking["scores"], ranking["rejected"],
                elapsed_seconds=ranking.get("elapsed_seconds"))
        return result


class CandidateScorer:
    """Upstream-style stochastic candidate scoring with atomic chunks."""

    def __init__(self, env_id, schema, *, device, seed, env_kwargs=None,
                 rejection_reward=None):
        self.env_id, self.schema = env_id, schema
        self.device, self.seed = device, int(seed)
        self.env_kwargs = dict(env_kwargs or {})
        self.rejection_reward = rejection_reward

    @staticmethod
    def _expired(deadline):
        return deadline is not None and deadline.expired

    def _score_chunk(self, model, normalizer, designs, *, round_index,
                     chunk_index, horizon, deadline=None, progress=None):
        env, _ = make_baseline_vec(
            self.env_id, len(designs), device=self.device,
            seed=self.seed + 1_000_003 + round_index + chunk_index * 10_007,
            is_eval_env=True, control_mode="native",
            rejection_reward=self.rejection_reward, **self.env_kwargs)
        conditioner = TokenConditioner(env, self.schema)
        physical = env.num_envs
        episode_return = torch.zeros(physical, device=env.device)
        completed = [[] for _ in range(physical)]
        rejected_worlds = torch.zeros(physical, dtype=torch.bool,
                                      device=env.device)
        frames = 0
        try:
            design_tensor = torch.as_tensor(
                designs, dtype=torch.float32, device=env.device)
            obs, rejected, _ = env.reset(
                seed=self.seed + 2_000_003 + round_index + chunk_index * 10_007,
                embodiment=design_tensor)
            rejected_worlds |= rejected.reshape(-1)
            for _ in range(horizon):
                if progress is not None:
                    progress()
                if self._expired(deadline):
                    return None, None, frames, False
                conditioned = conditioner(obs, design_tensor).cpu().numpy()
                normalized = normalizer.normalize_obs(conditioned)
                actions, _ = model.predict(normalized, deterministic=False)
                obs, reward, terminated, truncated, _ = env.step(actions)
                done = (terminated | truncated).reshape(-1)
                episode_return += reward.reshape(-1)
                for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
                    if not rejected_worlds[index]:
                        completed[index].append(float(episode_return[index]))
                    episode_return[index] = 0
                if (bool(done.any()) and
                        not getattr(env, "autoresets_same_step", False)):
                    obs, reset_rejected, _ = env.reset(
                        embodiment=design_tensor, options={"reset_mask": done})
                    rejected_worlds |= reset_rejected.reshape(-1)
                frames += physical
            world_scores = np.asarray([
                float(np.median(values)) if values else float(episode_return[index])
                for index, values in enumerate(completed)], dtype=np.float64)
            worlds_per_design = getattr(env, "worlds_per_design", 1)
            scores = world_scores.reshape(len(designs), worlds_per_design).mean(axis=1)
            rejected = rejected_worlds.reshape(
                len(designs), worlds_per_design).any(dim=1).cpu().numpy()
            scores[rejected] = -np.inf
            return scores, rejected, frames, True
        finally:
            env.close()

    def score(self, model, normalizer, designs, *, round_index, horizon,
              batch_size=None, deadline=None, progress=None):
        designs = np.asarray(designs, dtype=np.float32)
        if designs.ndim != 2 or len(designs) == 0:
            raise ValueError("Candidate designs must be a nonempty matrix")
        batch_size = len(designs) if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("Candidate batch size must be positive")
        scores = np.full(len(designs), -np.inf, dtype=np.float64)
        rejected = np.ones(len(designs), dtype=bool)
        frames = 0
        for chunk_index, chunk_start in enumerate(
                range(0, len(designs), batch_size)):
            if self._expired(deadline):
                return CandidateScoreResult(scores, rejected, frames, False)
            chunk_stop = min(chunk_start + batch_size, len(designs))
            values = self._score_chunk(
                model, normalizer, designs[chunk_start:chunk_stop],
                round_index=round_index, chunk_index=chunk_index,
                horizon=horizon, deadline=deadline, progress=progress)
            chunk_scores, chunk_rejected, chunk_frames, complete = values
            frames += chunk_frames
            if not complete:
                return CandidateScoreResult(scores, rejected, frames, False)
            scores[chunk_start:chunk_stop] = chunk_scores
            rejected[chunk_start:chunk_stop] = chunk_rejected
        return CandidateScoreResult(scores, rejected, frames, True)


def _logical_slots(num_designs, elite_size):
    return np.arange(num_designs, dtype=np.int64) % elite_size


def _initialize_upstream_heads(model):
    # Released controller initializes the final actor and critic projections
    # uniformly in [-0.01, 0.01], with zero biases.
    for layer in (model.policy.action_net, model.policy.value_net):
        torch.nn.init.uniform_(layer.weight, -0.01, 0.01)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


class ClusterRuntime:
    """Independent train/search state for one frozen LOKI cluster."""

    def __init__(self, *, config, store, pool, rng, training_env, vector,
                 normalizer, model, scorer, logical_slots, counters=None,
                 torch_rng=None, cuda_rng=None):
        self.config = config
        self.store = store
        self.pool = pool
        self.rng = rng
        self.training_env = training_env
        self.vector = vector
        self.normalizer = normalizer
        self.model = model
        self.scorer = scorer
        self.logical_slots = np.asarray(logical_slots, dtype=np.int64)
        self.counters = dict(counters or {
            "ppo_frames": 0,
            "candidate_frames": 0,
            "search_rounds": 0,
            "incomplete_search_rounds": 0,
            "candidate_rejections": 0,
        })
        self.counters.setdefault("incomplete_search_rounds", 0)
        self.counters.setdefault("elite_replacements", 0)
        self.torch_rng = (torch.get_rng_state().clone() if torch_rng is None
                          else torch_rng.clone())
        if cuda_rng is None:
            self.cuda_rng = ([value.clone() for value in
                              torch.cuda.get_rng_state_all()]
                             if torch.cuda.is_available() else None)
        else:
            self.cuda_rng = [value.clone() for value in cuda_rng]

    @contextmanager
    def rng_scope(self):
        """Install this cluster's generators without perturbing its siblings."""
        outer_torch = torch.get_rng_state()
        outer_cuda = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_rng)
        try:
            yield
        finally:
            self.torch_rng = torch.get_rng_state().clone()
            if torch.cuda.is_available():
                self.cuda_rng = [value.clone() for value in
                                 torch.cuda.get_rng_state_all()]
            torch.set_rng_state(outer_torch)
            if outer_cuda is not None:
                torch.cuda.set_rng_state_all(outer_cuda)

    @property
    def cluster_id(self):
        return self.store.cluster_id

    @property
    def completed_iterations(self):
        return int(self.model.completed_ppo_iterations)

    @property
    def policy_version(self):
        return self.completed_iterations

    def install_pool(self, changed_slots=None):
        designs = self.store.flat_designs(
            self.training_env, self.pool.assigned_ids(self.logical_slots))
        if (changed_slots is None or
                not getattr(self.training_env, "supports_selective_reset", True)):
            self.vector.set_designs(designs, self.logical_slots)
            self.model._last_obs = self.normalizer.reset()
            self.model._last_episode_starts = np.ones(
                self.normalizer.num_envs, dtype=bool)
            return

        raw_obs, physical_mask = self.vector.replace_designs(
            designs, self.logical_slots, changed_slots)
        self.normalizer.old_obs = raw_obs
        self.normalizer.returns[physical_mask] = 0
        if self.normalizer.training and self.normalizer.norm_obs:
            self.normalizer.obs_rms.update(raw_obs[physical_mask])
        self.model._last_obs = self.normalizer.normalize_obs(raw_obs)
        self.model._last_episode_starts[physical_mask] = True

    def train_iteration(self, *, budget=None, progress=None):
        """Attempt one complete PPO iteration and account actual transitions."""
        if budget is not None and budget.expired:
            return {"complete": False, "frames": 0,
                    "iteration": self.completed_iterations}
        before_frames = int(self.vector.batch.frames)
        before_iterations = self.completed_iterations
        if isinstance(self.model.lr_schedule, UpstreamLearningRate):
            self.model.lr_schedule.iteration = before_iterations
        self.model.budget = budget
        self.model.progress_hook = progress

        # SB3 checks this callback during rollout collection. TimedPPO separately
        # checks the same budget between optimizer minibatches.
        callback = None
        if budget is not None or progress is not None:
            from stable_baselines3.common.callbacks import BaseCallback

            class DeadlineCallback(BaseCallback):
                def _on_step(callback_self):
                    if progress is not None:
                        progress()
                    return budget is None or not budget.expired

            callback = DeadlineCallback()
        try:
            with self.rng_scope():
                self.model.learn(
                    self.config.rollout_steps * self.normalizer.num_envs,
                    reset_num_timesteps=False, callback=callback)
        except TrainingDeadlineReached:
            pass
        finally:
            self.model.budget = None
            self.model.progress_hook = None
        frames = int(self.vector.batch.frames) - before_frames
        self.counters["ppo_frames"] += frames
        complete = self.completed_iterations == before_iterations + 1
        return {"complete": complete, "frames": frames,
                "iteration": self.completed_iterations}

    def score_search(self, *, deadline=None, batch_size=None, progress=None):
        """Score candidates without mutating the elite pool."""
        candidate_ids = self.store.sample_ids(
            self.rng, self.config.candidates_per_search)
        candidate_designs = self.store.flat_designs(
            self.training_env, candidate_ids)
        with self.rng_scope():
            scored = self.scorer.score(
                self.model, self.normalizer, candidate_designs,
                round_index=self.counters["search_rounds"],
                horizon=self.config.candidate_horizon,
                batch_size=batch_size, deadline=deadline, progress=progress)
        self.counters["candidate_frames"] += int(scored.frames)
        if not scored.complete:
            self.counters["incomplete_search_rounds"] += 1
        else:
            self.counters["candidate_rejections"] += int(scored.rejected.sum())
        return SearchRoundResult(
            self.cluster_id, self.policy_version, candidate_ids,
            scored.scores, scored.rejected, scored.frames, scored.complete)

    def commit_search(self, scored):
        """Atomically install a previously completed score batch once."""
        if not scored.complete:
            raise ValueError("Cannot commit an incomplete candidate round")
        if scored.cluster_id != self.cluster_id:
            raise ValueError("Candidate round belongs to another cluster")
        if scored.policy_version != self.policy_version:
            raise ValueError("Candidate round was scored by a stale policy")
        frequency = self.config.search_frequency
        if (self.policy_version % frequency != 0 or
                self.counters["search_rounds"] >= self.policy_version // frequency):
            raise ValueError("This policy version has no pending search round")
        replacements = self.pool.replace_from_candidates(
            scored.candidate_ids, scored.scores, scored.rejected,
            self.config.replacements_per_search)
        self.counters["search_rounds"] += 1
        self.counters["elite_replacements"] += len(replacements)
        with self.rng_scope():
            self.install_pool([event["slot"] for event in replacements])
        return SearchRoundResult(
            scored.cluster_id, scored.policy_version, scored.candidate_ids,
            scored.scores, scored.rejected, scored.frames, True,
            tuple(replacements))

    def search(self, *, deadline=None, batch_size=None, progress=None):
        """Convenience path for one cluster's score-and-commit transaction."""
        scored = self.score_search(
            deadline=deadline, batch_size=batch_size, progress=progress)
        return self.commit_search(scored) if scored.complete else scored

    def score_elites(self, *, deadline=None, batch_size=None, progress=None):
        """Freshly score the current elite IDs without changing training state."""
        design_ids = self.pool.design_ids.copy()
        designs = self.store.flat_designs(self.training_env, design_ids)
        with self.rng_scope():
            scored = self.scorer.score(
                self.model, self.normalizer, designs,
                round_index=1_000_000 + self.counters["search_rounds"],
                horizon=self.config.candidate_horizon,
                batch_size=batch_size, deadline=deadline, progress=progress)
        self.counters.setdefault("ranking_frames", 0)
        self.counters.setdefault("incomplete_ranking_rounds", 0)
        self.counters.setdefault("ranking_rejections", 0)
        self.counters["ranking_frames"] += int(scored.frames)
        if scored.complete:
            self.counters["ranking_rejections"] += int(scored.rejected.sum())
        else:
            self.counters["incomplete_ranking_rounds"] += 1
        return SearchRoundResult(
            self.cluster_id, self.policy_version, design_ids,
            scored.scores, scored.rejected, scored.frames, scored.complete)

    def state_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "policy_version": self.policy_version,
            "pool": self.pool.state_dict(),
            "numpy_rng": self.rng.bit_generator.state,
            "torch_rng": self.torch_rng.clone(),
            "cuda_rng": (None if self.cuda_rng is None else
                         [value.clone() for value in self.cuda_rng]),
            "counters": dict(self.counters),
        }

    def close(self):
        self.normalizer.close()


def create_cluster_runtime(env_id, corpus, clusters, cluster_id, *, num_envs,
                           device="cuda:0", seed=0, env_kwargs=None,
                           rejection_reward=None,
                           config=SingleClusterConfig(), store=None,
                           checkpoint_folder=None, runtime_state=None):
    """Construct one independent controller, simulator batch, and search state."""
    config.validate()
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    store = store or ClusterDesignStore(corpus, clusters, cluster_id)
    if store.cluster_id != int(cluster_id):
        raise ValueError("Provided design store belongs to another cluster")
    if len(store.cluster_ids) < max(config.elite_size,
                                    config.candidates_per_search):
        raise ValueError("Cluster is too small for the configured pools")
    if (runtime_state is None) != (checkpoint_folder is None):
        raise ValueError(
            "Runtime state and checkpoint folder must be provided together")

    rng = np.random.default_rng(seed)
    if runtime_state is None:
        pool = ElitePool(store.sample_ids(rng, config.elite_size))
        counters = None
    else:
        pool = ElitePool.from_state_dict(runtime_state["pool"])
        rng.bit_generator.state = runtime_state["numpy_rng"]
        counters = runtime_state["counters"]

    training_env = vector = normalizer = None
    try:
        training_env, _ = make_baseline_vec(
            env_id, num_envs, device=device, seed=seed,
            control_mode="native", rejection_reward=rejection_reward,
            **(env_kwargs or {}))
        logical_slots = _logical_slots(
            training_env.num_designs, config.elite_size)
        training_designs = store.flat_designs(
            training_env, pool.assigned_ids(logical_slots))
        batch = TokenConditionedGymBatch(
            training_env, training_designs, store.schema)
        vector = TrackingSB3GymBatch(batch, pool, logical_slots)
        if runtime_state is not None:
            checkpoint_folder = Path(checkpoint_folder)
            normalizer = VecNormalize.load(
                checkpoint_folder / "vecnormalize.pkl", vector)
            normalizer.training = True
            normalizer.norm_reward = False
            model = TimedPPO.load(
                checkpoint_folder / "policy.zip", env=normalizer,
                device=device, budget=None)
            if isinstance(model.lr_schedule, UpstreamLearningRate):
                model.lr_schedule.max_iterations = config.max_iterations
        else:
            normalizer = VecNormalize(
                vector, training=True, norm_obs=True, norm_reward=False,
                gamma=config.gamma, clip_obs=10.0)
            rollout_size = config.rollout_steps * normalizer.num_envs
            if config.ppo_batch_size > rollout_size:
                raise ValueError("PPO batch size exceeds one rollout")
            policy_kwargs = controller_policy_kwargs(
                store.schema,
                int(np.prod(training_env.single_observation_space.shape)))
            policy = "MlpPolicy"
            action_mask_slice = getattr(training_env, "action_mask_slice", None)
            if action_mask_slice is not None:
                policy = FlatMaskedActorCriticPolicy
                policy_kwargs["action_mask_slice"] = action_mask_slice
            model = TimedPPO(
                policy, normalizer, n_steps=config.rollout_steps,
                batch_size=config.ppo_batch_size,
                learning_rate=UpstreamLearningRate(
                    config.learning_rate, config.max_iterations),
                gamma=config.gamma, gae_lambda=config.gae_lambda,
                clip_range=config.clip_range, clip_range_vf=config.clip_range,
                normalize_advantage=True,
                ent_coef=config.entropy_coefficient,
                vf_coef=config.value_coefficient,
                max_grad_norm=config.max_grad_norm,
                n_epochs=config.ppo_epochs, target_kl=config.target_kl,
                seed=seed, device=device, budget=None,
                policy_kwargs=policy_kwargs)
            _initialize_upstream_heads(model)
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.requires_grad_(False)
        scorer = CandidateScorer(
            env_id, store.schema, device=device, seed=seed,
            env_kwargs=env_kwargs, rejection_reward=rejection_reward)
        result = ClusterRuntime(
            config=config, store=store, pool=pool, rng=rng,
            training_env=training_env, vector=vector,
            normalizer=normalizer, model=model, scorer=scorer,
            logical_slots=logical_slots, counters=counters,
            torch_rng=(None if runtime_state is None else
                       runtime_state["torch_rng"]),
            cuda_rng=(None if runtime_state is None else
                      runtime_state["cuda_rng"]))
        if (runtime_state is not None and result.completed_iterations !=
                int(runtime_state["iteration"])):
            raise ValueError("Policy PPO iteration differs from checkpoint runtime")
        return result
    except Exception:
        if normalizer is not None:
            normalizer.close()
        elif vector is not None:
            vector.close()
        elif training_env is not None:
            training_env.close()
        raise


def _atomic_torch_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_checkpoint(output, iteration, model, normalizer, pool, rng, config,
                    store, run_identity, counters, *, status,
                    cluster_runtime=None):
    root = Path(output) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    name = f"{iteration:06d}"
    destination = root / name
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    staging = root / f".{name}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    model.save(staging / "policy.zip")
    normalizer.save(staging / "vecnormalize.pkl")
    runtime = {
        "format_version": 1,
        "iteration": int(iteration),
        "config": asdict(config),
        "pool": pool.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": (cluster_runtime.torch_rng.clone()
                      if cluster_runtime is not None else torch.get_rng_state()),
        "cuda_rng": ((None if cluster_runtime.cuda_rng is None else
                      [value.clone() for value in cluster_runtime.cuda_rng])
                     if cluster_runtime is not None else
                     (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)),
        "cluster_identity": store.identity,
        "run_identity": run_identity,
        "counters": counters,
    }
    _atomic_torch_save(staging / "runtime.pt", runtime)
    write_json(staging / "manifest.json", {
        "format_version": 1, "status": status, "iteration": iteration,
        "cluster_identity": store.identity, "run_identity": run_identity,
        "config": asdict(config), "counters": counters,
        "upstream_loki_commit": UPSTREAM_LOKI_COMMIT,
    })
    staging.rename(destination)
    temporary = root / ".latest.tmp"
    write_json(temporary, {"checkpoint": name, "status": status})
    temporary.replace(root / "latest.json")
    return destination


def _load_runtime(output, config, store, run_identity):
    latest = Path(output) / "checkpoints" / "latest.json"
    if not latest.is_file():
        raise FileNotFoundError("--resume requested but no latest checkpoint exists")
    folder = latest.parent / json.loads(latest.read_text())["checkpoint"]
    runtime = torch.load(folder / "runtime.pt", map_location="cpu",
                         weights_only=False)
    if runtime.get("format_version") != 1:
        raise ValueError("Unsupported single-cluster checkpoint format")
    saved = dict(runtime["config"])
    requested = asdict(config)
    saved_iterations = saved.pop("max_iterations")
    requested_iterations = requested.pop("max_iterations")
    if saved != requested:
        raise ValueError("Resume configuration differs from the checkpoint")
    if requested_iterations < runtime["iteration"]:
        raise ValueError("Requested iterations precede the checkpoint")
    if runtime["cluster_identity"] != store.identity:
        raise ValueError("Resume cluster artifact differs from the checkpoint")
    if runtime.get("run_identity") != run_identity:
        raise ValueError("Resume task, seed, environments, or options differ from the checkpoint")
    return folder, runtime, saved_iterations


def train_single_cluster(env_id, corpus, clusters, cluster_id, output, *,
                         num_envs, device="cuda:0", seed=0, env_kwargs=None,
                         rejection_reward=None, config=SingleClusterConfig(),
                         resume=False):
    """Run a bounded one-cluster PPO/search experiment and return its summary."""
    config.validate()
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    store = ClusterDesignStore(corpus, clusters, cluster_id)
    run_identity = json_value({
        "env_id": env_id, "seed": int(seed),
        "requested_num_envs": int(num_envs),
        "env_kwargs": dict(env_kwargs or {}),
        "rejection_reward": rejection_reward,
    })
    checkpoint_folder = saved_runtime = None
    if resume:
        checkpoint_folder, saved_runtime, _ = _load_runtime(
            output, config, store, run_identity)
    start_iteration = (0 if saved_runtime is None
                       else int(saved_runtime["iteration"]))

    cluster_runtime = None
    history_path = output / "history.jsonl"
    try:
        cluster_runtime = create_cluster_runtime(
            env_id, corpus, clusters, cluster_id, num_envs=num_envs,
            device=device, seed=seed, env_kwargs=env_kwargs,
            rejection_reward=rejection_reward, config=config, store=store,
            checkpoint_folder=checkpoint_folder,
            runtime_state=saved_runtime)
        model = cluster_runtime.model
        normalizer = cluster_runtime.normalizer
        pool = cluster_runtime.pool
        rng = cluster_runtime.rng
        training_env = cluster_runtime.training_env

        for _ in range(start_iteration, config.max_iterations):
            trained = cluster_runtime.train_iteration()
            if not trained["complete"]:
                raise RuntimeError("Iteration-bounded PPO stopped before an update")
            completed = cluster_runtime.completed_iterations
            replacements = []
            if completed % config.search_frequency == 0:
                search = cluster_runtime.search()
                if not search.complete:
                    raise RuntimeError(
                        "Iteration-bounded candidate search was incomplete")
                replacements = list(search.replacements)
            counters = cluster_runtime.counters
            status = ("complete" if completed == config.max_iterations
                      else "running")
            checkpoint = save_checkpoint(
                output, completed, model, normalizer, pool, rng, config,
                store, run_identity, counters, status=status,
                cluster_runtime=cluster_runtime)
            record = {
                "iteration": completed,
                "status": status,
                "policy_version": cluster_runtime.policy_version,
                "learning_rate": float(model.lr_schedule(0.0)),
                "training_scores": pool.training_scores().tolist(),
                "elite_design_ids": pool.design_ids.tolist(),
                "replacements": replacements,
                "counters": dict(counters),
                "checkpoint": str(checkpoint),
            }
            with history_path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True,
                                        allow_nan=False) + "\n")
                stream.flush()
        result = {
            "status": "complete", "iterations": config.max_iterations,
            "cluster_id": cluster_id,
            "elite_design_ids": pool.design_ids.tolist(),
            "counters": cluster_runtime.counters,
            "policy_version": cluster_runtime.policy_version,
            "logical_design_slots": int(training_env.num_designs),
            "physical_worlds": int(training_env.num_envs),
            "output": str(output),
        }
        write_json(output / "result.json", result)
        return result
    finally:
        if cluster_runtime is not None:
            cluster_runtime.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=sorted(COLLECTORS), required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--clusters", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--elite-size", type=int, default=20)
    parser.add_argument("--search-frequency", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--replacements", type=int, default=2)
    parser.add_argument("--candidate-horizon", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=2560)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--ppo-batch-size", type=int, default=5120)
    parser.add_argument("--rejection-reward", type=float)
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as error:
        parser.error(f"env-kwargs must be a JSON object: {error}")
    if not isinstance(env_kwargs, dict):
        parser.error("env-kwargs must be a JSON object")
    canonical = PRESET_TO_CORPUS[args.env_id]
    corpus = args.corpus or REPO / "data/loki" / corpus_dir(canonical)
    clusters = args.clusters or REPO / "artifacts/loki" / corpus_dir(canonical) / "clusters"
    config = SingleClusterConfig(
        elite_size=args.elite_size,
        search_frequency=args.search_frequency,
        candidates_per_search=args.candidates,
        replacements_per_search=args.replacements,
        candidate_horizon=args.candidate_horizon,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        ppo_batch_size=args.ppo_batch_size,
        max_iterations=args.iterations)
    return args, env_kwargs, corpus, clusters, config


def main(argv=None):
    args, env_kwargs, corpus, clusters, config = parse_args(argv)
    result = train_single_cluster(
        args.env_id, corpus, clusters, args.cluster_id, args.output,
        num_envs=args.num_envs, device=args.device, seed=args.seed,
        env_kwargs=env_kwargs, rejection_reward=args.rejection_reward,
        config=config, resume=args.resume)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
