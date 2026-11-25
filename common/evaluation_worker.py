"""Held-out evaluation backends used exclusively in the evaluation process."""

import io
import json
import os
from pathlib import Path
import time

# Evaluation workers are independent headless processes. Select EGL before
# importing any environment module that may import MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

from .codesign_adapter import make_baseline_vec
from .gym_batch import EvaluationAccumulator
from .embodiment_conditioning import condition_observation


class SnapshotEvaluator:
    supports_metrics_callback = True

    def __init__(self, config):
        self.config = config
        self.method = config["method"]
        self.seed = config["seed"] + 3_000_003
        self.device = config["device"]
        self.output = Path(config["output_dir"])
        self.replay_dir = None
        env_kwargs = json.loads(config.get("env_kwargs", "{}"))
        if config["env_id"].startswith("Pokenv"):
            self.replay_dir = self.output / "evaluation_replays"
            self.replay_dir.mkdir(exist_ok=True)
            env_kwargs["save_replays"] = str(self.replay_dir)
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        self.env, _ = make_baseline_vec(config["env_id"], config["evaluation_worlds"],
            device=self.device, seed=self.seed, is_eval_env=True, render_mode="rgb_array",
            control_mode="native" if (self.method.startswith("ppo_") or
                                      self.method == "loki") else "logits",
            rejection_reward=config.get("rejection_reward"),
            mesh_bound_m=config.get("mesh_bound_m"), **env_kwargs)
        if self.method == "fasttd3_cmaes" and config.get("use_cbd", False):
            from fasttd3_cmaes.cbdsa_codec import apply_cbdsa_codec
            apply_cbdsa_codec(self.env, bound_m=config.get("mesh_bound_m"))
        self._last_replays = ()

    def _rollout(self, design, action, deadline, stop, selected=()):
        selected = list(selected)
        replay_before = set(self.replay_dir.glob("*.html")) if self.replay_dir else set()
        torch.manual_seed(self.seed)
        observation, rejected, _ = self.env.reset(seed=self.seed, embodiment=design)
        # Reset forfeits any same-step autoreset battle remaining from the prior panel.
        # Exclude those cleanup files from the current evaluation's replay set.
        if self.replay_dir:
            replay_before = set(self.replay_dir.glob("*.html"))
        result = EvaluationAccumulator(self.env)
        result.reject(rejected, self.env.rejection_reward)
        collapsed = self.env.collapse_rejections(rejected) if hasattr(self.env, "collapse_rejections") else rejected
        rejection_count = int(collapsed.sum())
        rejection_rate = rejection_count / max(len(collapsed), 1)
        initial_frames = {index: self.env.render(env_idx=index) for index in set(selected)}
        frames = [[initial_frames[index]] for index in selected]
        step = 0
        with torch.no_grad():
            while not result.selected_complete(selected) and (deadline is None or time.monotonic() < deadline) and not stop.is_set():
                observation, reward, terminated, truncated, _ = self.env.step(action(observation))
                done = (terminated | truncated).reshape(-1)
                step += 1
                rendered = {}
                for position, index in enumerate(selected):
                    if not bool(result.finished[index]) and (step % self.config["video_stride"] == 0 or bool(done[index])):
                        if index not in rendered:
                            rendered[index] = self.env.render(env_idx=index)
                        frames[position].append(rendered[index])
                result.add(reward, terminated, truncated)
        if self.replay_dir:
            from pokemon_common import collect_evaluation_replays
            self._last_replays = collect_evaluation_replays(
                self.env.env, self.replay_dir, replay_before, self.env.num_envs)
        return result, frames, rejection_count, rejection_rate

    def evaluate(self, path, *, deadline, stop, on_metrics=None):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        meta = snapshot["_evaluation"]
        metrics = {**meta["metrics"], "eval/snapshot_version": meta["version"],
            "eval/snapshot_elapsed_seconds": meta["elapsed_seconds"], "eval/snapshot_kind": meta["kind"]}
        started = time.monotonic()
        if self.method == "fasttd3":
            from fast_td3.codesign.evaluation import load_policy, evaluate, render_extremes
            _, policy = load_policy(path, self.device, compile=False)
            result = evaluate(self.env, policy, seed=self.seed, deadline=deadline, stop_event=stop)
            complete = result["status"] == "complete"
            rejection_count = result["embodiment_rejections"]
            rejection_rate = result["embodiment_rejection_rate"]
        else:
            if self.method == "loki":
                from ppo_ngopt_rlgo.timed_ppo import TimedPPO
                from loki.single_cluster import TokenConditioner
                policy = TimedPPO.load(
                    io.BytesIO(snapshot["ppo"]), device=self.device)
                conditioner = TokenConditioner(
                    self.env, snapshot["token_schema"])
                normalization = snapshot["normalizer"]
                mean = np.asarray(normalization["mean"], dtype=np.float32)
                variance = np.asarray(normalization["var"], dtype=np.float32)
                epsilon = float(normalization["epsilon"])
                clip = float(normalization["clip_obs"])

                def action(observation):
                    conditioned = conditioner(
                        observation, snapshot["design"]).cpu().numpy()
                    normalized = np.clip(
                        (conditioned - mean) / np.sqrt(variance + epsilon),
                        -clip, clip).astype(np.float32)
                    return policy.predict(normalized, deterministic=True)[0]
            elif self.method == "fasttd3_cmaes":
                # The CMA-ES design is fixed by the snapshot; the actor only controls.
                from fasttd3_cmaes.normalization import load_policy
                _, policy = load_policy(path, self.device, compile=False)
                action = lambda observation: self.env.project_control(policy(
                    condition_observation(self.env, observation, snapshot["design"]),
                    deterministic=True, update=False).clamp(-1, 1))
            elif self.method.startswith("ppo_"):
                from stable_baselines3 import PPO
                policy = PPO.load(io.BytesIO(snapshot["ppo"]), device=self.device)
                action = lambda observation: policy.predict(
                    (condition_observation(self.env, observation, snapshot["design"])
                     if snapshot.get("embodiment_conditioned", False) else observation).cpu().numpy(),
                    deterministic=True)[0]
            else:
                from cmaes_evolutionary.gym_train import control
                weights = torch.as_tensor(snapshot["policy"], dtype=torch.float32, device=self.device)
                action = lambda observation: control(
                    condition_observation(self.env, observation, snapshot["design"])
                    if snapshot.get("embodiment_conditioned", False) else observation,
                    weights, snapshot["action_size"])
            accumulator, _, rejection_count, rejection_rate = self._rollout(snapshot["design"], action, deadline, stop)
            complete = accumulator.complete
            result = accumulator.summary() if complete else {}
        metrics["eval/rollout_seconds"] = time.monotonic() - started
        metrics["eval/embodiment_rejections"] = rejection_count
        metrics["eval/embodiment_rejection_rate"] = rejection_rate
        if not complete:
            return {**metrics, "eval/status": "budget_exhausted" if deadline is not None and time.monotonic() >= deadline else "cancelled"}, {}
        metrics.update({"eval/status": "complete", "eval/reward": result["return_mean"],
            **{f"eval/{key}": result[key] for key in ("return_std", "length_mean", "success_rate", "divergence_rate")}})
        if on_metrics is not None:
            on_metrics(dict(metrics))  # Publish scores and save the checkpoint before rendering.
        if rejection_count:
            metrics["eval/video_complete"] = False
            metrics["eval/render_seconds"] = 0.0
            return metrics, {}
        values = result["episode_returns"]
        if self.replay_dir is not None:
            from pokemon_common import best_worst_replays
            best, worst = best_worst_replays({
                "episode_returns": values,
                "replays_by_env": self._last_replays,
            })
            paths = {name: str(path) for name, path in
                     (("best", best), ("worst", worst)) if path is not None}
            metrics["eval/replay_count"] = sum(
                path is not None for path in self._last_replays)
            metrics["eval/replay_complete"] = "best" in paths and "worst" in paths
            return metrics, paths
        indices = (max(range(len(values)), key=values.__getitem__), min(range(len(values)), key=values.__getitem__))
        started = time.monotonic()
        # Video work must not occupy the next scheduled evaluation boundary.
        render_deadline = min(deadline, started + self.config.get("video_budget_seconds", 60.0))
        paths = {}
        try:
            if self.method == "fasttd3":
                frames, video_complete = render_extremes(self.env, policy, seed=self.seed, indices=indices,
                    deadline=render_deadline, stride=self.config["video_stride"], stop_event=stop, return_status=True)
            else:
                accumulator, frames, _, _ = self._rollout(snapshot["design"], action, render_deadline, stop, indices)
                video_complete = accumulator.selected_complete(list(indices))
            if video_complete:
                import imageio.v2 as imageio
                folder = self.output / "evaluation_videos"
                folder.mkdir(exist_ok=True)
                for name, sequence in zip(("best", "worst"), frames):
                    sequence = [frame for frame in sequence if frame is not None]
                    if not sequence:
                        raise RuntimeError(f"evaluation produced no renderable {name} frames")
                    target = folder / f"{path.stem}_{name}.mp4"
                    imageio.mimwrite(target, sequence, fps=self.config["video_fps"],
                        codec="libx264", quality=7, macro_block_size=None)
                    paths[name] = str(target)
        except Exception as error:
            video_complete = False
            paths = {}
            metrics["eval/video_error"] = f"{type(error).__name__}: {error}"
        metrics["eval/render_seconds"] = time.monotonic() - started
        metrics["eval/video_complete"] = bool(video_complete)
        return metrics, paths

    def save_best(self, path, score):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if self.method == "fasttd3":
            snapshot["eval_reward"] = score
            torch.save(snapshot, self.output / "best_checkpoint.pt")
        elif self.method == "loki":
            snapshot["eval_reward"] = score
            torch.save(snapshot, self.output / "best_checkpoint.pt")
            np.save(self.output / "best_embodiments.npy", snapshot["design"])
        elif self.method == "fasttd3_cmaes":
            snapshot["eval_reward"] = score
            torch.save(snapshot, self.output / "best_checkpoint.pt")
            np.save(self.output / "best_embodiment.npy", snapshot["design"])
        elif self.method.startswith("ppo_"):
            (self.output / "best_checkpoint.zip").write_bytes(snapshot["ppo"])
            np.save(self.output / "best_embodiment.npy", snapshot["design"])
        else:
            np.savez(self.output / "best_checkpoint.npz", design=snapshot["design"], policy=snapshot["policy"],
                eval_reward=score, observation_size=snapshot["observation_size"],
                action_size=snapshot["action_size"],
                embodiment_conditioned=snapshot.get("embodiment_conditioned", False))

    def close(self):
        self.env.close()
