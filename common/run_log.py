"""JSONL metrics and optional W&B under one monotonic run clock."""

import json
import threading
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


class RunLog:
    def __init__(self, output, budget, config, *, project=None, entity=None, group=None, name=None):
        self._lock = threading.RLock()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.budget = budget
        versions = {}
        for package in ("torch", "stable-baselines3", "nevergrad", "cma", "mujoco", "mujoco-warp", "warp-lang", "codesign-gym", "pokemon-codesign-env"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                pass
        self.config = {**config, **budget.metadata(), "package_versions": versions}
        (self.output / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
        self.stream = (self.output / "metrics.jsonl").open("w", buffering=1)
        self.run = None
        if project:
            import wandb

            self.run = wandb.init(project=project, entity=entity, group=group, name=name, config=self.config, dir=str(self.output))

    def log(self, **metrics):
        row = {"elapsed_seconds": self.budget.elapsed, **metrics}
        with self._lock:
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            if self.run is not None:
                self.run.log(row)

    def log_evaluation(self, best_frames, worst_frames, *, fps=20, **metrics):
        import imageio.v2 as imageio

        boundary = metrics["evaluation_boundary_seconds"]
        videos = self.output / "evaluation_videos"
        videos.mkdir(exist_ok=True)
        tag = f"{boundary:012.3f}".replace(".", "_")
        paths = {}
        for name, frames in (("best", best_frames), ("worst", worst_frames)):
            frames = [frame for frame in frames if frame is not None]
            if not frames:
                raise RuntimeError(f"evaluation produced no renderable {name} frames")
            path = videos / f"boundary_{tag}_{name}.mp4"
            imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=None)
            paths[name] = path
        self.log_evaluation_paths(paths, **metrics)

    def log_evaluation_paths(self, paths, **metrics):
        """Log videos already encoded by an independent evaluation process."""
        if paths and all(Path(path).suffix.lower() == ".html" for path in paths.values()):
            return self.log_html_evaluation(
                paths.get("best"), paths.get("worst"), **metrics)
        row = {"elapsed_seconds": self.budget.elapsed, **metrics,
            **{f"eval/{name}_video_path": str(path) for name, path in paths.items()}}
        with self._lock:
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            if self.run is not None:
                import wandb
                payload = {key: value for key, value in row.items() if not key.endswith("_path")}
                payload.update({f"eval/{name}_video": wandb.Video(str(path), format="mp4")
                    for name, path in paths.items()})
                self.run.log(payload)

    def log_html_evaluation(self, best_replay, worst_replay, **metrics):
        paths = {name: Path(path) for name, path in (("best", best_replay), ("worst", worst_replay)) if path is not None}
        row = {"elapsed_seconds": self.budget.elapsed, **metrics,
            "eval/best_replay_path": str(paths["best"]) if "best" in paths else None,
            "eval/worst_replay_path": str(paths["worst"]) if "worst" in paths else None}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        if self.run is not None:
            import wandb
            payload = {key: value for key, value in row.items() if not key.endswith("_path")}
            for name, path in paths.items(): payload[f"eval/{name}_replay"] = wandb.Html(path.read_text(encoding="utf-8"), inject=False)
            self.run.log(payload)
            if paths:
                boundary = metrics["evaluation_boundary_seconds"]
                artifact = wandb.Artifact(f"{self.run.id}-evaluation-{boundary:.3f}-replays", type="evaluation-replays")
                for name, path in paths.items(): artifact.add_file(str(path), name=f"{name}.html")
                self.run.log_artifact(artifact)

    def finish(self, status, **summary):
        elapsed = self.budget.elapsed
        result = {"status": status, "elapsed_seconds": elapsed, "overrun_seconds": max(0, elapsed - self.budget.seconds), **summary}
        (self.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        self.stream.close()
        if self.run is not None:
            self.run.summary.update(result)
            self.run.finish(exit_code=0 if status == "complete" else 1)
