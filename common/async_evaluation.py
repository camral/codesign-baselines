"""Evaluate immutable snapshots on the run clock, independently of training.

Only the training thread creates snapshots, at safe points between updates.
The worker's timer pins the latest published file at each 5% boundary. A slow
rollout can therefore repeat a snapshot, but cannot stop the evaluation clock.
"""

import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import threading
import time
import traceback


def _schedule(root, started_at, seconds, jobs, events, stop, modeled_elapsed):
    # The budget includes evaluation. At 100% there is no time for another
    # episode; the trainer saves its final checkpoint instead.
    try:
        for index in range(1, 20):
            boundary = seconds * index / 20
            if modeled_elapsed is None:
                if stop.wait(max(0.0, started_at + boundary - time.monotonic())):
                    break
                now = time.monotonic() - started_at
            else:
                while modeled_elapsed.value < boundary and not stop.wait(0.1):
                    pass
                if stop.is_set():
                    break
                now = modeled_elapsed.value
            metrics = {"elapsed_seconds": now, "evaluation_boundary_seconds": boundary,
                "eval/scheduled_elapsed_seconds": now}
            pinned = root / f"boundary_{index:02d}.pt"
            try:
                # Atomic publication plus hard links makes the selected version
                # immutable even if training replaces latest.pt during evaluation.
                os.link(root / "latest.pt", pinned)
            except FileNotFoundError:
                events.put(("log", {**metrics, "eval/status": "no_snapshot"}))
                continue
            events.put(("log", {**metrics, "eval/status": "scheduled"}))
            jobs.put((pinned, boundary, now))
    except Exception:
        events.put(("error", traceback.format_exc()))
        stop.set()
    finally:
        jobs.put(None)


def _worker(config, root, started_at, seconds, events, stop, modeled_elapsed, backend_factory):
    try:
        os.setsid()
    except OSError:
        pass
    root = Path(root)
    # Optionally evaluate on another physical GPU (set before any CUDA use in this spawned process). A concurrent
    # SoftWalker trainer must be the only Warp process on its GPU, so its evaluation cannot share that GPU.
    if os.environ.get("CODESIGN_EVAL_GPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CODESIGN_EVAL_GPU"]
    # EGL device numbers are host-global, unlike the local CUDA index seen by
    # a process with CUDA_VISIBLE_DEVICES set.  Codesign Gym may therefore
    # inherit a physical training-GPU index here (for example 14), which MuJoCo
    # rejects and silently costs us every evaluation video.  Rendering belongs
    # on the reserved GPU 0; simulation and policy evaluation remain on the
    # worker's CUDA_VISIBLE_DEVICES GPU.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get(
        "CODESIGN_RENDER_DEVICE_ID", "0")
    # Training and evaluation initialize Warp concurrently on the same CUDA
    # device. They must not compile into the same cache directory: a reader can
    # otherwise observe a partially written module and fail with a missing CUDA
    # symbol or kernel metadata key.
    warp_cache = Path(os.environ.get("WARP_CACHE_PATH", root.parent / ".warp_cache"))
    os.environ["WARP_CACHE_PATH"] = str(warp_cache / "evaluation")
    jobs = queue.Queue()  # At most 19 immutable snapshot files per run.
    timer = threading.Thread(target=_schedule,
        args=(root, started_at, seconds, jobs, events, stop, modeled_elapsed), daemon=True)
    timer.start()
    backend = None
    best = None
    try:
        if backend_factory is None:
            from .evaluation_worker import SnapshotEvaluator
            backend_factory = SnapshotEvaluator
        backend = backend_factory(config)
        events.put(("ready", None))
        while True:
            job = jobs.get()
            if job is None:
                break
            path, boundary, scheduled = job
            try:
                started = time.monotonic() - started_at if modeled_elapsed is None else modeled_elapsed.value
                metrics = {"evaluation_boundary_seconds": boundary,
                    "eval/scheduled_elapsed_seconds": scheduled,
                    "eval/started_elapsed_seconds": started,
                    "eval/start_delay_seconds": max(0.0, started - boundary)}
                if stop.is_set() or started >= seconds:
                    events.put(("log", {**metrics, "elapsed_seconds": started,
                        "eval/status": "budget_exhausted" if started >= seconds else "cancelled"}))
                    continue
                events.put(("log", {**metrics, "elapsed_seconds": started, "eval/status": "started"}))
                published = False

                def publish_score(result, paths=None):
                    nonlocal best, published
                    elapsed = time.monotonic() - started_at if modeled_elapsed is None else modeled_elapsed.value
                    row = {**metrics, **result, "elapsed_seconds": elapsed}
                    row["eval/duration_seconds"] = elapsed - started
                    if "eval/snapshot_elapsed_seconds" in row:
                        row["eval/snapshot_age_seconds"] = max(0.0, started - row["eval/snapshot_elapsed_seconds"])
                    score = row.get("eval/reward")
                    if score is not None and (best is None or score > best):
                        backend.save_best(path, score)
                        best = score
                    events.put(("evaluation", (row, paths or {}, best)))
                    published = True

                kwargs = {"deadline": started_at + seconds if modeled_elapsed is None else None, "stop": stop}
                if getattr(backend, "supports_metrics_callback", False):
                    kwargs["on_metrics"] = publish_score
                result, paths = backend.evaluate(path, **kwargs)
                if not published:
                    publish_score(result, paths)
                else:
                    # A separate video event must not duplicate the evaluation score.
                    video = {key: value for key, value in result.items()
                             if key.startswith(("eval/video_", "eval/replay_")) or key in
                             ("eval/render_seconds", "eval/snapshot_version", "eval/snapshot_elapsed_seconds")}
                    video.update(metrics)
                    video["elapsed_seconds"] = time.monotonic() - started_at if modeled_elapsed is None else modeled_elapsed.value
                    events.put(("evaluation", (video, paths, best)))
            finally:
                path.unlink(missing_ok=True)
    except Exception:
        events.put(("error", traceback.format_exc()))
    finally:
        stop.set()
        timer.join()
        if backend is not None:
            try:
                backend.close()
            except Exception:
                events.put(("error", traceback.format_exc()))
        for path in root.glob("boundary_*.pt"):
            path.unlink(missing_ok=True)
        events.put(("done", None))


class AsyncEvaluator:
    """Publish from the training thread; evaluate and log without polling it."""

    def __init__(self, log, args, method, *, backend_factory=None):
        self.log, self.budget = log, log.budget
        self.root = log.output / "evaluation_snapshots"
        self.root.mkdir()
        self.interval = min(30.0, self.budget.evaluation_interval / 4)
        self.last_published = float("-inf")
        self.version = 0
        self.best_eval_reward = None
        self.error = None
        self.closed = False
        self.terminated = False
        self.ready = threading.Event()
        context = mp.get_context("spawn")
        self.stop = context.Event()
        self.events = context.Queue()
        self.modeled_elapsed = context.Value("d", self.budget.elapsed) if method == "ppo_llm" else None
        config = {**vars(args), "method": method, "output_dir": str(log.output)}
        self.process = context.Process(target=_worker,
            args=(config, str(self.root), self.budget.started_at, self.budget.seconds,
                self.events, self.stop, self.modeled_elapsed, backend_factory), name="codesign-evaluation")
        self.process.start()
        self.listener = threading.Thread(target=self._listen, name="evaluation-logging", daemon=True)
        self.listener.start()

    def _listen(self):
        try:
            while True:
                try:
                    kind, value = self.events.get(timeout=0.2)
                except queue.Empty:
                    if not self.process.is_alive():
                        if self.process.exitcode != 0 and not self.terminated:
                            self.error = f"evaluation worker exited with code {self.process.exitcode}"
                        break
                    continue
                if kind == "done":
                    break
                if kind == "ready":
                    self.ready.set()
                elif kind == "error":
                    self.error = value
                    self.log.log(**{"eval/status": "failed", "eval/error": value})
                elif kind == "log":
                    self.log.log(**value)
                elif kind == "evaluation":
                    metrics, paths, self.best_eval_reward = value
                    self.log.log_evaluation_paths(paths, **metrics)
        except Exception:
            self.error = traceback.format_exc()
            self.stop.set()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"independent evaluation failed:\n{self.error}")

    def publish(self, snapshot, *, force=False, kind="current", **metrics):
        """Build a snapshot lazily, only on the training thread at safe points."""
        self.check()
        now = self.budget.elapsed
        if self.closed:
            return
        if not force and now - self.last_published < self.interval:
            if self.modeled_elapsed is not None:
                self.modeled_elapsed.value = now
            return
        import torch
        payload = snapshot()
        self.version += 1
        payload["_evaluation"] = {"version": self.version, "elapsed_seconds": now,
            "kind": kind, "metrics": {key: value for key, value in metrics.items() if value is not None}}
        temporary = self.root / "pending.pt"
        torch.save(payload, temporary)
        os.replace(temporary, self.root / "latest.pt")
        self.last_published = now
        if self.modeled_elapsed is not None:
            self.modeled_elapsed.value = now

    def close(self, *, raise_errors=True):
        if self.modeled_elapsed is not None:
            self.modeled_elapsed.value = self.budget.elapsed
        if not self.closed:
            self.closed = True
            self.stop.set()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.terminated = True
                pid = self.process.pid
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    self.process.terminate()
                self.process.join(timeout=3)
                if self.process.is_alive():
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        self.process.kill()
                    self.process.join()
                self.log.log(**{"eval/status": "shutdown_timeout"})
            self.listener.join(timeout=5)
            if self.listener.is_alive():
                self.error = self.error or "evaluation logging did not stop"
            else:
                self.events.close()
                self.events.join_thread()
            for path in self.root.glob("*.pt"):
                path.unlink(missing_ok=True)
        if raise_errors:
            self.check()
