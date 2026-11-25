"""Evaluation proceeds and reaches the logger while training publishes nothing."""

import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.async_evaluation import AsyncEvaluator
from common.run_log import RunLog
from common.wall_clock import WallClockBudget


class FakeEvaluator:
    def __init__(self, config):
        self.output = config["output_dir"]
        self.delay = config.get("delay", 0)

    def evaluate(self, path, *, deadline, stop):
        # Read after the delay to exercise pinned-file ownership during publish.
        stop.wait(self.delay)
        payload = torch.load(path, weights_only=False)
        meta = payload["_evaluation"]
        return {"eval/status": "complete", "eval/reward": float(payload["value"]),
            "eval/snapshot_version": meta["version"],
            "eval/snapshot_elapsed_seconds": meta["elapsed_seconds"]}, {}

    def save_best(self, path, score):
        from pathlib import Path
        payload = torch.load(path, weights_only=False)
        Path(self.output, "best.json").write_text(json.dumps({"value": float(payload["value"]), "score": score}))

    def close(self):
        pass


class FailingEvaluator(FakeEvaluator):
    def evaluate(self, *args, **kwargs):
        raise ValueError("deliberate evaluation failure")


class CacheRecordingEvaluator(FakeEvaluator):
    def __init__(self, config):
        super().__init__(config)
        from pathlib import Path
        Path(self.output, "evaluation_warp_cache.txt").write_text(
            os.environ["WARP_CACHE_PATH"])
        Path(self.output, "evaluation_render_device.txt").write_text(
            os.environ["MUJOCO_EGL_DEVICE_ID"])


def rows(log):
    return [json.loads(line) for line in (log.output / "metrics.jsonl").read_text().splitlines()]


def wait_for(predicate, evaluator, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        evaluator.check()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("evaluation did not produce the expected event")


def test_evaluates_and_logs_without_training_progress(tmp_path):
    log = RunLog(tmp_path / "run", WallClockBudget.start(30 / 3600), {})
    worker = AsyncEvaluator(log, SimpleNamespace(), "fake", backend_factory=FakeEvaluator)
    try:
        worker.publish(lambda: {"value": torch.tensor(7.)}, force=True)
        # No training callbacks or publication happen while these boundaries pass.
        wait_for(lambda: sum("eval/reward" in row for row in rows(log)) >= 2, worker)
        completed = [row for row in rows(log) if "eval/reward" in row]
        assert all(row["eval/reward"] == 7 and row["eval/snapshot_version"] == 1 for row in completed)
        assert completed[1]["evaluation_boundary_seconds"] - completed[0]["evaluation_boundary_seconds"] == pytest.approx(1.5)
        assert completed[1]["eval/snapshot_age_seconds"] > completed[0]["eval/snapshot_age_seconds"]
        assert worker.best_eval_reward == 7
        assert json.loads((log.output / "best.json").read_text()) == {"value": 7, "score": 7}
    finally:
        worker.close()
        log.finish("complete")
    assert not worker.process.is_alive() and not worker.listener.is_alive()
    assert not list(worker.root.glob("*.pt"))


def test_pinned_snapshot_survives_new_publication(tmp_path):
    log = RunLog(tmp_path / "run", WallClockBudget.start(30 / 3600), {})
    worker = AsyncEvaluator(log, SimpleNamespace(delay=0.5), "fake", backend_factory=FakeEvaluator)
    try:
        worker.publish(lambda: {"value": 3}, force=True)
        wait_for(lambda: any(row.get("eval/status") == "started" for row in rows(log)), worker)
        worker.publish(lambda: {"value": 8}, force=True)
        wait_for(lambda: any(row.get("eval/reward") == 8 for row in rows(log)), worker)
        completed = [row for row in rows(log) if "eval/reward" in row]
        assert completed[0]["eval/reward"] == 3
        assert completed[0]["eval/snapshot_version"] == 1
        assert completed[-1]["eval/snapshot_version"] == 2
    finally:
        worker.close()
        log.finish("complete")


def test_worker_failure_is_visible_and_stops_cleanly(tmp_path):
    log = RunLog(tmp_path / "run", WallClockBudget.start(30 / 3600), {})
    worker = AsyncEvaluator(log, SimpleNamespace(), "fake", backend_factory=FailingEvaluator)
    try:
        worker.publish(lambda: {"value": 1}, force=True)
        with pytest.raises(RuntimeError, match="deliberate evaluation failure"):
            wait_for(lambda: False, worker)
    finally:
        worker.close(raise_errors=False)
        log.finish("failed")
    assert any(row.get("eval/status") == "failed" for row in rows(log))
    assert not worker.process.is_alive()


def test_evaluation_worker_uses_an_isolated_warp_cache(tmp_path, monkeypatch):
    training_cache = tmp_path / "warp" / "gpu_3"
    monkeypatch.setenv("WARP_CACHE_PATH", str(training_cache))
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "14")
    log = RunLog(tmp_path / "run", WallClockBudget.start(30 / 3600), {})
    worker = AsyncEvaluator(log, SimpleNamespace(), "fake",
                            backend_factory=CacheRecordingEvaluator)
    try:
        assert worker.ready.wait(10)
        assert (log.output / "evaluation_warp_cache.txt").read_text() == str(
            training_cache / "evaluation")
        assert (log.output / "evaluation_render_device.txt").read_text() == "0"
        assert os.environ["WARP_CACHE_PATH"] == str(training_cache)
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "14"
    finally:
        worker.close()
        log.finish("complete")


def test_training_and_evaluation_logging_are_serialized(tmp_path):
    log = RunLog(tmp_path / "run", WallClockBudget.start(1), {})
    frame = tmp_path / "video.mp4"
    def write_training():
        for value in range(50):
            log.log(train_value=value)
    thread = threading.Thread(target=write_training)
    thread.start()
    for value in range(50):
        log.log_evaluation_paths({"best": frame, "worst": frame},
            evaluation_boundary_seconds=value, **{"eval/reward": value})
    thread.join()
    log.finish("complete")
    assert len(rows(log)) == 100


def test_adapter_renders_only_the_requested_mobile_world(monkeypatch):
    from common.codesign_adapter import make_baseline_vec
    env, _ = make_baseline_vec("Network-Urban", 8, device="cpu", render_mode="rgb_array")
    calls = []
    try:
        for index, single in enumerate(env.env.envs):
            monkeypatch.setattr(single, "render", lambda i=index: calls.append(i) or i)
        assert env.render(env_idx=6) == 6
        assert calls == [6]
    finally:
        env.close()


class ScoringBeforeVideoEvaluator(FakeEvaluator):
    supports_metrics_callback = True

    def evaluate(self, path, *, deadline, stop, on_metrics=None):
        from pathlib import Path
        payload = torch.load(path, weights_only=False)
        metrics = {"eval/status": "complete", "eval/reward": float(payload["value"]),
                   "eval/snapshot_version": payload["_evaluation"]["version"]}
        on_metrics(metrics)
        Path(self.output, "render_started").touch()
        stop.wait(10)  # Deliberately blocked video work, independent of score publication.
        return {**metrics, "eval/video_complete": False}, {}


def test_score_and_checkpoint_are_published_before_video_finishes(tmp_path):
    log = RunLog(tmp_path / "run", WallClockBudget.start(30 / 3600), {})
    worker = AsyncEvaluator(log, SimpleNamespace(), "fake", backend_factory=ScoringBeforeVideoEvaluator)
    try:
        worker.publish(lambda: {"value": 7}, force=True)
        wait_for(lambda: (log.output / "render_started").exists()
                 and any("eval/reward" in row for row in rows(log)), worker)
        assert worker.best_eval_reward == 7
        assert json.loads((log.output / "best.json").read_text())["score"] == 7
    finally:
        worker.close()
        log.finish("complete")
    assert sum("eval/reward" in row for row in rows(log)) == 1
    assert any("eval/video_complete" in row for row in rows(log))


def test_adapter_renders_only_requested_warehouse_world(monkeypatch):
    from common.codesign_adapter import make_baseline_vec

    env, _ = make_baseline_vec("Warehouse-Small", 8, device="cpu", render_mode="rgb_array")
    calls = []
    try:
        env.reset(seed=0)
        monkeypatch.setattr(env.env, "_render_rgb",
                            lambda i: calls.append(i) or np.zeros((4, 4, 3), np.uint8))
        assert env.render(env_idx=6).shape == (4, 4, 3)
        assert calls == [6]
    finally:
        env.close()
