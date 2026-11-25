"""Evaluation workers can run on GPUs other than their trainer's (SoftWalker needs one Warp process per trainer GPU)."""

import queue
import threading

import pytest

from common import async_evaluation
from launch_environment_suite import eval_gpu


def test_eval_gpu_round_robin_by_training_gpu():
    gpus = [0, 1, 2, 3, 4, 5]
    assert [eval_gpu(g, gpus, [6, 7]) for g in gpus] == [6, 7, 6, 7, 6, 7]
    assert eval_gpu(3, gpus, None) is None


def test_worker_moves_to_the_eval_gpu_before_building_its_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("CODESIGN_EVAL_GPU", "7")
    seen = {}

    class Backend:
        def __init__(self, config):
            import os
            seen["visible"] = os.environ["CUDA_VISIBLE_DEVICES"]

        def close(self):
            pass

    root = tmp_path / "snapshots"
    root.mkdir()
    events, stop = queue.Queue(), threading.Event()
    async_evaluation._worker({}, str(root), 0.0, 0.0, events, stop, None, Backend)
    assert seen["visible"] == "7"


def test_suite_rejects_eval_gpus_that_overlap_training(monkeypatch, tmp_path):
    import sys
    import launch_environment_suite
    monkeypatch.setattr(sys, "argv", ["launch_environment_suite.py", "--families", "soft_walker", "--seeds", "0",
                                      "--gpus", "0", "1", "--eval-gpus", "1", "--hours", "1", "--num-envs", "16",
                                      "--output-dir", str(tmp_path / "runs"), "--dry-run"])
    with pytest.raises(SystemExit):
        launch_environment_suite.main()
