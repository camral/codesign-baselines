"""Multi-GPU all-corpora clustering launcher contracts."""

import json
import threading

import numpy as np
import pytest

from loki import cluster_all


def fake_jobs(tmp_path):
    return [
        {"corpus_id": name, "corpus": str(tmp_path / name),
         "vae": str(tmp_path / "out" / name / "vae"),
         "output": str(tmp_path / "out" / name / "clusters"),
         "tokens": tokens, "latent_width": tokens * 32,
         "accepted": 500_000, "batch_size": batch,
         "estimated_work": tokens ** 2, "status": "queued"}
        for name, tokens, batch in (
            ("Warehouse-Congested", 438, 1024),
            ("NeroGrasp-All", 146, 4096),
            ("Microgrid-Random", 3, 4096),
        )
    ]


def test_dry_run_assigns_one_visible_gpu_per_process(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cluster_all, "_load_jobs",
                        lambda args, parser: fake_jobs(tmp_path))
    assert cluster_all.main([
        "--gpus", "2", "5", "--output-dir", str(tmp_path),
        "--threads-per-job", "7", "--dry-run",
    ]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("GPU 2, CPU threads 7:")
    assert lines[1].startswith("GPU 5, CPU threads 7:")
    assert lines[2].startswith("GPU 2, CPU threads 7:")
    assert all("--device cuda:0" in line for line in lines)
    assert all("--n-clusters 6" in line for line in lines)
    assert "--batch-size 1024" in lines[0]


def test_workers_partition_gpus_and_cpu_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_all, "_load_jobs",
                        lambda args, parser: fake_jobs(tmp_path))
    lock = threading.Lock()
    active, calls = set(), []

    class Process:
        _pid = 100

        def __init__(self, command, **kwargs):
            self.command, self.environment = command, kwargs["env"]
            self.returncode = None
            Process._pid += 1
            self.pid = Process._pid
            gpu = self.environment["CUDA_VISIBLE_DEVICES"]
            with lock:
                assert gpu not in active
                active.add(gpu)
                calls.append((gpu, command, self.environment))

        def wait(self, timeout=None):
            with lock:
                active.remove(self.environment["CUDA_VISIBLE_DEVICES"])
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(cluster_all.subprocess, "Popen", Process)
    assert cluster_all.main([
        "--gpus", "1", "4", "--output-dir", str(tmp_path / "artifacts"),
        "--threads-per-job", "9",
    ]) == 0
    assert len(calls) == 3
    assert {gpu for gpu, _, _ in calls} <= {"1", "4"}
    assert all(env["OMP_NUM_THREADS"] == "9" for _, _, env in calls)
    assert all(env["MKL_NUM_THREADS"] == "9" for _, _, env in calls)


def test_compatible_completed_output_is_recognized(tmp_path):
    output = tmp_path / "clusters"
    (output / "clusters").mkdir(parents=True)
    expected = {
        "corpus_identity": {"manifest_sha256": "corpus"},
        "vae_checkpoint_sha256": "vae", "design_count": 12,
        "vae_fixed_epoch_recipe": True, "vae_epochs_completed": 200,
        "n_clusters": 6, "posterior_seed": 3, "kmeans_seed": 4,
        "min_cluster_size": 1, "save_latents": False,
    }
    manifest = {
        "status": "complete", "corpus_identity": expected["corpus_identity"],
        "vae_checkpoint_sha256": "vae", "design_count": 12,
        "vae_fixed_epoch_recipe": True, "vae_epochs_completed": 200,
        "clustering_config": {
            key: expected[key] for key in (
                "n_clusters", "posterior_seed", "kmeans_seed",
                "min_cluster_size", "save_latents")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    for name in ("labels.npy", "centers.npy", "design_ids.npy"):
        np.save(output / name, np.zeros(1), allow_pickle=False)
    for index in range(6):
        np.save(output / "clusters" / f"{index}.npy", np.zeros(1),
                allow_pickle=False)
    assert cluster_all._completed_output_matches(output, expected) == (True, None)
    manifest["vae_checkpoint_sha256"] = "stale"
    (output / "manifest.json").write_text(json.dumps(manifest))
    matches, reason = cluster_all._completed_output_matches(output, expected)
    assert not matches
    assert "vae_checkpoint_sha256" in reason


def test_gpu_ids_and_threads_must_be_positive(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_all, "_load_jobs",
                        lambda args, parser: fake_jobs(tmp_path))
    with pytest.raises(SystemExit):
        cluster_all.main(["--gpus", "1", "1", "--dry-run"])
    with pytest.raises(SystemExit):
        cluster_all.main([
            "--gpus", "1", "--threads-per-job", "0", "--dry-run"])


def test_default_threads_are_capped_for_openblas(monkeypatch):
    monkeypatch.setattr(cluster_all.os, "cpu_count", lambda: 256)
    args, _ = cluster_all.parse_args(["--gpus", "1", "--dry-run"])
    assert args.threads_per_job == 64
