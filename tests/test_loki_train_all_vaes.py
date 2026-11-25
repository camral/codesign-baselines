"""Multi-GPU LOKI VAE launcher contracts."""

import threading

import pytest

from loki import train_all_vaes


def fake_jobs(tmp_path):
    return [
        {"corpus_id": name, "corpus": str(tmp_path / name),
         "output": str(tmp_path / "out" / name / "vae"), "tokens": tokens,
         "accepted": 500_000, "batch_size": batch, "estimated_work": tokens ** 2,
         "resume": False, "status": "queued"}
        for name, tokens, batch in (
            ("Warehouse-Congested", 438, 1024),
            ("NeroGrasp-All", 146, 4096),
            ("Microgrid-Random", 3, 4096),
        )
    ]


def test_dry_run_assigns_one_visible_gpu_per_process(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        train_all_vaes, "_load_jobs",
        lambda args, parser: fake_jobs(tmp_path))
    assert train_all_vaes.main([
        "--gpus", "2", "5", "--output-dir", str(tmp_path), "--dry-run",
    ]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("GPU 2:")
    assert lines[1].startswith("GPU 5:")
    assert lines[2].startswith("GPU 2:")
    assert all("--device cuda:0" in line for line in lines)
    assert "--batch-size 1024" in lines[0]


def test_workers_never_share_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_all_vaes, "_load_jobs",
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
                calls.append((gpu, command))

        def wait(self, timeout=None):
            gpu = self.environment["CUDA_VISIBLE_DEVICES"]
            with lock:
                active.remove(gpu)
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(train_all_vaes.subprocess, "Popen", Process)
    assert train_all_vaes.main([
        "--gpus", "1", "4", "--output-dir", str(tmp_path / "artifacts"),
    ]) == 0
    assert len(calls) == 3
    assert {gpu for gpu, _ in calls} <= {"1", "4"}
    assert all(command[command.index("--device") + 1] == "cuda:0"
               for _, command in calls)


def test_gpu_ids_must_be_unique(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_all_vaes, "_load_jobs",
        lambda args, parser: fake_jobs(tmp_path))
    with pytest.raises(SystemExit):
        train_all_vaes.main(["--gpus", "1", "1", "--dry-run"])


def test_concurrent_launcher_can_use_separate_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_all_vaes, "_load_jobs",
        lambda args, parser: fake_jobs(tmp_path)[:1])

    class Process:
        pid = 100
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(train_all_vaes.subprocess, "Popen", Process)
    output = tmp_path / "artifacts"
    assert train_all_vaes.main([
        "--gpus", "2", "--output-dir", str(output),
        "--summary-name", "locomotion_training_summary.json",
    ]) == 0
    assert (output / "locomotion_training_summary.json").is_file()
    assert not (output / "training_summary.json").exists()


@pytest.mark.parametrize("name", ("", "../summary.json", "nested/summary.json"))
def test_summary_name_must_be_local_filename(name):
    with pytest.raises(SystemExit):
        train_all_vaes.main([
            "--gpus", "2", "--summary-name", name, "--dry-run",
        ])
