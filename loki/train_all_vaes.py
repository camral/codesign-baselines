"""Distribute canonical LOKI VAE training across one process per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

from .collect_all import CORPUS_GROUPS, REPO, corpus_dir
from .collectors.storage import write_json
from .train_vae import TrainingConfig
from .vae import CorpusTokenSchema


DEFAULT_CONGESTED_BATCH_SIZE = 1024


def _save_summary(path, args, jobs):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, {"config": vars(args), "jobs": jobs})
    temporary.replace(path)


def _training_command(job, args):
    command = [
        sys.executable, "-u", "-m", "loki.train_vae",
        "--corpus", job["corpus"],
        "--output", job["output"],
        "--device", "cuda:0",
        "--batch-size", str(job["batch_size"]),
        "--max-epochs", str(args.max_epochs),
    ]
    if args.resume or job["resume"]:
        command.append("--resume")
    return command


def _load_jobs(args, parser):
    corpus_root = args.corpus_dir.resolve()
    output_root = args.output_dir.resolve()
    requested = args.corpora or list(CORPUS_GROUPS)
    jobs, unavailable = [], []
    for corpus_id in requested:
        corpus = corpus_root / corpus_dir(corpus_id)
        manifest_path = corpus / "manifest.json"
        if not manifest_path.is_file():
            unavailable.append(f"{corpus_id}: missing manifest")
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as error:
            parser.error(f"cannot read {manifest_path}: {error}")
        if manifest.get("status") != "complete":
            unavailable.append(
                f"{corpus_id}: status={manifest.get('status', 'missing')}")
            continue
        schema = CorpusTokenSchema.from_corpus(corpus)
        output = output_root / corpus_dir(corpus_id) / "vae"
        report_path = output / "convergence.json"
        report = (json.loads(report_path.read_text())
                  if report_path.is_file() else None)
        status = ("skipped" if report and
                  report.get("complete", report.get("converged", False))
                  else "queued")
        batch_size = (args.warehouse_congested_batch_size
                      if corpus_id == "Warehouse-Congested"
                      else args.batch_size)
        accepted = int(manifest.get("state", {}).get(
            "stats", {}).get("accepted", 0))
        jobs.append({
            "corpus_id": corpus_id,
            "corpus": str(corpus.resolve()),
            "output": str(output),
            "tokens": schema.token_count,
            "accepted": accepted,
            "batch_size": batch_size,
            "estimated_work": accepted * schema.token_count ** 2,
            "resume": (output / "last.pt").is_file(),
            "status": status,
        })
    if unavailable and not args.skip_unavailable:
        parser.error("unavailable corpora:\n  " + "\n  ".join(unavailable))
    for message in unavailable:
        print(f"Skipping {message}", flush=True)
    # Longest-processing-time-first scheduling reduces the tail after short
    # corpora finish. Workers dynamically claim the next queued job.
    jobs.sort(key=lambda job: (-job["estimated_work"], job["corpus_id"]))
    return jobs


def _stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main(argv=None):
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True,
                        help="Physical GPU IDs; one VAE process runs on each GPU")
    parser.add_argument("--corpus-dir", type=Path, default=REPO / "data/loki")
    parser.add_argument("--output-dir", type=Path, default=REPO / "artifacts/loki")
    parser.add_argument("--summary-name", default="training_summary.json",
                        help="Summary filename under output-dir; use a unique name for concurrent launchers")
    parser.add_argument("--corpora", nargs="+", choices=tuple(CORPUS_GROUPS),
                        help="Optional canonical corpus subset")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warehouse-congested-batch-size", type=int,
                        default=DEFAULT_CONGESTED_BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--resume", action="store_true",
                        help="Require every queued job to resume an existing run")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="Train complete corpora and skip missing/incomplete ones")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.gpus) != len(set(args.gpus)) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU IDs must be unique nonnegative integers")
    if min(args.batch_size, args.warehouse_congested_batch_size,
           args.max_epochs) < 1:
        parser.error("batch sizes and max-epochs must be positive")
    if (not args.summary_name or Path(args.summary_name).name != args.summary_name
            or args.summary_name in (".", "..")):
        parser.error("summary-name must be a filename under output-dir")

    jobs = _load_jobs(args, parser)
    for job in jobs:
        job["command"] = _training_command(job, args)
    if args.resume:
        missing = [job["corpus_id"] for job in jobs
                   if job["status"] == "queued" and not job["resume"]]
        if missing:
            parser.error("--resume requested but no checkpoint exists for: "
                         + ", ".join(missing))
    if args.dry_run:
        for index, job in enumerate(job for job in jobs if job["status"] == "queued"):
            gpu = args.gpus[index % len(args.gpus)]
            print(f"GPU {gpu}: {shlex.join(job['command'])}")
        for job in jobs:
            if job["status"] == "skipped":
                print(f"Skipping completed {job['corpus_id']}")
        return 0

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    summary_path = output_root / args.summary_name
    lock, stop = threading.Lock(), threading.Event()
    active = {}
    _save_summary(summary_path, args, jobs)

    def worker(gpu):
        while not stop.is_set():
            with lock:
                job = next((candidate for candidate in jobs
                            if candidate["status"] == "queued"), None)
                if job is None:
                    return
                job.update(status="running", gpu=gpu,
                           started_at=time.time())
                _save_summary(summary_path, args, jobs)
                print(f"[{job['corpus_id']}] GPU {gpu}: "
                      f"{shlex.join(job['command'])}", flush=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            started, returncode = time.monotonic(), None
            try:
                log_path = logs / f"{corpus_dir(job['corpus_id'])}.log"
                with log_path.open("a") as stream:
                    stream.write(f"\nLaunch {time.strftime('%Y-%m-%d %H:%M:%S')}: "
                                 f"{shlex.join(job['command'])}\n")
                    stream.flush()
                    process = subprocess.Popen(
                        job["command"], cwd=REPO, env=environment,
                        stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    with lock:
                        active[gpu] = process
                    returncode = process.wait()
            except Exception as error:
                job["error"] = repr(error)
            finally:
                with lock:
                    active.pop(gpu, None)
                    job.update(
                        status="complete" if returncode == 0 else "failed",
                        returncode=returncode,
                        elapsed_seconds=time.monotonic() - started,
                        finished_at=time.time())
                    _save_summary(summary_path, args, jobs)
            print(f"[{job['corpus_id']}] GPU {gpu}: {job['status']} "
                  f"(return code {returncode})", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in args.gpus]
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        stop.set()
        with lock:
            processes = list(active.values())
        _stop_processes(processes)
        return 130
    return 1 if any(job["status"] == "failed" for job in jobs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
