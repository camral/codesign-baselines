"""Distribute all canonical LOKI clustering jobs across a GPU list."""

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

from .cluster import ClusteringConfig
from .collect_all import CORPUS_GROUPS, REPO, corpus_dir
from .collectors.storage import file_digest, write_json
from .vae import CorpusTokenSchema


DEFAULT_CONGESTED_BATCH_SIZE = 1024


def _save_summary(path, args, jobs):
    temporary = path.with_suffix(".tmp")
    config = vars(args).copy()
    config["threads_per_job"] = args.threads_per_job
    write_json(temporary, {"config": config, "jobs": jobs})
    temporary.replace(path)


def _cluster_command(job, args):
    command = [
        sys.executable, "-u", "-m", "loki.cluster",
        "--corpus", job["corpus"],
        "--vae", job["vae"],
        "--output", job["output"],
        "--device", "cuda:0",
        "--n-clusters", str(args.n_clusters),
        "--batch-size", str(job["batch_size"]),
        "--posterior-seed", str(args.posterior_seed),
        "--kmeans-seed", str(args.kmeans_seed),
        "--min-cluster-size", str(args.min_cluster_size),
    ]
    if args.save_latents:
        command.append("--save-latents")
    return command


def _read_json(path, description, parser):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        parser.error(f"cannot read {description} {path}: {error}")


def _completed_output_matches(path, expected):
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False, "missing cluster manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        return False, f"unreadable cluster manifest: {error}"
    checks = {
        "status": "complete",
        "corpus_identity": expected["corpus_identity"],
        "vae_checkpoint_sha256": expected["vae_checkpoint_sha256"],
        "vae_fixed_epoch_recipe": expected["vae_fixed_epoch_recipe"],
        "vae_epochs_completed": expected["vae_epochs_completed"],
        "design_count": expected["design_count"],
    }
    for key, value in checks.items():
        if manifest.get(key) != value:
            return False, f"cluster manifest differs in {key}"
    clustering = manifest.get("clustering_config", {})
    for key in ("n_clusters", "posterior_seed", "kmeans_seed",
                "min_cluster_size", "save_latents"):
        if clustering.get(key) != expected[key]:
            return False, f"cluster manifest differs in clustering_config.{key}"
    required = [path / "labels.npy", path / "centers.npy", path / "design_ids.npy"]
    required.extend(path / "clusters" / f"{index}.npy"
                    for index in range(expected["n_clusters"]))
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        return False, f"missing cluster artifact {missing[0]}"
    if expected["save_latents"] and not (path / "latents.npy").is_file():
        return False, "missing requested latents.npy"
    return True, None


def _load_jobs(args, parser):
    corpus_root = args.corpus_dir.resolve()
    artifact_root = args.output_dir.resolve()
    requested = args.corpora or list(CORPUS_GROUPS)
    jobs, unavailable = [], []
    for corpus_id in requested:
        corpus = corpus_root / corpus_dir(corpus_id)
        corpus_manifest_path = corpus / "manifest.json"
        if not corpus_manifest_path.is_file():
            unavailable.append(f"{corpus_id}: missing corpus manifest")
            continue
        corpus_manifest = _read_json(corpus_manifest_path, "corpus manifest", parser)
        if corpus_manifest.get("status") != "complete":
            unavailable.append(
                f"{corpus_id}: corpus status={corpus_manifest.get('status', 'missing')}")
            continue
        accepted = int(corpus_manifest.get("state", {}).get(
            "stats", {}).get("accepted", 0))
        vae = artifact_root / corpus_dir(corpus_id) / "vae"
        report_path, checkpoint_path = vae / "convergence.json", vae / "best.pt"
        if not report_path.is_file() or not checkpoint_path.is_file():
            unavailable.append(f"{corpus_id}: missing completed VAE artifacts")
            continue
        report = _read_json(report_path, "VAE report", parser)
        if (not report.get("complete", False)
                or not report.get("fixed_epoch_recipe", False)
                or report.get("epochs_completed") != 200):
            unavailable.append(
                f"{corpus_id}: VAE did not complete the fixed 200-epoch recipe")
            continue
        checkpoint_hash = file_digest(checkpoint_path)
        if report.get("best_checkpoint_sha256") != checkpoint_hash:
            parser.error(f"{corpus_id}: best.pt does not match convergence.json")
        schema = CorpusTokenSchema.from_corpus(corpus)
        vae_config_path = vae / "config.json"
        if not vae_config_path.is_file():
            unavailable.append(f"{corpus_id}: missing VAE config.json")
            continue
        vae_config = _read_json(vae_config_path, "VAE config", parser)
        hidden_dim = int(vae_config["model"]["hidden_dim"])
        corpus_identity = {
            "manifest_sha256": file_digest(corpus_manifest_path),
            "constants_sha256": file_digest(corpus / "constants.npz"),
            "accepted": accepted,
            "status": corpus_manifest["status"],
        }
        expected = {
            "corpus_identity": corpus_identity,
            "vae_checkpoint_sha256": checkpoint_hash,
            "vae_fixed_epoch_recipe": True,
            "vae_epochs_completed": 200,
            "design_count": accepted,
            "n_clusters": args.n_clusters,
            "posterior_seed": args.posterior_seed,
            "kmeans_seed": args.kmeans_seed,
            "min_cluster_size": args.min_cluster_size,
            "save_latents": args.save_latents,
        }
        output = artifact_root / corpus_dir(corpus_id) / "clusters"
        status = "queued"
        if output.exists():
            matches, reason = _completed_output_matches(output, expected)
            if not matches:
                parser.error(
                    f"{corpus_id}: incompatible existing cluster output: {reason}; "
                    f"move or remove {output}")
            status = "skipped"
        batch_size = (args.warehouse_congested_batch_size
                      if corpus_id == "Warehouse-Congested" else args.batch_size)
        jobs.append({
            "corpus_id": corpus_id,
            "corpus": str(corpus.resolve()),
            "vae": str(vae.resolve()),
            "output": str(output.resolve()),
            "tokens": schema.token_count,
            "latent_width": schema.token_count * hidden_dim,
            "accepted": accepted,
            "batch_size": batch_size,
            "estimated_work": accepted * schema.token_count ** 2,
            "status": status,
        })
    if unavailable and not args.skip_unavailable:
        parser.error("unavailable clustering inputs:\n  " + "\n  ".join(unavailable))
    for message in unavailable:
        print(f"Skipping {message}", flush=True)
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


def parse_args(argv=None):
    defaults = ClusteringConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True,
                        help="Physical GPU IDs; one clustering process per GPU")
    parser.add_argument("--corpus-dir", type=Path, default=REPO / "data/loki")
    parser.add_argument("--output-dir", type=Path, default=REPO / "artifacts/loki")
    parser.add_argument("--corpora", nargs="+", choices=tuple(CORPUS_GROUPS),
                        help="Optional canonical corpus subset")
    parser.add_argument("--n-clusters", type=int, default=defaults.n_clusters)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warehouse-congested-batch-size", type=int,
                        default=DEFAULT_CONGESTED_BATCH_SIZE)
    parser.add_argument("--posterior-seed", type=int,
                        default=defaults.posterior_seed)
    parser.add_argument("--kmeans-seed", type=int, default=defaults.kmeans_seed)
    parser.add_argument("--min-cluster-size", type=int,
                        default=defaults.min_cluster_size)
    parser.add_argument("--threads-per-job", type=int,
                        help="CPU threads per K-means process; default divides CPUs across GPUs and caps at 64 for OpenBLAS")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--skip-unavailable", action="store_true",
                        help="Cluster completed VAEs and skip unfinished ones")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.gpus) != len(set(args.gpus)) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU IDs must be unique nonnegative integers")
    if min(args.n_clusters, args.batch_size, args.warehouse_congested_batch_size,
           args.min_cluster_size) < 1:
        parser.error("cluster count, batch sizes, and minimum size must be positive")
    if min(args.posterior_seed, args.kmeans_seed) < 0:
        parser.error("seeds must be nonnegative")
    if args.threads_per_job is None:
        args.threads_per_job = min(
            64, max(1, (os.cpu_count() or 1) // len(args.gpus)))
    elif args.threads_per_job < 1:
        parser.error("threads-per-job must be positive")
    return args, parser


def main(argv=None):
    args, parser = parse_args(argv)
    jobs = _load_jobs(args, parser)
    for job in jobs:
        job["command"] = _cluster_command(job, args)
    if args.dry_run:
        for index, job in enumerate(job for job in jobs if job["status"] == "queued"):
            gpu = args.gpus[index % len(args.gpus)]
            print(f"GPU {gpu}, CPU threads {args.threads_per_job}: "
                  f"{shlex.join(job['command'])}")
        for job in jobs:
            if job["status"] == "skipped":
                print(f"Skipping compatible completed {job['corpus_id']}")
        return 0

    artifact_root = args.output_dir.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    logs = artifact_root / "cluster_logs"
    logs.mkdir(exist_ok=True)
    summary_path = artifact_root / "clustering_summary.json"
    lock, stop, active = threading.Lock(), threading.Event(), {}
    _save_summary(summary_path, args, jobs)

    def worker(gpu):
        while not stop.is_set():
            with lock:
                job = next((candidate for candidate in jobs
                            if candidate["status"] == "queued"), None)
                if job is None:
                    return
                job.update(status="running", gpu=gpu, started_at=time.time())
                _save_summary(summary_path, args, jobs)
                print(f"[{job['corpus_id']}] GPU {gpu}: "
                      f"{shlex.join(job['command'])}", flush=True)
            environment = dict(
                os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS=str(args.threads_per_job),
                MKL_NUM_THREADS=str(args.threads_per_job),
                OPENBLAS_NUM_THREADS=str(args.threads_per_job),
                NUMEXPR_NUM_THREADS=str(args.threads_per_job),
                PYTHONUNBUFFERED="1")
            started, returncode = time.monotonic(), None
            try:
                log_path = logs / f"{corpus_dir(job['corpus_id'])}.log"
                with log_path.open("a") as stream:
                    stream.write(
                        f"\nLaunch {time.strftime('%Y-%m-%d %H:%M:%S')}: "
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
