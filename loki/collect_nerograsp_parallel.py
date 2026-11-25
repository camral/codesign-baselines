"""Finish NeroGrasp-All with parallel conditioned samplers and merge the corpus."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from .collect_all import REPO
from .collectors.storage import (CorpusReader, CorpusStore, corpus_lock,
                                 file_digest, write_json)


def cpu_partitions(count):
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        groups.setdefault((package, core), []).append(cpu)
    buckets = [[] for _ in range(count)]
    for index, siblings in enumerate(groups.values()):
        buckets[index % count].extend(siblings)
    return [",".join(map(str, sorted(bucket))) for bucket in buckets]


def source_count(path):
    manifest = json.loads((path / "manifest.json").read_text())
    return int(manifest["state"]["stats"]["accepted"])


def merge_corpora(primary, workers, target):
    primary = Path(primary)
    sources = [primary, *map(Path, workers)]
    manifests = [json.loads((source / "manifest.json").read_text())
                 for source in sources]
    if any(manifest.get("status") != "complete"
           for manifest in manifests[1:]):
        raise RuntimeError("Every parallel worker corpus must be complete before merge")
    schema = manifests[0]["schema"]
    if any(manifest["schema"] != schema for manifest in manifests[1:]):
        raise ValueError("Parallel NeroGrasp corpus schemas differ")
    total = sum(int(manifest["state"]["stats"]["accepted"])
                for manifest in manifests)
    if total != target:
        raise ValueError(f"Source corpora contain {total} rows, expected {target}")

    merged = primary.with_name(primary.name + ".merging")
    backup = primary.with_name(primary.name + ".pre_parallel")
    if merged.exists() or backup.exists():
        raise FileExistsError(f"Merge paths already exist: {merged} or {backup}")
    reader = CorpusReader(primary)
    identity = {
        "collector": "ParallelNeroGraspMerge",
        "target": target,
        "sampler": {
            "distribution": "mixture of equivalent native joint rejection and independently conditioned fingertip proposals",
            "sample_perturbation_m": manifests[0]["identity"]["sampler"][
                "sample_perturbation_m"],
            "topology": "fixed native template",
            "mountings": "unchanged",
            "validation": "full native geometry validator",
        },
        "sources": [
            {"path": str(source.resolve()),
             "manifest_sha256": file_digest(source / "manifest.json"),
             "accepted": int(manifest["state"]["stats"]["accepted"]),
             "seed": manifest.get("identity", {}).get("seed")}
            for source, manifest in zip(sources, manifests)
        ],
    }
    started = time.monotonic()
    stats = {"attempted": sum(int(m["state"]["stats"].get("attempted", 0))
                              for m in manifests),
             "accepted": 0, "invalid": 0, "duplicates": 0, "errors": 0,
             "reset_audits": sum(int(m["state"]["stats"].get("reset_audits", 0))
                                 for m in manifests),
             "reset_rejections": 0, "last_accept_attempt": 0,
             "rejection_reasons": {}}
    pending, hashes = [], []

    def state():
        return {"stats": stats, "rng": None,
                "elapsed_seconds": time.monotonic() - started}

    with corpus_lock(merged) as root:
        store = CorpusStore(root, reader.codec, identity, resume=False)
        fields = tuple(reader.codec.fields)
        for source in sources:
            for arrays in CorpusReader(source).iter_shards():
                for index, stored_hash in enumerate(arrays["hashes"]):
                    fingerprint = bytes(stored_hash)
                    if fingerprint in store.seen:
                        stats["duplicates"] += 1
                        continue
                    store.seen.add(fingerprint)
                    pending.append({field: arrays[field][index] for field in fields})
                    hashes.append(fingerprint)
                    if len(pending) == 1024:
                        stats["accepted"] += len(pending)
                        store.commit(pending, hashes, state())
                        pending.clear()
                        hashes.clear()
        if pending:
            stats["accepted"] += len(pending)
            store.commit(pending, hashes, state())
        if stats["accepted"] != target:
            raise RuntimeError(
                f"Merge produced {stats['accepted']} unique rows; expected {target}")
        store.update_manifest(state(), "complete", target)
    primary.rename(backup)
    merged.rename(primary)
    return backup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True,
                        help="Physical GPU IDs shared round-robin by workers")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--target", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--corpus", type=Path,
                        default=REPO / "data/loki/NeroGraspAll")
    parser.add_argument("--worker-dir", type=Path,
                        default=REPO / "data/loki/nerograsp_parallel")
    parser.add_argument("--audit-every", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.workers < 1 or args.target < 1 or args.seed < 0
            or not args.gpus or len(args.gpus) != len(set(args.gpus))
            or any(gpu < 0 for gpu in args.gpus)):
        parser.error("workers/target must be positive and GPU IDs unique/nonnegative")
    primary = args.corpus.resolve()
    existing = source_count(primary)
    if existing >= args.target:
        parser.error(f"primary already contains {existing} rows")
    remaining = args.target - existing
    quotas = [remaining // args.workers + (index < remaining % args.workers)
              for index in range(args.workers)]
    affinities = cpu_partitions(args.workers)
    root = args.worker_dir.resolve()
    jobs = []
    for index, (quota, affinity) in enumerate(zip(quotas, affinities)):
        output = root / f"worker_{index:02d}"
        command = [
            "taskset", "-c", affinity,
            sys.executable, "-u", "-m", "loki.collectors",
            "--env-id", "NeroGrasp-All", "--output-dir", str(output),
            "--count", str(quota), "--seed", str(args.seed + index),
            "--device", "cuda:0", "--shard-size", "1024",
            "--audit-every", str(args.audit_every),
            "--max-stalled-attempts", "10000", "--env-kwargs", "{}",
        ]
        if (output / "manifest.json").exists():
            command.append("--resume")
        jobs.append({"index": index, "quota": quota, "output": str(output),
                     "gpu": args.gpus[index % len(args.gpus)],
                     "cpu_affinity": affinity, "command": command,
                     "status": "queued"})
    if args.dry_run:
        print(json.dumps({"existing": existing, "remaining": remaining,
                          "jobs": jobs}, indent=2))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    summary = root / "parallel_summary.json"
    lock, active = threading.Lock(), {}

    def save():
        temporary = summary.with_suffix(".tmp")
        write_json(temporary, {"target": args.target, "existing": existing,
                               "jobs": jobs})
        temporary.replace(summary)

    def run(job):
        environment = dict(os.environ,
                           CUDA_VISIBLE_DEVICES=str(job["gpu"]),
                           PYTHONUNBUFFERED="1")
        started = time.monotonic()
        with lock:
            job["status"] = "running"
            save()
        try:
            with (logs / f"worker_{job['index']:02d}.log").open("a") as stream:
                stream.write(f"\nLaunch: {shlex.join(job['command'])}\n")
                stream.flush()
                process = subprocess.Popen(
                    job["command"], cwd=REPO, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                with lock:
                    active[job["index"]] = process
                code = process.wait()
        except Exception as error:
            code, job["error"] = None, repr(error)
        with lock:
            active.pop(job["index"], None)
            job.update(status="complete" if code == 0 else "failed",
                       returncode=code,
                       elapsed_seconds=time.monotonic() - started)
            save()
        return code

    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            codes = [future.result() for future in
                     [pool.submit(run, job) for job in jobs]]
    except KeyboardInterrupt:
        with lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        return 130
    if any(code != 0 for code in codes):
        return 1
    backup = merge_corpora(primary, [job["output"] for job in jobs], args.target)
    print(json.dumps({"status": "complete", "corpus": str(primary),
                      "backup": str(backup), "accepted": args.target}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
