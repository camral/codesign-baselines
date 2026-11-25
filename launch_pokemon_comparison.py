"""Queue Pokenv baseline/seed jobs with one run per GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from importlib.util import find_spec
import os
import signal
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

from codesign_gym import list_envs


METHODS = ("fasttd3", "ppo_ngopt", "cmaes")
DEFAULT_NUM_ENVS = 8
DEFAULT_SHARDS = 4
POKENV_IDS = tuple(env_id for env_id in list_envs() if env_id.startswith("Pokenv"))


def wait_with_limits(process, gpu, seconds, vram_limit_gb, job):
    deadline = time.monotonic() + seconds
    while process.poll() is None:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(process.args, seconds)
        if vram_limit_gb is not None:
            used = float(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()) * 2**20
            job['peak_sampled_gpu_gb'] = max(job.get('peak_sampled_gpu_gb', 0), used / 1e9)
            if used > vram_limit_gb * 1e9:
                job['error'] = f'VRAM watchdog: GPU usage exceeded {vram_limit_gb} GB'
                raise subprocess.TimeoutExpired(process.args, seconds)
        try:
            return process.wait(timeout=min(2, max(0.01, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
    return process.returncode


def command_for(args, method, seed, gpu, output):
    num_envs = args.num_envs if args.num_envs is not None else DEFAULT_NUM_ENVS
    common = ["--env-id", args.env_id, "--hours", str(args.hours), "--output-dir", str(output), "--evaluation-worlds", str(args.evaluation_worlds),
        "--envs-per-worker", str(args.envs_per_worker)]
    if args.shards is not None: common += ["--shards", str(args.shards)]
    logging = ["--project", args.project, "--group", args.group, "--run-name", f"{args.env_id}_{method}_seed_{seed}"] if args.project else []
    if args.entity and args.project:
        logging += ["--entity", args.entity]
    return [sys.executable, "-m", "pokemon_train", "--method", method, "--seed", str(seed), "--num-envs", str(num_envs), "--device", "cuda:0", *common, *logging]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--env-id", choices=POKENV_IDS, default="Pokenv-Random")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS, help="Shared environment count for every method (default:8)")
    parser.add_argument("--evaluation-worlds", type=int, default=8)
    parser.add_argument("--envs-per-worker", type=int, default=8)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project", default="cdgym-pokemon-random")
    parser.add_argument("--entity", default="projectavi")
    parser.add_argument("--group", default="pokemon-random-comparison")
    parser.add_argument("--shutdown-grace", type=float, default=60)
    parser.add_argument("--vram-limit-gb", type=float, help="Stop a job if sampled GPU memory exceeds this decimal-GB limit; use dedicated GPUs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.vram_limit_gb is not None and (not math.isfinite(args.vram_limit_gb) or args.vram_limit_gb <= 0):
        parser.error('vram-limit-gb must be finite and positive')
    if not math.isfinite(args.hours) or args.hours <= 0 or (args.num_envs is not None and args.num_envs < 1) or args.evaluation_worlds < 1 or args.envs_per_worker < 1 or (args.shards is not None and args.shards < 1) or not math.isfinite(args.shutdown_grace) or args.shutdown_grace < 0 or any(gpu < 0 for gpu in args.gpus):
        parser.error("invalid budget, environment count, GPU or grace")
    for values in (args.seeds, args.gpus, args.methods):
        if len(values) != len(set(values)):
            parser.error("seeds, GPUs and methods must not contain duplicates")
    required = {"fasttd3": ("fast_td3", "torch"), "ppo_ngopt": ("numpy", "nevergrad", "stable_baselines3"), "cmaes": ("numpy", "cma", "torch")}
    modules = {"codesign_gym", "pokemon_codesign_env", *(module for method in args.methods for module in required.get(method, ())) }
    missing = sorted(module for module in modules if find_spec(module) is None)
    if missing:
        parser.error(f"active Python {sys.executable} is missing required modules: {', '.join(missing)}; run with conda run -n codesign python")
    output = Path(args.output_dir).resolve()
    jobs = [{"method": method, "seed": seed, "status": "queued"} for seed in args.seeds for method in args.methods]
    print(f"Launching {len(jobs)} runs with {sys.executable} across GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for index, job in enumerate(jobs):
            print(json.dumps(command_for(args, job["method"], job["seed"], args.gpus[index % len(args.gpus)], output / f'{job["method"]}_seed_{job["seed"]}' / "artifacts")))
        return
    output.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()

    def save_manifest():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps({"config": vars(args), "runs": jobs}, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    def worker(gpu):
        while True:
            with lock:
                job = next((job for job in jobs if job["status"] == "queued"), None)
                if job is None:
                    return
                folder = output / f'{job["method"]}_seed_{job["seed"]}'
                folder.mkdir()
                command = command_for(args, job["method"], job["seed"], gpu, folder / "artifacts")
                job.update(status="running", gpu=gpu, command=command, directory=str(folder))
                save_manifest()
                print(f"[{job['method']} seed={job['seed']}] starting on GPU {gpu}; log: {folder / 'train.log'}", flush=True)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            if args.entity:
                environment["WANDB_ENTITY"] = args.entity
            started = time.monotonic()
            environment["CODESIGN_RUN_STARTED"] = str(started)
            code = None
            try:
                with (folder / "train.log").open("w") as stream:
                    process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent, start_new_session=True)
                    try:
                        code = wait_with_limits(process, gpu, args.hours * 3600 + args.shutdown_grace, args.vram_limit_gb, job)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        job.setdefault("error", "deadline watchdog: checkpoint completion not guaranteed")
            except Exception as error:
                job["error"] = repr(error)
            with lock:
                job.update(status="complete" if code == 0 else "failed", returncode=code, elapsed_seconds=time.monotonic() - started)
                save_manifest()
                print(f"[{job['method']} seed={job['seed']}] {job['status']} after {job['elapsed_seconds']:.1f}s" + (f"; inspect {folder / 'train.log'}" if code != 0 else ""), flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in args.gpus]
        for future in futures:
            future.result()
    if any(job["status"] != "complete" for job in jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
