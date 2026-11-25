"""Queue native Codesign Gym baseline/seed jobs with one run per GPU."""

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

from ppo_llm_rlgo.config import add_arguments as add_llm_arguments, command_arguments as llm_command_arguments


METHODS = ("fasttd3", "ppo_ngopt", "ppo_llm", "cmaes", "loki", "fasttd3_cmaes")
DEFAULT_NUM_ENVS = 16384


def manifest_config(args):
    """Return argparse values in the JSON-native form used by run manifests."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def watchdog_seconds(method, hours, shutdown_grace):
    return None if method == "ppo_llm" else hours * 3600 + shutdown_grace


def wait_with_limits(process, gpu, seconds, vram_limit_gb, job):
    deadline = None if seconds is None else time.monotonic() + seconds
    while process.poll() is None:
        if deadline is not None and time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(process.args, seconds)
        if vram_limit_gb is not None:
            used = float(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).strip()) * 2**20
            job['peak_sampled_gpu_gb'] = max(job.get('peak_sampled_gpu_gb', 0), used / 1e9)
            if used > vram_limit_gb * 1e9:
                job['error'] = f'VRAM watchdog: GPU usage exceeded {vram_limit_gb} GB'
                raise subprocess.TimeoutExpired(process.args, seconds)
        try:
            timeout = 2 if deadline is None else min(2, max(0.01, deadline - time.monotonic()))
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    return process.returncode


def add_resource_arguments(parser):
    parser.add_argument("--replay-transitions", type=int,
        help="FastTD3 requested transition count, further capped by replay memory")
    parser.add_argument("--replay-memory-gb", type=float,
        help="FastTD3 replay cap in GiB (default: 4 for SoftWalker, whose 8101-wide conditioned observations OOM an L4 at 8, else 8); also limited to half of free CUDA memory")
    parser.add_argument("--cma-population-size", type=int, default=128,
        help="CMA candidates per generation; extra worlds repeat candidates")
    parser.add_argument("--cma-covariance", choices=("diagonal", "full"), default="diagonal")
    parser.add_argument("--search-frequency", type=int,
        help="FastTD3 + CMA-ES: FastTD3 vector steps before each CMA-ES generation (default: one episode in every world)")
    parser.add_argument("--evaluation-worlds", type=int,
        help="Held-out worlds; LOKI also selects this many ranked designs (default: 8 for SoftWalker, else 128)")
    parser.add_argument("--loki-candidates", type=int,
        help="LOKI candidates scored per cluster and search (default: 16 for SoftWalker, else LOKI's 128)")
    parser.add_argument("--loki-corpus-root", type=Path, default=Path("data/loki"))
    parser.add_argument("--loki-artifact-root", type=Path, default=Path("artifacts/loki"))
    parser.add_argument("--loki-rollout-steps", type=int, default=19,
        help="Per-world LOKI PPO rollout length; 19 preserves the upstream "
             "approximately 51,200 transitions per cluster update at 16,384 total worlds")


def evaluation_worlds(args, env_id):
    """SoftWalker steps its worlds serially, so its held-out evaluation uses fewer of them."""
    if getattr(args, "evaluation_worlds", None) is not None:
        return args.evaluation_worlds
    return 8 if env_id.startswith("SoftWalker") else 128


def replay_memory_gb(args, env_id):
    """FastTD3 replay cap; SoftWalker's conditioned observations are 8101 floats, so 8 GiB leaves too little of an L4."""
    if getattr(args, "replay_memory_gb", None) is not None:
        return args.replay_memory_gb
    return 4.0 if env_id.startswith("SoftWalker") else 8.0


def loki_candidates(args, env_id):
    """SoftWalker scores each candidate in a serial world, so its LOKI search scores fewer (None: LOKI's default)."""
    if getattr(args, "loki_candidates", None) is not None:
        return args.loki_candidates
    return 16 if env_id.startswith("SoftWalker") else None


def command_for(args, method, seed, gpu, output):
    num_envs = args.num_envs if args.num_envs is not None else DEFAULT_NUM_ENVS
    env_id = getattr(args, "env_id", "BallCatcher-Pitch")
    env_name = env_id.lower().replace("-v0", "").replace("-", "_")
    common = ["--hours", str(args.hours), "--output-dir", str(output)]
    logging = ["--project", args.project, "--group", args.group] if args.project else []
    if args.entity and args.project:
        logging += ["--entity", args.entity]
    logging = ["--project", args.project, "--group", args.group, "--run-name", f"{env_name}_{method}_seed_{seed}"] if args.project else []
    if args.entity and args.project:
        logging += ["--entity", args.entity]
    modules = {"fasttd3": "fast_td3.codesign.gym_train", "ppo_ngopt": "ppo_ngopt_rlgo.gym_train", "ppo_llm": "ppo_llm_rlgo.gym_train", "cmaes": "cmaes_evolutionary.gym_train", "loki": "loki.gym_train", "fasttd3_cmaes": "fasttd3_cmaes.gym_train"}
    extra = llm_command_arguments(args) if method == "ppo_llm" else []
    extra += ["--evaluation-worlds", str(evaluation_worlds(args, env_id))]
    if method in ("fasttd3", "fasttd3_cmaes") and getattr(args, "replay_transitions", None) is not None:
        extra = [*extra, "--replay-transitions", str(args.replay_transitions)]
    if method in ("fasttd3", "fasttd3_cmaes"):
        extra += ["--replay-memory-gb", str(replay_memory_gb(args, env_id))]
    if method == "fasttd3_cmaes" and getattr(args, "search_frequency", None) is not None:
        extra += ["--search-frequency", str(args.search_frequency)]
    if method in ("cmaes", "fasttd3_cmaes"):
        if getattr(args, "cma_population_size", None) is not None:
            extra += ["--population-size", str(args.cma_population_size)]
        if getattr(args, "cma_covariance", None) is not None:
            extra += ["--covariance", args.cma_covariance]
    if method == "loki":
        from loki.collect_all import PRESET_TO_CORPUS, corpus_dir
        canonical = corpus_dir(PRESET_TO_CORPUS[env_id])
        corpus_root = Path(getattr(args, "loki_corpus_root", "data/loki"))
        artifact_root = Path(getattr(args, "loki_artifact_root", "artifacts/loki"))
        extra += ["--rollout-steps",
                  str(getattr(args, "loki_rollout_steps", 19)),
                  "--corpus", str(corpus_root / canonical),
                  "--cluster-artifact",
                  str(artifact_root / canonical / "clusters")]
        if loki_candidates(args, env_id) is not None:
            extra += ["--candidates", str(loki_candidates(args, env_id))]
    return [sys.executable, "-m", modules[method], "--env-id", env_id, "--seed", str(seed), "--num-envs", str(num_envs), "--device", "cuda:0", *common, *logging, *extra]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    add_resource_arguments(parser)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["fasttd3", "ppo_ngopt", "cmaes"])
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS, help="Shared environment count for every method (default:16384)")
    parser.add_argument("--env-id", default="BallCatcher-Pitch")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project", default="cdgym-ball-catcher")
    parser.add_argument("--entity", default="projectavi")
    parser.add_argument("--group", default="ball-catcher-comparison")
    parser.add_argument("--shutdown-grace", type=float, default=60)
    parser.add_argument("--vram-limit-gb", type=float, help="Stop a job if sampled GPU memory exceeds this decimal-GB limit; use dedicated GPUs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.vram_limit_gb is not None and (not math.isfinite(args.vram_limit_gb) or args.vram_limit_gb <= 0):
        parser.error('vram-limit-gb must be finite and positive')
    if not math.isfinite(args.hours) or args.hours <= 0 or (args.num_envs is not None and args.num_envs < 1) or args.loki_rollout_steps < 1 or not math.isfinite(args.shutdown_grace) or args.shutdown_grace < 0 or any(gpu < 0 for gpu in args.gpus):
        parser.error("invalid budget, environment count, GPU or grace")
    for values in (args.seeds, args.gpus, args.methods):
        if len(values) != len(set(values)):
            parser.error("seeds, GPUs and methods must not contain duplicates")
    required = {"fasttd3": ("fast_td3", "torch"), "ppo_ngopt": ("numpy", "nevergrad", "stable_baselines3"), "ppo_llm": ("numpy", "nevergrad", "stable_baselines3"), "cmaes": ("numpy", "cma", "torch"), "loki": ("numpy", "torch", "stable_baselines3"), "fasttd3_cmaes": ("fast_td3", "torch", "numpy", "nevergrad")}
    if "ppo_llm" in args.methods and args.llm_gateway == "huggingface":
        required["ppo_llm"] += ("transformers", "accelerate")
    modules = {"codesign_gym", *(module for method in args.methods for module in required.get(method, ())) }
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
        temporary.write_text(json.dumps(
            {"config": manifest_config(args), "runs": jobs}, indent=2) + "\n")
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
            warp_cache = (output / ".warp_cache" / f"gpu_{gpu}").resolve()
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                WARP_CACHE_PATH=str(warp_cache))
            if args.entity:
                environment["WANDB_ENTITY"] = args.entity
            started = time.monotonic()
            environment["CODESIGN_RUN_STARTED"] = str(started)
            code = None
            try:
                with (folder / "train.log").open("w") as stream:
                    process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parent, start_new_session=True)
                    try:
                        code = wait_with_limits(process, gpu, watchdog_seconds(job["method"], args.hours, args.shutdown_grace), args.vram_limit_gb, job)
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
