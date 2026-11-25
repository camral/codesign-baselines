"""Launch supported baselines over Codesign Gym environment families."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import subprocess
import sys
import threading
import time

from launch_comparison import (add_llm_arguments, add_resource_arguments, command_for,
    manifest_config, wait_with_limits)


METHODS = ("fasttd3", "ppo_ngopt", "ppo_llm", "cmaes", "loki", "fasttd3_cmaes")
DEFAULT_METHODS = ("fasttd3", "ppo_ngopt", "cmaes")
# Registered environments that are intentionally outside the current comparison
# suite. Keep this explicit so registry additions cannot disappear silently.
DEFERRED_ENV_IDS = {"TruckUnload-Multi"} | {
    f"SoftWalker-{model}{terrain}"
    for model in ("Spring2D", "MPM2D", "MPM3D", "Beam3D", "HexFEM3D")
    for terrain in ("", "Stairs", "Gaps")} - {"SoftWalker-Beam3D"}
FAMILIES = {
    "ball_catcher": ("BallCatcher-Pitch", "BallCatcher-Arc", "BallCatcher-Fielding"),
    "openai_gym": ("HalfCheetah-v4", "Swimmer-v4", "Hopper-v4", "Walker2d-v4",
        "Ant-v4", "Humanoid-v4", "HumanoidStandup-v4"),
    "nerograsp": ("NeroGrasp-All", "NeroGrasp-Small", "NeroGrasp-Tiny", "NeroGrasp-Thin",
        "NeroGrasp-SetScrew", "NeroGrasp-TNut", "NeroGrasp-LockNut", "NeroGrasp-Screw",
        "NeroGrasp-Scissors", "NeroGrasp-Tweezers", "NeroGrasp-Spatula", "NeroGrasp-Stick",
        "NeroGrasp-ScrewLong"),
    "lq": ("LQStructure-LQR", "LQStructure-LQG", "LQStructure-QP"),
    "microgrid": ("Microgrid-Random", "Microgrid-Forecast",
        "Microgrid-Campus", "Microgrid-OffGridSingle", "Microgrid-OffGridCampus"),
    "mobile": ("Network-Urban", "Network-Suburban", "Network-Motorway"),
    "racing": ("Racing-Spielberg", "Racing-Levine"),
    "solar_cleaner": ("SolarCleaner-Reach", "SolarCleaner-Stability", "SolarCleaner-Traverse"),
    "truck_unload": ("TruckUnload-Single",),
    "warehouse": ("Warehouse-Small", "Warehouse-Medium", "Warehouse-Congested"),
    # These two need the per-world heightfield mujoco_warp fork that requirements.txt installs.
    "rotate_in_hand": ("RotateInHand-Cube",),
    "hand_pick_up": ("HandPickUp-Cube", "HandPickUp-SquashBall", "HandPickUp-Hammer"),
    # Launch with a small --num-envs (tens): SoftWalker steps its worlds serially.
    "soft_walker": ("SoftWalker-Beam3D",),
}


def eval_gpu(gpu, gpus, eval_gpus):
    """The evaluation GPU for jobs trained on ``gpu`` (None: evaluate on the training GPU)."""
    return None if not eval_gpus else eval_gpus[list(gpus).index(gpu) % len(eval_gpus)]


def slug(env_id):
    return env_id.lower().replace("-v0", "").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_llm_arguments(parser)
    add_resource_arguments(parser)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--env-ids", nargs="+", help="Exact registry IDs; overrides --families")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--hours", type=float, required=True, help="Wall-clock budget per run")
    parser.add_argument("--num-envs", type=int, required=True, help="Shared count for every method and environment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project", default="cdgym-environment-suite")
    parser.add_argument("--entity", default="projectavi")
    parser.add_argument("--group", default="codesign-gym-suite")
    parser.add_argument("--shutdown-grace", type=float, default=60)
    parser.add_argument("--vram-limit-gb", type=float)
    parser.add_argument("--eval-gpus", nargs="+", type=int,
        help="Run each job's evaluation worker on these GPUs instead of its training GPU, round-robin by training GPU "
             "(SoftWalker: a concurrent trainer must be the only Warp process on its GPU)")
    parser.add_argument("--wait-for-free-gpus", action="store_true",
        help="Let each GPU worker wait for existing work to release its GPU before claiming a job")
    parser.add_argument("--gpu-free-threshold-mib", type=float, default=2048,
        help="Maximum used memory at which a waiting GPU is considered free")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env_ids = args.env_ids or [env_id for family in args.families for env_id in FAMILIES[family]]
    if len(env_ids) != len(set(env_ids)) or len(args.seeds) != len(set(args.seeds)) or len(args.gpus) != len(set(args.gpus)) or len(args.methods) != len(set(args.methods)):
        parser.error("environment IDs, seeds, GPUs and methods must not contain duplicates")
    if (not math.isfinite(args.hours) or args.hours <= 0 or args.num_envs < 1
            or any(gpu < 0 for gpu in args.gpus)
            or not math.isfinite(args.gpu_free_threshold_mib) or args.gpu_free_threshold_mib < 0):
        parser.error("hours, num-envs and GPU indices must be valid")
    if args.eval_gpus is not None and (len(set(args.eval_gpus)) != len(args.eval_gpus) or set(args.eval_gpus) & set(args.gpus)
                                       or any(gpu < 0 for gpu in args.eval_gpus)):
        parser.error("eval GPUs must be distinct, non-negative and disjoint from the training GPUs")
    output = Path(args.output_dir).resolve()
    jobs = [{"env_id": env_id, "method": method, "seed": seed, "status": "queued"}
        for env_id in env_ids for seed in args.seeds for method in args.methods]
    if args.dry_run:
        for index, job in enumerate(jobs):
            run_args = SimpleNamespace(**vars(args), env_id=job["env_id"])
            folder = output / slug(job["env_id"]) / f'{job["method"]}_seed_{job["seed"]}' / "artifacts"
            print(json.dumps(command_for(run_args, job["method"], job["seed"], args.gpus[index % len(args.gpus)], folder)))
        return
    output.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()

    def save_manifest():
        temporary = output / "manifest.tmp"
        temporary.write_text(json.dumps(
            {"config": manifest_config(args), "env_ids": env_ids, "runs": jobs}, indent=2) + "\n")
        temporary.replace(output / "manifest.json")

    def wait_for_gpu(gpu):
        announced = False
        while args.wait_for_free_gpus:
            try:
                used = float(subprocess.check_output([
                    "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                if not announced:
                    print(f"[GPU {gpu}] waiting; memory query failed: {error}", flush=True)
                    announced = True
                time.sleep(30)
                continue
            if used <= args.gpu_free_threshold_mib:
                if announced:
                    print(f"[GPU {gpu}] free at {used:.0f} MiB; joining the queue", flush=True)
                return
            if not announced:
                print(f"[GPU {gpu}] occupied at {used:.0f} MiB; waiting to join the queue", flush=True)
                announced = True
            time.sleep(30)

    def worker(gpu):
        while True:
            wait_for_gpu(gpu)
            with lock:
                job = next((item for item in jobs if item["status"] == "queued"), None)
                if job is None:
                    return
                folder = output / slug(job["env_id"]) / f'{job["method"]}_seed_{job["seed"]}'
                folder.mkdir(parents=True)
                run_args = SimpleNamespace(**vars(args), env_id=job["env_id"])
                command = command_for(run_args, job["method"], job["seed"], gpu, folder / "artifacts")
                job.update(status="running", gpu=gpu, command=command, directory=str(folder))
                save_manifest()
                print(f'[{job["env_id"]} {job["method"]} seed={job["seed"]}] GPU {gpu}', flush=True)
            warp_cache = (output / ".warp_cache" / f"gpu_{gpu}").resolve()
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                PYTHONUNBUFFERED="1", WARP_CACHE_PATH=str(warp_cache))
            if eval_gpu(gpu, args.gpus, args.eval_gpus) is not None:
                environment["CODESIGN_EVAL_GPU"] = str(eval_gpu(gpu, args.gpus, args.eval_gpus))
            if args.entity:
                environment["WANDB_ENTITY"] = args.entity
            started, code = time.monotonic(), None
            environment["CODESIGN_RUN_STARTED"] = str(started)
            try:
                with (folder / "train.log").open("w") as stream:
                    process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                        cwd=Path(__file__).resolve().parent, start_new_session=True)
                    try:
                        code = wait_with_limits(process, gpu, args.hours * 3600 + args.shutdown_grace,
                            args.vram_limit_gb, job)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        job.setdefault("error", "deadline watchdog")
            except Exception as error:
                job["error"] = repr(error)
            with lock:
                job.update(status="complete" if code == 0 else "failed", returncode=code,
                    elapsed_seconds=time.monotonic() - started)
                save_manifest()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
            future.result()
    if any(job["status"] != "complete" for job in jobs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
