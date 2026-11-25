"""Concurrent end-to-end random co-design workloads, without an RL learner."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    started = float(os.environ['BENCHMARK_STARTED'])
    import torch
    from codesign_gym import make_vec

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = None
    frames = steps = reset_calls = replacements = 0
    peak_used = 0
    try:
        env, _ = make_vec('BallCatcher-Pitch', args.worker, device='cuda:0', seed=args.seed)
        low = torch.as_tensor(env.embodiment_space.low, device='cuda')
        high = torch.as_tensor(env.embodiment_space.high, device='cuda')
        designs = low + torch.rand((args.worker, len(low)), device='cuda') * (high - low)
        _, rejected, _ = env.reset(seed=args.seed, embodiment=designs)
        if bool(rejected.any()):
            raise RuntimeError('initial random embodiment rejected')
        reset_calls, replacements = 1, args.worker
        done = torch.zeros(args.worker, dtype=torch.bool, device='cuda')
        actions = torch.empty((args.worker, env.single_action_space.shape[0]), device='cuda')
        timed_replacements = torch.zeros((), dtype=torch.long, device='cuda')
        torch.cuda.synchronize()
        initialized = time.monotonic()
        last_time, last_frames = initialized, 0
        intervals = []
        status = 'complete'
        with torch.no_grad():
            while time.monotonic() - started < args.seconds:
                if bool(done.any()):
                    count = int(done.sum())
                    designs[done] = low + torch.rand((count, len(low)), device='cuda') * (high - low)
                    _, rejected, _ = env.reset(embodiment=designs, options={'reset_mask': done})
                    if bool(rejected.any()):
                        raise RuntimeError('replacement random embodiment rejected')
                    reset_calls += 1
                    timed_replacements += count
                actions.uniform_(-1, 1)
                observation, reward, terminated, truncated, info = env.step(actions)
                done = (terminated | truncated).reshape(-1)
                frames += args.worker
                steps += 1
                if steps % 16 == 0:
                    torch.cuda.synchronize()
                    free, total = torch.cuda.mem_get_info()
                    peak_used = max(peak_used, total - free)
                    if total - free > args.vram_gb * 1e9:
                        status = 'memory_limit'
                        break
                    now = time.monotonic()
                    if now - last_time >= 15:
                        intervals.append({'start_seconds': last_time - started, 'end_seconds': now - started, 'frames': frames - last_frames, 'fps': (frames - last_frames) / (now - last_time)})
                        last_time, last_frames = now, frames
                        print(json.dumps({'num_envs': args.worker, 'elapsed_seconds': now - started, 'frames': frames, 'end_to_end_fps': frames / (now - started), 'peak_device_used_gb': peak_used / 1e9}), flush=True)
        torch.cuda.synchronize()
        ended = time.monotonic()
        if ended > last_time:
            intervals.append({'start_seconds': last_time - started, 'end_seconds': ended - started, 'frames': frames - last_frames, 'fps': (frames - last_frames) / (ended - last_time)})
        result = {'status': status, 'num_envs': args.worker, 'physical_gpu': args.gpu, 'gpu': torch.cuda.get_device_name(), 'seed': args.seed, 'budget_seconds': args.seconds, 'elapsed_seconds': ended - started, 'initialization_seconds': initialized - started, 'frames': frames, 'vector_steps': steps, 'end_to_end_fps': frames / (ended - started), 'post_initialization_fps': frames / (ended - initialized), 'reset_calls': reset_calls, 'initial_embodiments': replacements, 'episode_embodiment_replacements': int(timed_replacements), 'diverged_worlds': env.diverged_worlds, 'peak_device_used_gb': peak_used / 1e9, 'vram_limit_gb': args.vram_gb, 'intervals': intervals, 'torch_version': torch.__version__, 'protocol': 'fresh uniform random control each step; fresh independent uniform in-bounds embodiment per completed world; public partial reset; full elapsed denominator includes imports/init/JIT and periodic progress logging; no learner or policy; final report writing/close excluded'}
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--vram-gb', type=float, default=50)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)

    def run(item):
        gpu, count = item
        output = args.output / f'{count}.json'
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(count), '--gpu', str(gpu), '--seconds', str(args.seconds), '--seed', str(args.seed), '--vram-gb', str(args.vram_gb), '--output', str(output)]
        with output.with_suffix('.log').open('w') as log:
            started = time.monotonic()
            completed = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1', BENCHMARK_STARTED=str(started)), stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds + 120)
        if completed.returncode:
            return {'status': 'failed', 'num_envs': count, 'physical_gpu': gpu, 'returncode': completed.returncode}
        return json.loads(output.read_text())

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, [(0, 4096), (1, 8192), (2, 16384)]))
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'intervals'} for row in results], indent=2))


if __name__ == '__main__':
    main()
