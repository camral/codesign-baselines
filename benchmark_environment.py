"""Measure standalone Gym stepping throughput, without any baseline learner."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    import torch
    from codesign_gym import make_vec

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env, _ = make_vec('BallCatcher-Pitch', args.worker, device='cuda:0', seed=args.seed)
    try:
        kwargs = {}
        if args.workload == 'random':
            low = torch.as_tensor(env.embodiment_space.low, device='cuda')
            high = torch.as_tensor(env.embodiment_space.high, device='cuda')
            kwargs['embodiment'] = low + torch.rand((args.worker, len(low)), device='cuda') * (high - low)
        _, rejected, _ = env.reset(seed=args.seed, **kwargs)
        if bool(rejected.any()):
            raise RuntimeError('benchmark embodiment rejected')
        actions = torch.zeros((32, args.worker, env.single_action_space.shape[0]), device='cuda')
        if args.workload == 'random':
            actions.uniform_(-1, 1)
        with torch.no_grad():
            for step in range(args.warmup):
                env.step(actions[step % len(actions)])
            torch.cuda.synchronize()
            warm_diverged = env.diverged_worlds
            measurements = []
            for repeat in range(args.repeats):
                torch.cuda.synchronize()
                started, steps = time.perf_counter(), 0
                while time.perf_counter() - started < args.seconds:
                    for _ in range(16):
                        env.step(actions[steps % len(actions)])
                        steps += 1
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                measurements.append({'repeat': repeat, 'vector_steps': steps, 'seconds': elapsed, 'fps': steps * args.worker / elapsed, 'step_ms': elapsed * 1000 / steps})
        free, total = torch.cuda.mem_get_info()
        process_memory = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
        memory = next((float(line.split(',')[1]) for line in process_memory.splitlines() if int(line.split(',')[0]) == os.getpid()), None)
        row = {'num_envs': args.worker, 'workload': args.workload, 'status': 'complete', 'gpu': torch.cuda.get_device_name(), 'measurements': measurements, 'median_fps': statistics.median(item['fps'] for item in measurements), 'min_fps': min(item['fps'] for item in measurements), 'max_fps': max(item['fps'] for item in measurements), 'median_step_ms': statistics.median(item['step_ms'] for item in measurements), 'process_gpu_mib': memory, 'device_used_mib': (total - free) / 2**20, 'torch_peak_mib': torch.cuda.max_memory_allocated() / 2**20, 'diverged_worlds': env.diverged_worlds - warm_diverged, 'action_dim': env.single_action_space.shape[0], 'seed': args.seed, 'torch_version': torch.__version__, 'numpy_version': np.__version__}
        args.output.write_text(json.dumps(row, indent=2) + '\n')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', nargs='+', type=int, default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--workload', choices=['base', 'random'])
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps({**vars(args), 'output': str(args.output), 'protocol': 'public env.step; native autoreset; fixed embodiments; 32 precomputed action batches; no policy, learner, rendering or per-episode embodiment replacement; synchronization every 16 vector steps; separate worker process per count/workload'}, indent=2) + '\n')
    rows = []
    for count in args.counts:
        for workload in ('base', 'random'):
            output = args.output / f'{count}_{workload}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(count), '--workload', workload, '--seconds', str(args.seconds), '--repeats', str(args.repeats), '--warmup', str(args.warmup), '--seed', str(args.seed), '--output', str(output)]
            with output.with_suffix('.log').open('w') as log:
                try:
                    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT, timeout=240)
                    row = json.loads(output.read_text()) if result.returncode == 0 else {'num_envs': count, 'workload': workload, 'status': 'failed', 'returncode': result.returncode}
                except subprocess.TimeoutExpired:
                    row = {'num_envs': count, 'workload': workload, 'status': 'timeout'}
            rows.append(row)
            (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps({key: value for key, value in row.items() if key != 'measurements'}), flush=True)
    flat = [{key: value for key, value in row.items() if key != 'measurements'} for row in rows]
    with (args.output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in flat))))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == '__main__':
    main()
