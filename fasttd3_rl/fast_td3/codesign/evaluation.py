"""Deterministic held-out co-design evaluation and portable policy snapshots."""

import argparse
import json
import time
from pathlib import Path

import torch

from common.codesign_adapter import make_baseline_vec
from common.gym_batch import EvaluationAccumulator
from ..fast_td3 import Actor
from ..fast_td3_utils import EmpiricalNormalization
from .inference import RolloutInference
from .joint import JointRollout, project_action


def policy_snapshot(actor, normalization, *, env_id, seed, state_dim, action_dim, design_dim, hp, amp):
    norm_state = None if normalization is None else {key: value.detach().cpu().clone() for key, value in normalization.state_dict().items()}
    return {'format_version': 1, 'env_id': env_id, 'seed': seed, 'state_dim': state_dim, 'action_dim': action_dim, 'design_dim': design_dim, 'num_envs': actor.n_envs, 'hp': hp, 'amp': amp, 'actor': {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}, 'normalization': norm_state}


@torch.no_grad()
def evaluate(env, inference, *, seed, deadline=float('inf'), stop_event=None):
    """One episode per world. Finished worlds never contribute a second one."""
    started = time.monotonic()
    device = torch.device(env.device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        rollout = JointRollout(env, seed=seed)
        output = inference(rollout.state(), deterministic=True, update=False)
        _, rejected, _, _, _ = rollout.select_design(output)
        design = rollout.design.clone()
        evaluation = EvaluationAccumulator(env)
        evaluation.reject(rejected, env.rejection_reward)
        while not evaluation.complete and time.monotonic() < deadline and (stop_event is None or not stop_event.is_set()):
            torch.compiler.cudagraph_mark_step_begin()
            state = torch.cat((rollout.observation, design, torch.zeros((env.num_envs, 1), device=device)), dim=-1)
            action = project_action(inference(state, deterministic=True, update=False), design, torch.zeros_like(evaluation.finished), env)
            rollout.observation, reward, terminated, truncated, _ = env.step(action[:, rollout.design_dim:])
            evaluation.add(reward, terminated, truncated)
        complete = evaluation.complete
        rejections = int(rejected.sum())
        result = {'status': 'complete' if complete else 'budget_exhausted', 'seed': seed, 'num_envs': env.num_envs, 'episodes': int(evaluation.finished.sum()), 'frames': evaluation.frames, 'seconds': time.monotonic() - started, 'embodiment_rejections': rejections, 'embodiment_rejection_rate': rejections / max(env.num_envs, 1)}
        if complete:
            result.update(**evaluation.summary(), embodiments=(rollout.low + (design + 1) * 0.5 * (rollout.high - rollout.low)).cpu().tolist())
        return result


@torch.no_grad()
def render_extremes(env, inference, *, seed, indices, deadline=float("inf"), stride=4, stop_event=None, return_status=False):
    """Replay fixed evaluation worlds and render two selected trajectories."""

    rollout = JointRollout(env, seed=seed)
    output = inference(rollout.state(), deterministic=True, update=False)
    _, rejected, _, _, _ = rollout.select_design(output)
    design = rollout.design.clone()
    selected = [int(index) for index in indices]
    initial_frames = {index: env.render(env_idx=index) for index in set(selected)}
    frames = [[initial_frames[index]] for index in selected]
    finished = rejected.clone()
    step = 0
    while not bool(finished[selected].all()) and time.monotonic() < deadline and (stop_event is None or not stop_event.is_set()):
        state = torch.cat((rollout.observation, design, torch.zeros((env.num_envs, 1), device=env.device)), dim=-1)
        action = project_action(inference(state, deterministic=True, update=False), design, torch.zeros_like(finished), env)
        rollout.observation, _, terminated, truncated, _ = env.step(action[:, rollout.design_dim:])
        done = (terminated | truncated).reshape(-1)
        step += 1
        rendered = {}
        for position, index in enumerate(selected):
            if not bool(finished[index]) and (step % stride == 0 or bool(done[index])):
                if index not in rendered:
                    rendered[index] = env.render(env_idx=index)
                frames[position].append(rendered[index])
        finished |= done
    return (frames, bool(finished[selected].all())) if return_status else frames


def load_policy(path, device, compile=True):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    hp = payload['hp']
    actor = Actor(payload['state_dim'], payload['action_dim'], payload['num_envs'], hp['init_scale'], hp['actor_hidden_dim'], std_min=hp['std_min'], std_max=hp['std_max'], device=device)
    actor.load_state_dict(payload['actor'])
    actor.eval()
    norm_state = payload.get('normalization')
    if norm_state is None or not hp.get('obs_normalization', True):
        normalization = None
    else:
        normalization = EmpiricalNormalization(payload['state_dim'], device)
        normalization.load_state_dict(norm_state)
        normalization.eval()
    return payload, RolloutInference(actor, normalization, compile=compile, amp=payload['amp'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1000003)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    payload, inference = load_policy(args.checkpoint, args.device, args.compile)
    env, _ = make_baseline_vec(payload['env_id'], args.num_envs, device=args.device, seed=args.seed,
        is_eval_env=True, control_mode=payload.get('control_mode', 'logits'))
    try:
        result = evaluate(env, inference, seed=args.seed)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: value for key, value in result.items() if key not in ('episode_returns', 'episode_lengths', 'embodiments')}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
