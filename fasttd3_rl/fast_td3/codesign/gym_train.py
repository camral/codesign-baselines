"""FastTD3 distributional critics with executed reset/control joint actions."""

import argparse
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
import yaml

from common.codesign_adapter import make_baseline_vec
from common.run_log import RunLog
from common.async_evaluation import AsyncEvaluator
from common.wall_clock import WallClockBudget
from ..fast_td3 import Actor, Critic
from ..fast_td3_utils import EmpiricalNormalization
from ..hyperparams import BaseArgs
from .joint import JointRollout, project_action
from .inference import RolloutInference
from .evaluation import policy_snapshot


REPLAY_REFERENCE_ENVS = 128
DEFAULT_REPLAY_TRANSITIONS = 51200 * REPLAY_REFERENCE_ENVS


def replay_capacity(buffer_size=None, replay_transitions=None):
    """Legacy vector-step units refer to128 worlds, never the live batch."""
    if buffer_size is not None and replay_transitions is not None:
        raise ValueError('choose buffer-size or replay-transitions, not both')
    capacity = replay_transitions if replay_transitions is not None else (51200 if buffer_size is None else buffer_size) * REPLAY_REFERENCE_ENVS
    if capacity < 1:
        raise ValueError('replay capacity must be positive')
    return capacity


def fit_replay_capacity(requested, state_dim, action_dim, num_envs, max_bytes):
    """Cap storage without reducing the simulator batch or training batch size."""
    bytes_per_transition = 4 * (2 * state_dim + action_dim + 3)
    capacity = min(requested, int(max_bytes) // bytes_per_transition)
    if capacity < num_envs:
        required = num_envs * bytes_per_transition / 2**30
        raise ValueError(f"replay memory budget cannot hold one simulator batch ({required:.3f} GiB)")
    return capacity


class Replay:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity, self.count, self.position = capacity, 0, 0
        self.data = {key: torch.empty((capacity, width), device=device) for key, width in (("state", state_dim), ("action", action_dim), ("next", state_dim), ("reward", 1), ("bootstrap", 1), ("discount", 1))}

    def add(self, state, action, reward, next_state, bootstrap, discount):
        if len(state) > self.capacity:
            raise ValueError('replay capacity must hold a complete insertion batch')
        count = len(state)
        indices = (torch.arange(count, device=state.device) + self.position) % self.capacity
        for key, value in zip(self.data, (state, action, next_state, reward, bootstrap, discount)):
            self.data[key][indices] = value.reshape(count, -1)
        self.position = (self.position + count) % self.capacity
        self.count = min(self.capacity, self.count + count)

    def sample(self, size):
        index = torch.randint(self.count, (size,), device=self.data["state"].device)
        return {key: value[index] for key, value in self.data.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="BallCatcher-Pitch")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--evaluation-worlds", type=int, default=128)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--rejection-reward", type=float)
    replay_args = parser.add_mutually_exclusive_group()
    replay_args.add_argument("--buffer-size", type=int, help="Legacy units: each unit is128 transitions, independent of num-envs")
    replay_args.add_argument("--replay-transitions", type=int, help="Requested transition capacity; default 6,553,600, capped by the memory budget")
    parser.add_argument("--replay-memory-gb", type=float, default=8.0,
        help="Replay storage cap in GiB, further limited to half the currently free CUDA memory")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    capacity = replay_capacity(args.buffer_size, args.replay_transitions)
    if min(args.num_envs, args.batch_size, args.evaluation_worlds, args.video_stride, args.video_fps) < 1 or capacity < args.num_envs or not math.isfinite(args.replay_memory_gb) or args.replay_memory_gb <= 0:
        parser.error('positive environment/batch sizes and replay capacity >= num-envs are required')
    budget = WallClockBudget.start(args.hours)
    hp = asdict(BaseArgs())
    amp_enabled = args.amp and torch.device(args.device).type == "cuda"
    log = RunLog(args.output_dir, budget, {**hp, **vars(args), "method": "fasttd3_joint", "amp_effective": amp_enabled, "source_preset": "FastTD3 BaseArgs (HumanoidBench)", "design_transition_discount": 1.0, "replay_capacity_requested": capacity, "replay_reference_envs": REPLAY_REFERENCE_ENVS, "evaluation_execution": "independent_process", "evaluation_protocol": "current deterministic joint policy selects one embodiment per evaluation world"}, project=args.project, entity=args.entity, group=args.group, name=args.run_name)
    env = evaluator = None
    try:
        evaluator = AsyncEvaluator(log, args, "fasttd3")
        torch.manual_seed(args.seed)
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = hp["torch_deterministic"]
        env, _ = make_baseline_vec(args.env_id, args.num_envs, device=args.device, seed=args.seed,
            control_mode="logits", rejection_reward=args.rejection_reward)
        if capacity < env.num_envs:
            raise ValueError(f"replay capacity {capacity} is smaller than the {env.num_envs} simulator worlds")
        rollout = JointRollout(env)
        state_dim, design_dim = rollout.state().shape[-1], rollout.design_dim
        action_dim = design_dim + env.single_action_space.shape[0]
        actor = Actor(state_dim, action_dim, env.num_envs, hp["init_scale"], hp["actor_hidden_dim"], std_min=hp["std_min"], std_max=hp["std_max"], device=args.device)
        critic = Critic(state_dim, action_dim, hp["num_atoms"], hp["v_min"], hp["v_max"], hp["critic_hidden_dim"], "", 64, 8, device=args.device)
        target = deepcopy(critic)
        normalization = EmpiricalNormalization(state_dim, args.device)
        inference = RolloutInference(actor, normalization, compile=args.compile, amp=amp_enabled)
        actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=hp["actor_learning_rate"], weight_decay=hp["weight_decay"])
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=hp["critic_learning_rate"], weight_decay=hp["weight_decay"])
        requested_capacity = capacity
        max_replay_bytes = args.replay_memory_gb * 2**30
        if torch.device(args.device).type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(args.device)
            max_replay_bytes = min(max_replay_bytes, free_bytes // 2)
        capacity = fit_replay_capacity(capacity, state_dim, action_dim, env.num_envs, max_replay_bytes)
        log.log(replay_capacity_requested=requested_capacity, replay_memory_budget_bytes=int(max_replay_bytes),
                replay_capacity_transitions=capacity)
        replay = Replay(capacity, state_dim, action_dim, args.device)
        log.log(replay_capacity_transitions=capacity, replay_bytes=sum(value.numel() * value.element_size() for value in replay.data.values()))
        steps, updates, frames = 0, 0, 0
        rejections = candidates = interval_rejections = interval_candidates = 0
        episode_returns = torch.zeros(env.num_envs, device=args.device)
        train_reward = None

        def checkpoint_payload():
            return {"format_version": 2, "env_id": args.env_id, "seed": args.seed, "state_dim": state_dim, "action_dim": action_dim, "design_dim": design_dim, "num_envs": env.num_envs, "num_designs": env.num_designs, "hp": hp, "amp": amp_enabled, "actor": actor.state_dict(), "critic": critic.state_dict(), "target": target.state_dict(), "normalization": normalization.state_dict(), "actor_optimizer": actor_optimizer.state_dict(), "critic_optimizer": critic_optimizer.state_dict(), "design": rollout.design, "design_low": rollout.low, "design_high": rollout.high, "control_mode": "logits"}

        def publish_snapshot(force=False):
            evaluator.publish(lambda: policy_snapshot(actor, normalization,
                env_id=args.env_id, seed=args.seed, state_dim=state_dim,
                action_dim=action_dim, design_dim=design_dim, hp=hp, amp=amp_enabled),
                force=force, vector_steps=steps, updates=updates, frames=frames,
                **{"train/reward": train_reward, "train/fps": frames / max(budget.elapsed, 1e-9)})

        publish_snapshot(force=True)

        def projected(state, output):
            return project_action(output, state[:, -design_dim - 1:-1], state[:, -1] > 0.5, env)

        def update_critic(data, obs, next_obs):
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=amp_enabled):
                with torch.no_grad():
                    noise = (torch.randn_like(data["action"]) * hp["policy_noise"]).clamp(-hp["noise_clip"], hp["noise_clip"])
                    next_action = projected(data["next"], (actor(next_obs) + noise).clamp(-1, 1))
                    p1, p2 = target.projection(next_obs, next_action, data["reward"][:, 0], data["bootstrap"][:, 0], data["discount"][:, 0])
                    target_dist = torch.where((target.get_value(p1) < target.get_value(p2))[:, None], p1, p2)
                q1, q2 = critic(obs, data["action"])
                loss = -(target_dist * (F.log_softmax(q1, dim=1) + F.log_softmax(q2, dim=1))).sum(dim=1).mean()
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            critic_optimizer.step()
            return loss.detach()

        def update_actor(data, obs):
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=amp_enabled):
                action = projected(data["state"], actor(obs))
                q1, q2 = critic(obs, action)
                actor_loss = -torch.minimum(critic.get_value(q1.softmax(1)), critic.get_value(q2.softmax(1))).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()
            return actor_loss.detach()

        if args.compile:
            update_critic = torch.compile(update_critic, mode=hp["compile_mode"])
            update_actor = torch.compile(update_actor, mode=hp["compile_mode"])

        while not budget.expired:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                if bool(rollout.design_phase.any()):
                    output = inference(rollout.state(), dones=rollout.design_phase)
                    selected, rejected, state, executed, next_state = rollout.select_design(output)
                    candidate_selected = env.collapse_rejections(selected)
                    candidate_rejected = env.collapse_rejections(rejected)
                    new_candidates = int(candidate_selected.sum())
                    new_rejections = int(candidate_rejected[candidate_selected].sum())
                    candidates += new_candidates
                    rejections += new_rejections
                    interval_candidates += new_candidates
                    interval_rejections += new_rejections
                    rejected = rejected[selected]
                    reward = torch.where(rejected, torch.full_like(rejected, env.rejection_reward, dtype=torch.float32), torch.zeros_like(rejected, dtype=torch.float32))
                    replay.add(state[selected], executed[selected], reward, next_state[selected], (~rejected).float(), torch.ones_like(reward))
                output = inference(rollout.state())
                state, executed, reward, next_state, terminated, truncated, info = rollout.step(output)
                replay.add(state, executed, reward, next_state, (~terminated).float(), torch.full_like(reward, hp["gamma"]))
                episode_returns += reward.reshape(-1)
                done = (terminated | truncated).reshape(-1)
                if bool(done.any()):
                    train_reward = float(episode_returns[done].mean())
                    episode_returns[done] = 0
                frames += env.num_envs
            if steps > hp["learning_starts"]:
                for update_index in range(hp["num_updates"]):
                    if budget.expired:
                        break
                    data = replay.sample(args.batch_size)
                    obs = inference.normalize(data["state"])
                    next_obs = inference.normalize(data["next"])
                    torch.compiler.cudagraph_mark_step_begin()
                    update_critic(data, obs, next_obs)
                    if update_index % hp["policy_frequency"] == 1:
                        update_actor(data, obs)
                    with torch.no_grad():
                        for source, destination in zip(critic.parameters(), target.parameters()):
                            destination.lerp_(source, hp["tau"])
                    updates += 1
            steps += 1
            publish_snapshot()
            if steps % 100 == 0:
                metrics = {"vector_steps": steps, "updates": updates, "frames": frames, "fps": frames / max(budget.elapsed, 1e-9),
                    "candidates": candidates, "rejections": rejections, "embodiment_candidates": interval_candidates,
                    "embodiment_rejections": interval_rejections, "embodiment_rejection_rate": interval_rejections / max(interval_candidates, 1)}
                if train_reward is not None:
                    metrics["train/reward"] = train_reward
                log.log(**metrics)
                interval_rejections = interval_candidates = 0
        evaluator.close()
        torch.save(checkpoint_payload(), log.output / "checkpoint.pt")
        if not (log.output / "best_checkpoint.pt").exists():
            torch.save(checkpoint_payload(), log.output / "best_checkpoint.pt")
        log.finish("complete", vector_steps=steps, updates=updates, frames=frames, candidates=candidates, rejections=rejections, best_eval_reward=evaluator.best_eval_reward)
    except Exception:
        if evaluator is not None:
            evaluator.close(raise_errors=False)
        log.finish("failed", error=traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        if evaluator is not None:
            evaluator.close(raise_errors=False)


if __name__ == "__main__":
    main()
