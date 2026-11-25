"""Timed CMA-ES with independent population size and repeated-world fitness."""

import argparse
import traceback

import numpy as np
import torch

from common.codesign_adapter import make_baseline_vec
from common.gym_batch import EvaluationAccumulator
from common.embodiment_conditioning import condition_observation
from common.run_log import RunLog
from common.async_evaluation import AsyncEvaluator
from common.wall_clock import WallClockBudget
from .main import CMAESConfig, build_optimizer, decode_design


def policy_size(observation_size, action_size, hidden=32):
    return observation_size * hidden + hidden + hidden * action_size + action_size


def control(observation, weights, action_size, hidden=32):
    width = observation.shape[-1]
    first = width * hidden
    second = first + hidden
    third = second + hidden * action_size
    if weights.ndim == 2:
        first_layer = weights[:, :first].reshape(-1, width, hidden)
        state = torch.tanh(torch.bmm(observation[:, None, :], first_layer)[:, 0] + weights[:, first:second])
        second_layer = weights[:, second:third].reshape(-1, hidden, action_size)
        return torch.tanh(torch.bmm(state[:, None, :], second_layer)[:, 0] + weights[:, third:])
    state = torch.tanh(observation @ weights[:first].reshape(width, hidden) + weights[first:second])
    return torch.tanh(state @ weights[second:third].reshape(hidden, action_size) + weights[third:])


def population_assignment(num_designs, population_size):
    """Use every design slot, balancing repeated trials across candidates."""
    if min(num_designs, population_size) < 1 or population_size > num_designs:
        raise ValueError("population must fit in the available design slots")
    return np.arange(num_designs) % population_size


def population_scores(returns, rejected, assignment, population_size):
    counts = np.bincount(assignment, minlength=population_size)
    scores = np.bincount(assignment, weights=returns, minlength=population_size) / counts
    invalid = np.bincount(assignment, weights=np.asarray(rejected, dtype=float),
                          minlength=population_size) > 0
    return scores, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="BallCatcher-Pitch")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--population-size", type=int, default=128,
        help="CMA candidates per generation; remaining simulator worlds repeat candidates")
    parser.add_argument("--covariance", choices=("diagonal", "full"), default="diagonal",
        help="Diagonal covariance scales to large neural controllers; full remains available")
    parser.add_argument("--evaluation-worlds", type=int, default=128)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--rejection-reward", type=float)
    parser.add_argument("--project")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    if min(args.num_envs, args.population_size, args.evaluation_worlds, args.video_stride, args.video_fps) < 1:
        parser.error("environment and video settings must be positive")
    budget = WallClockBudget.start(args.hours)
    log = RunLog(args.output_dir, budget, {**vars(args), "method": "cmaes", "controller": "32-unit tanh MLP", "covariance": args.covariance, "evaluation_execution": "independent_process", "evaluation_protocol": "best completed CMA-ES embodiment-policy pair in every evaluation world; initial mean candidate until the first completion"}, project=args.project, entity=args.entity, group=args.group, name=args.run_name)
    env = evaluator = None
    frames, evaluations, rejections, best = 0, 0, 0, None
    best_value = None
    try:
        evaluator = AsyncEvaluator(log, args, "cmaes")
        torch.manual_seed(args.seed)
        env, base = make_baseline_vec(args.env_id, args.num_envs, device=args.device, seed=args.seed,
            control_mode="logits", rejection_reward=args.rejection_reward)
        obs_size = env.single_observation_space.shape[0]
        act_size = env.single_action_space.shape[0]
        codec = env.design_codec
        binary_design = bool(codec.discrete.all()) and np.all(codec.low == 0) and np.all(codec.high == 1)
        population_size = min(args.population_size, env.num_designs)
        assignment = population_assignment(env.num_designs, population_size)
        log.log(cma_population_size=population_size, cma_design_slots=env.num_designs,
                cma_repeats_min=int(np.bincount(assignment).min()),
                cma_repeats_max=int(np.bincount(assignment).max()),
                cma_binary_logits=bool(binary_design))
        config = CMAESConfig(len(base), policy_size(obs_size + len(base), act_size), 1000000000, num_workers=population_size,
            design_lower_bound=env.embodiment_space.low, design_upper_bound=env.embodiment_space.high,
            design_initial=base, seed=args.seed, population_size=population_size,
            binary_design=binary_design, diagonal=args.covariance == "diagonal")
        initial = config.validated_arrays()

        def publish_snapshot(force=False):
            design, policy = best_value if best_value is not None else (initial["design_initial"], initial["policy_initial"])
            evaluator.publish(lambda: {"design": design.copy(), "policy": policy.copy(),
                "observation_size": obs_size, "action_size": act_size,
                "embodiment_conditioned": True}, force=force,
                kind="incumbent" if best_value is not None else "initial_candidate",
                frames=frames,
                **{"eval/snapshot_completed_candidates": evaluations, "train/reward": best, "train/fps": frames / max(budget.elapsed, 1e-9)})

        publish_snapshot(force=True)
        optimizer = build_optimizer(config)

        while not budget.expired:
            evaluator.check()
            candidates = [optimizer.ask() for _ in range(population_size)]
            if budget.expired:
                break
            designs = np.stack([decode_design(candidate.value["design"], binary_design) for candidate in candidates])
            weights = np.stack([candidate.value["policy"] for candidate in candidates])
            observation, rejected, _ = env.reset(seed=args.seed, embodiment=designs[assignment])
            result = EvaluationAccumulator(env)
            result.reject(rejected, env.rejection_reward)
            weights_t = torch.as_tensor(env.expand_design_values(weights[assignment]), dtype=torch.float32, device=args.device)
            while not result.complete and not budget.expired:
                with torch.no_grad():
                    action = control(condition_observation(env, observation, designs[assignment]),
                                     weights_t, act_size)
                observation, reward, terminated, truncated, _ = env.step(action)
                result.add(reward, terminated, truncated)
                frames += env.num_envs
            if not result.complete:
                break  # Never give the optimizer a partial-episode fitness.
            scores, rejected_array = population_scores(
                env.collapse_returns(result.returns).cpu().numpy(),
                env.collapse_rejections(rejected).cpu().numpy(), assignment, population_size)
            for index, (candidate, score) in enumerate(zip(candidates, scores)):
                optimizer.tell(candidate, -float(score))
                evaluations += 1
                if rejected_array[index]:
                    rejections += 1
                elif best is None or score > best:
                    best = float(score)
                    best_value = (designs[index].copy(), weights[index].copy())
                    publish_snapshot(force=True)
                    np.savez(log.output / "best.npz", design=best_value[0], policy=best_value[1], score=best,
                        observation_size=obs_size, action_size=act_size, embodiment_conditioned=True)
            log.log(evaluations=evaluations, rejections=rejections, embodiment_rejections=int(rejected_array.sum()),
                embodiment_rejection_rate=float(rejected_array.mean()),
                population_size=population_size, frames=frames, fps=frames / max(budget.elapsed, 1e-9), score=float(scores.mean()),
                score_std=float(scores.std()), best_score=best, **{"train/reward": float(scores.mean())})
            publish_snapshot()
        evaluator.close()
        if best is None:
            raise TimeoutError("budget expired before any full candidate evaluation")
        optimizer.dump(log.output / "optimizer.pkl")
        if not (log.output / "best_checkpoint.npz").exists():
            np.savez(log.output / "best_checkpoint.npz", design=best_value[0], policy=best_value[1], eval_reward=np.nan, observation_size=obs_size, action_size=act_size, embodiment_conditioned=True)
        log.finish("complete", evaluations=evaluations, rejections=rejections, frames=frames, best_score=best, best_eval_reward=evaluator.best_eval_reward)
    except Exception:
        if evaluator is not None:
            evaluator.close(raise_errors=False)
        log.finish("failed", error=traceback.format_exc(), evaluations=evaluations, frames=frames)
        raise
    finally:
        if env is not None:
            env.close()
        if evaluator is not None:
            evaluator.close(raise_errors=False)


if __name__ == "__main__":
    main()
