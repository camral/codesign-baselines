"""Alternate retained SB3 PPO training and frozen-policy NGOpt design rounds."""

import argparse
import io
import traceback

import nevergrad as ng
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from common.codesign_adapter import make_baseline_vec
from common.gym_batch import EvaluationAccumulator, GymBatch
from common.embodiment_conditioning import condition_observation
from common.nevergrad_compat import install_numpy_scalar_compat
from common.run_log import RunLog
from common.async_evaluation import AsyncEvaluator
from common.sb3_env import SB3GymBatch
from common.wall_clock import WallClockBudget
from .timed_ppo import TimedPPO, TrainingDeadlineReached


# SB3 2.9.0 PPO defaults with the pre-existing n_steps=256; the 65,536-sample batch gives 64 minibatches per epoch at 16,384 worlds.
PPO_SETTINGS = dict(n_steps=256, batch_size=65536, learning_rate=3e-4, gamma=0.99,
    n_epochs=10, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
    normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    use_sde=False, sde_sample_freq=-1, target_kl=None,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh, ortho_init=True))


def ask_candidate_batch(optimizer, count, budget, progress=None):
    if hasattr(optimizer, "ask_batch"):
        return optimizer.ask_batch(count, budget, progress=progress)
    candidates = []
    while len(candidates) < count and not budget.expired:
        if progress is not None:
            progress()
        try:
            candidates.append(optimizer.ask())
        except AssertionError:
            continue
    return candidates


def tell_candidate_batch(optimizer, candidates, returns, rejected, rejection_reward):
    scores = np.asarray(returns, dtype=float).copy()
    rejected = np.asarray(rejected, dtype=bool).reshape(-1)
    scores[rejected] = rejection_reward
    best_score, best_design = None, None
    for candidate, score, invalid in zip(candidates, scores, rejected):
        if hasattr(optimizer, "tell_result"):
            optimizer.tell_result(candidate, float(score), bool(invalid))
        else:
            optimizer.tell(candidate, -float(score))
        if not invalid and (best_score is None or score > best_score):
            best_score, best_design = float(score), np.asarray(candidate.value).copy()  # value may be a tuple (discrete param)
    if hasattr(optimizer, "end_round"):
        optimizer.end_round()
    return scores, best_score, best_design


def replace_rejected_designs(designs, rejected, fallback):
    designs = np.asarray(designs).copy()
    designs[np.asarray(rejected, dtype=bool).reshape(-1)] = fallback
    return designs


def repeat_designs(designs, count):
    """Spread a bounded search population deterministically over PPO worlds."""
    designs = np.asarray(designs)
    if not len(designs):
        raise ValueError("at least one searched design is required")
    return designs[np.arange(count) % len(designs)].copy()


def deterministic_policy_actions(model, observations, action_space):
    """Run SB3's policy without round-tripping a GPU batch through NumPy."""
    with torch.inference_mode():
        actions, _, _ = model.policy(observations, deterministic=True)
    if hasattr(action_space, "low") and hasattr(action_space, "high"):
        low = torch.as_tensor(action_space.low, device=actions.device,
            dtype=actions.dtype)
        high = torch.as_tensor(action_space.high, device=actions.device,
            dtype=actions.dtype)
        actions = torch.maximum(torch.minimum(actions, high), low)
    return actions


def main(method="ppo_ngopt"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="BallCatcher-Pitch")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ppo-iterations", type=int, default=1)
    parser.add_argument("--ngopt-candidates", type=int, default=64,
        help="maximum frozen-policy designs scored per NGOpt round")
    parser.add_argument("--evaluation-worlds", type=int, default=128)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--rejection-reward", type=float)
    parser.add_argument("--project")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--run-name")
    if method == "ppo_llm":
        from ppo_llm_rlgo.sampler import LLMSampler, add_arguments, validate_arguments
        add_arguments(parser)
    args = parser.parse_args()
    if method == "ppo_llm":
        try:
            validate_arguments(args)
        except ValueError as error:
            parser.error(str(error))
    if min(args.num_envs, args.ppo_iterations, args.ngopt_candidates,
            args.evaluation_worlds, args.video_stride, args.video_fps) < 1:
        parser.error("batch and iteration counts must be positive")
    args.ngopt_candidates = min(args.ngopt_candidates, args.num_envs)
    if method == "ppo_ngopt" and args.num_envs % args.ngopt_candidates:
        parser.error("num-envs must be divisible by ngopt-candidates")
    if method == "ppo_llm" and args.llm_gateway == "manual" and args.evaluation_worlds < min(args.llm_templates, args.num_envs):
        parser.error("manual evaluation-worlds must fit all template renders")
    budget = WallClockBudget.start(args.hours)
    logged_settings = {**PPO_SETTINGS, "policy_kwargs": {**PPO_SETTINGS["policy_kwargs"], "activation_fn": "torch.nn.Tanh"}}
    search_name = "LLM" if method == "ppo_llm" else "NGOpt"
    schedule = (f"one PPO update followed by one bounded frozen-policy {search_name} "
        "candidate round in the training environment"
        if method == "ppo_ngopt" else
        f"one complete PPO rollout before each {search_name} round; one candidate per environment")
    log = RunLog(args.output_dir, budget, {**vars(args), **logged_settings, "go_candidates": args.ngopt_candidates if method == "ppo_ngopt" else args.num_envs, "method": method, "schedule_source": schedule, "design_search_execution": "training_process", "evaluation_execution": "independent_process", "evaluation_protocol": f"best {search_name} embodiment to date under current PPO policy in every evaluation world"}, project=args.project, entity=args.entity, group=args.group, name=args.run_name)
    evaluation = evaluator = manual_render_env = vector = None
    model = None
    optimizer = None
    eval_frames, rounds, rejections = 0, 0, 0

    class DeadlineCallback(BaseCallback):
        def _on_step(self):
            publish_snapshot()
            return not budget.expired

    try:
        evaluator = AsyncEvaluator(log, args, method)
        evaluation, base = make_baseline_vec(args.env_id, args.num_envs, device=args.device, seed=args.seed,
            control_mode="native", rejection_reward=args.rejection_reward)
        if method == "ppo_llm" and args.llm_gateway == "manual":
            eval_seed = args.seed + 3_000_003
            manual_render_env, _ = make_baseline_vec(args.env_id, args.evaluation_worlds,
                device=args.device, seed=eval_seed, is_eval_env=True, render_mode="rgb_array",
                control_mode="native", rejection_reward=args.rejection_reward)
        vector = SB3GymBatch(GymBatch(np.stack([base] * args.num_envs), env=evaluation))
        model = TimedPPO("MlpPolicy", vector, **PPO_SETTINGS, seed=args.seed, device=args.device, budget=budget)
        training_designs = np.repeat(base[None], args.num_envs, axis=0)
        best_design = base.copy()
        best_search_score = None

        def snapshot():
            data = io.BytesIO()
            model.save(data)
            return {"ppo": data.getvalue(), "design": best_design.copy(),
                    "embodiment_conditioned": True}

        def publish_snapshot(force=False):
            frames = vector.batch.frames + eval_frames
            evaluator.publish(snapshot, force=force, ppo_frames=vector.batch.frames,
                search_frames=eval_frames, frames=frames,
                completed_ppo_iterations=model.completed_ppo_iterations,
                **{"train/reward": vector.training_reward, "train/fps": frames / max(budget.elapsed, 1e-9)})

        publish_snapshot(force=True)
        model.progress_hook = publish_snapshot

        if method == "ppo_llm":
            optimizer = LLMSampler(evaluation, base, args, log.output)
            log.log(**optimizer.metrics())
        else:
            install_numpy_scalar_compat()
            codec = evaluation.design_codec
            binary_design = bool(codec.discrete.all()) and np.all(codec.low == 0) and np.all(codec.high == 1)
            if binary_design:
                # A binary (MultiBinary) layout is searched better by an ordered-discrete
                # parametrization than by a continuous array NGOpt has to round: NGOpt can
                # pick discrete-appropriate sub-optimizers, and TransitionChoice mutates in
                # {0,1} directly. (CMA/TD3 stay continuous -- discrete parametrization breaks
                # CMA's covariance model; a neural actor emits continuous logits.)
                parameter = ng.p.TransitionChoice(range(2), repetitions=int(base.shape[0]))
                parameter.value = tuple(int(round(float(b))) for b in np.asarray(base).reshape(-1))
            else:
                parameter = ng.p.Array(init=base.astype(float)).set_bounds(codec.low, codec.high)
            parameter.random_state.seed(args.seed)
            optimizer = ng.optimizers.NGOpt(parametrization=parameter, budget=1000000000,
                num_workers=args.ngopt_candidates)
            optimizer.enable_pickling()

        reset_training_env = True

        while not budget.expired:
            if reset_training_env:
                vector.batch.designs[:] = torch.as_tensor(training_designs, device=args.device)
                model._last_obs = vector.reset()
                model._last_episode_starts = np.ones(evaluation.num_envs, dtype=bool)
                reset_training_env = False
            try:
                for _ in range(args.ppo_iterations):
                    model.learn(256 * evaluation.num_envs, reset_num_timesteps=False, callback=DeadlineCallback())
                    publish_snapshot()
                    if budget.expired:
                        break
            except TrainingDeadlineReached:
                pass  # Save the last completed minibatch, then stop below.
            model.save(log.output / "policy")
            np.save(log.output / "embodiments.npy", training_designs)
            if budget.expired:
                break
            # Every candidate within a round uses one frozen controller.  A
            # second concurrent MuJoCo-Warp simulator on the same GPU can
            # mutually starve with the live simulator, so bounded NGOpt rounds
            # reuse this environment between PPO updates.
            if method == "ppo_llm":
                optimizer.search_state.update(completed_ppo_iterations=model.completed_ppo_iterations, best_observed_design=best_design.tolist(), best_observed_reward=best_search_score)
                if args.llm_gateway == "manual":
                    from ppo_llm_rlgo.manual_train import render_boundary
                    sheet, paths = render_boundary(manual_render_env, base, optimizer.templates, log.output, optimizer.round, eval_seed)
                    optimizer.search_state.update(render_contact_sheet=sheet, render_images=paths, render_note="Attach the contact sheet to a multimodal LLM; labels show the base at round zero or exact prior-round templates at later rounds. These are fresh-pose robot renders.")
                    log.log(**{"manual/render_contact_sheet_path": sheet, "manual/round": optimizer.round})
                    if log.run is not None:
                        import wandb
                        log.run.log({"manual/robot_contact_sheet": wandb.Image(sheet), "manual/round": optimizer.round})
            try:
                candidate_count = (args.ngopt_candidates
                    if method == "ppo_ngopt" else args.num_envs)
                candidates = ask_candidate_batch(optimizer, candidate_count, budget,
                    progress=publish_snapshot)
            except Exception as error:
                from ppo_llm_rlgo.gateway import ManualStop
                if method == "ppo_llm" and isinstance(error, ManualStop):
                    break
                raise
            if len(candidates) != candidate_count:
                break
            candidate_designs = np.stack([candidate.value for candidate in candidates])
            evaluation_designs = repeat_designs(candidate_designs, args.num_envs)
            observation, rejected, _ = evaluation.reset(
                seed=args.seed, embodiment=evaluation_designs)
            result = EvaluationAccumulator(evaluation)
            result.reject(rejected, evaluation.rejection_reward)
            if method == "ppo_llm":
                from ppo_llm_rlgo.context import EpisodeDiagnostics
                diagnostics = EpisodeDiagnostics(evaluation)
            search_steps = 0
            while not result.complete and not budget.expired:
                action = deterministic_policy_actions(
                    model,
                    condition_observation(
                        evaluation, observation, evaluation_designs),
                    evaluation.single_action_space)
                observation, reward, terminated, truncated, _ = evaluation.step(action)
                if method == "ppo_llm":
                    diagnostics.add(result.finished)
                result.add(reward, terminated, truncated)
                search_steps += 1
                if search_steps % 100 == 0:
                    log.log(**{"search/status": "running", "search/round": rounds,
                        "search/steps": search_steps,
                        "search/candidates": candidate_count})
                    publish_snapshot()
            eval_frames += result.frames
            if not result.complete:
                break
            if method == "ppo_ngopt":
                repeats = args.num_envs // candidate_count
                if repeats * candidate_count != args.num_envs:
                    raise ValueError(
                        "num-envs must be divisible by ngopt-candidates")
                candidate_returns = result.returns.reshape(
                    repeats, candidate_count).mean(dim=0).cpu().numpy()
                candidate_rejected = rejected.reshape(
                    repeats, candidate_count).any(dim=0).cpu().numpy()
            else:
                candidate_returns = evaluation.collapse_returns(result.returns).cpu().numpy() if hasattr(evaluation, "collapse_returns") else result.returns.cpu().numpy()
                candidate_rejected = evaluation.collapse_rejections(rejected).cpu().numpy() if hasattr(evaluation, "collapse_rejections") else rejected.cpu().numpy()
                optimizer.set_diagnostics(candidates, diagnostics.values(result.lengths))
            scores, batch_best_score, candidate_design = tell_candidate_batch(optimizer, candidates,
                candidate_returns, candidate_rejected, evaluation.rejection_reward)
            batch_rejections = int(candidate_rejected.sum())
            rejections += batch_rejections
            if candidate_design is not None and (best_search_score is None or batch_best_score > best_search_score):
                best_search_score, best_design = batch_best_score, candidate_design
            publish_snapshot(force=True)
            searched_designs = replace_rejected_designs(
                candidate_designs, candidate_rejected, best_design)
            training_designs = repeat_designs(searched_designs, args.num_envs)
            ppo_frames = vector.batch.frames
            frames = ppo_frames + eval_frames
            metrics = {"round": rounds, "score": float(scores.mean()), "best_score": best_search_score,
                "candidates": len(candidates), "embodiment_rejections": batch_rejections,
                "embodiment_rejection_rate": batch_rejections / max(len(candidates), 1),
                "rejections": rejections, "frames": frames, "ppo_frames": ppo_frames,
                "search_frames": eval_frames, "fps": frames / max(budget.elapsed, 1e-9),
                "train/reward": vector.training_reward}
            log.log(**{key: value for key, value in metrics.items() if value is not None})
            if method == "ppo_llm":
                log.log(**optimizer.metrics())
            publish_snapshot()
            np.save(log.output / "embodiments.npy", training_designs)
            reset_training_env = True
            rounds += 1
        evaluator.close()
        optimizer.dump(log.output / "optimizer.pkl")
        if method == "ppo_llm":
            log.log(**optimizer.metrics())
        if not (log.output / "best_checkpoint.zip").exists():
            model.save(log.output / "best_checkpoint")
            np.save(log.output / "best_embodiment.npy", best_design)
        log.finish("complete", rounds=rounds, rejections=rejections, frames=vector.batch.frames + eval_frames, ppo_frames=vector.batch.frames, search_frames=eval_frames, completed_ppo_iterations=model.completed_ppo_iterations, best_eval_reward=evaluator.best_eval_reward)
    except Exception:
        if evaluator is not None:
            evaluator.close(raise_errors=False)
        if method == "ppo_llm" and optimizer is not None:
            log.log(**optimizer.metrics())
        log.finish("failed", error=traceback.format_exc())
        raise
    finally:
        if method == "ppo_llm" and optimizer is not None:
            optimizer.close()
        if vector is not None:
            vector.close()
        if evaluation is not None and (vector is None or evaluation is not vector.batch.env):
            evaluation.close()
        if manual_render_env is not None:
            manual_render_env.close()
        if evaluator is not None:
            evaluator.close(raise_errors=False)


if __name__ == "__main__":
    main()
