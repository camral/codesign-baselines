import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# The sibling `cmaes_evolutionary` package (repo_root/cmaes_evolutionary) ships
# as a plain directory, not a pip-installed package. Make it importable
# regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cmaes_evolutionary.main import CMAESConfig, run_cmaes

# Import your custom environment and utilities
from ball_catcher.ball_catcher_env import BallCatcherParallelEnv
from hyperparameters.loader import load_hyperparameters

HP = load_hyperparameters("cmaes_evolutionary")


def parse_args():
    parser = argparse.ArgumentParser(description="CMA-ES co-design configuration")

    # Environment & Experiment
    parser.add_argument("--env_name", type=str, default="BallCatcher")
    parser.add_argument("--exp_name", type=str, default="cmaes_evolutionary_experiment")
    parser.add_argument("--seed", type=int, default=HP["seed"])
    parser.add_argument("--device", type=str, default=HP["device"])
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default="cmaes_evolutionary")

    # CMA-ES hyperparameters (see hyperparameters/cmaes_evolutionary.yaml;
    # mirrors cmaes_evolutionary.main.CMAESConfig)
    parser.add_argument("--budget", type=int, default=HP["budget"])
    parser.add_argument("--evaluation_repeats", type=int, default=HP["evaluation_repeats"])
    parser.add_argument("--scale", type=float, default=HP["scale"])
    parser.add_argument("--population_size", type=int, default=HP["population_size"])
    parser.add_argument("--diagonal", action="store_true", default=HP["diagonal"])
    parser.add_argument("--elitist", action="store_true", default=HP["elitist"])
    parser.add_argument("--policy_lower_bound", type=float, default=HP["policy_lower_bound"])
    parser.add_argument("--policy_upper_bound", type=float, default=HP["policy_upper_bound"])
    parser.add_argument("--max_rollout_steps", type=int, default=0,
                         help="0 uses max_episode_steps // phys_substeps")
    parser.add_argument("--output", type=str, default="")

    # Ball Catcher environment specifics
    parser.add_argument("--ant_xml", type=str, default="ant_reacher/dog.xml")
    parser.add_argument("--arm_xml", type=str, default="ant_reacher/ant_arm.xml")
    parser.add_argument("--scene_xml", type=str, default="_common/scene_ant.xml")
    parser.add_argument("--init_key", type=str, default="init")
    parser.add_argument("--ball_size", type=float, default=0.05)
    parser.add_argument("--ball_mass", type=float, default=0.1)
    parser.add_argument("--max_offset", type=float, default=1.0)
    parser.add_argument("--max_return", type=float, default=10.0)
    parser.add_argument("--max_goal_return", type=float, default=9.9)
    parser.add_argument("--goal_dwell_steps", type=int, default=10)
    parser.add_argument("--goal_radius", type=float, default=0.1)
    parser.add_argument("--glove_width", type=float, default=0.2)
    parser.add_argument("--glove_height", type=float, default=0.2)
    parser.add_argument("--glove_wall_h", type=float, default=0.05)
    parser.add_argument("--glove_max_offset", type=float, default=0.05)
    parser.add_argument("--glove_goal_only", type=bool, default=False)
    parser.add_argument("--offset_glove", type=bool, default=False,
                         help="Include glove width/height/wall_h in the design search. The "
                              "checked-in glove assets currently fail BallCatcherParallelEnv's "
                              "own inertia consistency check when this is enabled.")
    parser.add_argument("--pitch_from", nargs=3, type=float, default=[0, 0, 1])
    parser.add_argument("--pitch_to", nargs=3, type=float, default=[0, 1, 0])
    parser.add_argument("--pitch_to_radius", type=float, default=0.5)
    parser.add_argument("--pitch_vel", type=float, default=5.0)
    parser.add_argument("--field_pos", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--field_size", nargs=3, type=float, default=[10, 10, 10])
    parser.add_argument("--phys_substeps", type=int, default=5)
    parser.add_argument("--max_rel_dist", type=float, default=10.0)
    parser.add_argument("--vel_scale", type=float, default=1.0)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--disable_legs", type=bool, default=False)
    parser.add_argument("--force_mag", type=float, default=0.0)
    parser.add_argument("--force_every", type=int, default=100)
    parser.add_argument("--force_hold", type=int, default=10)
    parser.add_argument("--force_viz_scale", type=float, default=0.1)

    # Use parse_known_args in case your env feeds extra config
    args, _ = parser.parse_known_args()
    return args


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    return requested


def make_evaluator(env: BallCatcherParallelEnv, obs_dim: int, act_dim: int, max_steps: int):
    """Design = the env's built-in per-world morphology offsets (arm link
    lengths, and glove width/height/wall_h when ``offset_glove`` is set).
    Policy = the weights and bias of a linear ``tanh(W @ obs + b)`` controller.
    """
    weight_size = act_dim * obs_dim

    def evaluate_design_policy(design: np.ndarray, policy: np.ndarray) -> float:
        weight = policy[:weight_size].reshape(act_dim, obs_dim)
        bias = policy[weight_size:]
        embodiment = torch.as_tensor(
            design, dtype=torch.float32, device=env.device).reshape(1, -1)

        obs = env.reset()[0].detach().cpu().numpy()
        total_reward = 0.0
        for _ in range(max_steps):
            action = np.tanh(weight @ obs + bias)
            actions = torch.as_tensor(
                action, dtype=torch.float32, device=env.device).reshape(1, -1)
            next_obs, reward, terminated, truncated, _ = env.step(
                actions, embodiment=embodiment)
            total_reward += float(reward[0].item())
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                break
            obs = next_obs[0].detach().cpu().numpy()
        return total_reward

    return evaluate_design_policy


def main():
    args = parse_args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    device = _resolve_device(args.device)

    env_kwargs = dict(
        ant_xml=args.ant_xml,
        arm_xml=args.arm_xml,
        scene_xml=args.scene_xml,
        init_key=args.init_key,
        ball_size=args.ball_size,
        ball_mass=args.ball_mass,
        max_offset=args.max_offset,
        max_return=args.max_return,
        max_goal_return=args.max_goal_return,
        goal_dwell_steps=args.goal_dwell_steps,
        goal_radius=args.goal_radius,
        glove_width=args.glove_width,
        glove_height=args.glove_height,
        glove_wall_h=args.glove_wall_h,
        glove_max_offset=args.glove_max_offset,
        glove_goal_only=args.glove_goal_only,
        offset_glove=args.offset_glove,
        pitch_from=args.pitch_from,
        pitch_to=args.pitch_to,
        pitch_to_radius=args.pitch_to_radius,
        pitch_vel=args.pitch_vel,
        field_pos=args.field_pos,
        field_size=args.field_size,
        phys_substeps=args.phys_substeps,
        max_rel_dist=args.max_rel_dist,
        vel_scale=args.vel_scale,
        timestep=args.timestep,
        disable_legs=args.disable_legs,
        force_mag=args.force_mag,
        force_every=args.force_every,
        force_hold=args.force_hold,
        force_viz_scale=args.force_viz_scale,
        seed=args.seed,
    )

    env = BallCatcherParallelEnv(num_envs=1, device=device, **env_kwargs)
    obs_dim, act_dim = env.obs_dim(), env.act_dim()
    max_steps = args.max_rollout_steps or (env.max_episode_steps // env.phys_substeps)
    design_lower, design_upper = env.offset_bounds
    policy_dimension = act_dim * obs_dim + act_dim

    config = CMAESConfig(
        design_dimension=env.n_offsets,
        policy_dimension=policy_dimension,
        budget=args.budget,
        evaluation_repeats=args.evaluation_repeats,
        design_lower_bound=design_lower,
        design_upper_bound=design_upper,
        design_initial=np.zeros(env.n_offsets),
        policy_lower_bound=args.policy_lower_bound,
        policy_upper_bound=args.policy_upper_bound,
        policy_initial=np.zeros(policy_dimension),
        seed=args.seed,
        scale=args.scale,
        population_size=args.population_size,
        diagonal=args.diagonal,
        elitist=args.elitist,
        maximize=True,
    )

    wandb_run = None
    progress_callback = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.project, name=run_name, config=vars(args))

        def progress_callback(candidates_evaluated, score, best_score):
            wandb_run.log(
                {"candidate_score": score, "best_score": best_score},
                step=candidates_evaluated,
            )

    evaluator = make_evaluator(env, obs_dim, act_dim, max_steps)
    result = run_cmaes(evaluator, config, device=device, progress_callback=progress_callback)

    payload = {"env_name": args.env_name, "device": device, **result.to_dict()}
    if wandb_run is not None:
        wandb_run.log({"score": result.score, "loss": result.loss}, step=args.budget)
        wandb_run.finish()

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
