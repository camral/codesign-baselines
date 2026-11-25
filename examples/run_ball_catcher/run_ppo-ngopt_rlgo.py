import os
import sys
import argparse
from pathlib import Path

# The sibling `ppo_ngopt_rlgo` package (repo_root/ppo_ngopt_rlgo) ships as a
# plain directory, not a pip-installed package (see ppo_ngopt_rlgo/README.md,
# which runs it via `python -m ppo_ngopt_rlgo` from the repo root). Make it
# importable regardless of the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ppo_ngopt_rlgo.config import PPOConfig
from ppo_ngopt_rlgo.ppo import evaluate, train
from hyperparameters.loader import load_hyperparameters

HP = load_hyperparameters("ppo_ngopt_rlgo")


def parse_args():
    parser = argparse.ArgumentParser(description="PPO+NGOpt/rlgo (Stable-Baselines3 PPO) configuration")

    # Environment & Experiment
    parser.add_argument("--env_name", type=str, default="BallCatcher")
    parser.add_argument("--exp_name", type=str, default="ppo_ngopt_rlgo_experiment")
    parser.add_argument("--action", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--seed", type=int, default=HP["seed"])
    parser.add_argument("--device", type=str, default=HP["device"])
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default="ppo_ngopt_rlgo")

    # PPO hyperparameters (see hyperparameters/ppo_ngopt_rlgo.yaml;
    # mirrors ppo_ngopt_rlgo.config.PPOConfig)
    parser.add_argument("--policy", type=str, default=HP["policy"])
    parser.add_argument("--total_timesteps", type=int, default=HP["total_timesteps"])
    parser.add_argument("--learning_rate", type=float, default=HP["learning_rate"])
    parser.add_argument("--n_steps", type=int, default=HP["n_steps"])
    parser.add_argument("--batch_size", type=int, default=HP["batch_size"])
    parser.add_argument("--gamma", type=float, default=HP["gamma"])
    parser.add_argument("--evaluation_episodes", type=int, default=HP["evaluation_episodes"])
    parser.add_argument("--model_path", type=str, default="")

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
    parser.add_argument("--glove_goal_only", type=bool, default=False)
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


def main():
    args = parse_args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    # Ball Catcher takes these as constructor kwargs for a single (num_envs=1)
    # instance; see ball_catcher.gym_env.make_ball_catcher.
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
        glove_goal_only=args.glove_goal_only,
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
    )

    model_path = Path(args.model_path) if args.model_path else Path("models") / run_name

    config = PPOConfig(
        environment_factory="ball_catcher.gym_env:make_ball_catcher",
        environment_kwargs=env_kwargs,
        policy=args.policy,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        device=args.device,
        model_path=model_path,
        evaluation_episodes=args.evaluation_episodes,
    )

    wandb_run = None
    callback = None
    if args.use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project=args.project, name=run_name, config=vars(args), sync_tensorboard=True)
        if args.action == "train":
            callback = WandbCallback()

    record = train(config, callback=callback) if args.action == "train" else evaluate(config)

    if wandb_run is not None:
        wandb_run.log(record.to_dict())
        wandb_run.finish()

    print(record.to_json())


if __name__ == "__main__":
    main()
