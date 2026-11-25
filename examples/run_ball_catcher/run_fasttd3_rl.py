import os
import sys
import argparse

# Performance optimizations for PyTorch
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import random
import time
import numpy as np
import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tensordict import TensorDict, from_module

from fast_td3 import Actor, Critic
from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)

# Import your custom environment and utilities
from ball_catcher.ball_catcher_env import BallCatcherParallelEnv
from hyperparameters.loader import load_hyperparameters

torch.set_float32_matmul_precision("high")

HP = load_hyperparameters("fasttd3_rl")

def parse_args():
    parser = argparse.ArgumentParser(description="FastTD3 standard configuration")

    # Environment & Experiment
    parser.add_argument("--env_name", type=str, default="BallCatcher")
    parser.add_argument("--exp_name", type=str, default="fasttd3_experiment")
    parser.add_argument("--seed", type=int, default=HP["seed"])
    parser.add_argument("--device_rank", type=int, default=HP["device_rank"])
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp_dtype", type=str, default=HP["amp_dtype"], choices=["fp16", "bf16"])
    parser.add_argument("--torch_deterministic", action="store_true", default=True)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default="fasttd3")

    # Hyperparameters (see hyperparameters/fasttd3_rl.yaml; mirrors
    # fasttd3_rl/fast_td3/hyperparams.py:BaseArgs)
    parser.add_argument("--num_envs", type=int, default=HP["num_envs"])
    parser.add_argument("--eval_num_envs", type=int, default=HP["eval_num_envs"])
    parser.add_argument("--total_timesteps", type=int, default=HP["total_timesteps"])
    parser.add_argument("--learning_starts", type=int, default=HP["learning_starts"])
    parser.add_argument("--buffer_size", type=int, default=HP["buffer_size"])
    parser.add_argument("--batch_size", type=int, default=HP["batch_size"])
    parser.add_argument("--num_updates", type=int, default=HP["num_updates"])
    parser.add_argument("--policy_frequency", type=int, default=HP["policy_frequency"])
    parser.add_argument("--num_steps", type=int, default=HP["num_steps"])
    parser.add_argument("--gamma", type=float, default=HP["gamma"])
    parser.add_argument("--tau", type=float, default=HP["tau"])
    parser.add_argument("--policy_noise", type=float, default=HP["policy_noise"])
    parser.add_argument("--noise_clip", type=float, default=HP["noise_clip"])

    # Architecture
    parser.add_argument("--actor_hidden_dim", type=int, default=HP["actor_hidden_dim"])
    parser.add_argument("--critic_hidden_dim", type=int, default=HP["critic_hidden_dim"])
    parser.add_argument("--init_scale", type=float, default=HP["init_scale"])
    parser.add_argument("--std_min", type=float, default=HP["std_min"])
    parser.add_argument("--std_max", type=float, default=HP["std_max"])
    parser.add_argument("--sim_type", type=str, default=HP["sim_type"])
    parser.add_argument("--sim_dimension", type=int, default=HP["sim_dimension"])
    parser.add_argument("--actor_seq_len", type=int, default=HP["actor_seq_len"])
    parser.add_argument("--critic_seq_len", type=int, default=HP["critic_seq_len"])
    parser.add_argument("--num_atoms", type=int, default=HP["num_atoms"])
    parser.add_argument("--v_min", type=float, default=HP["v_min"])
    parser.add_argument("--v_max", type=float, default=HP["v_max"])

    # Optimization
    parser.add_argument("--actor_learning_rate", type=float, default=HP["actor_learning_rate"])
    parser.add_argument("--critic_learning_rate", type=float, default=HP["critic_learning_rate"])
    parser.add_argument("--actor_learning_rate_end", type=float, default=HP["actor_learning_rate_end"])
    parser.add_argument("--critic_learning_rate_end", type=float, default=HP["critic_learning_rate_end"])
    parser.add_argument("--weight_decay", type=float, default=HP["weight_decay"])
    parser.add_argument("--max_grad_norm", type=float, default=HP["max_grad_norm"])
    parser.add_argument("--use_grad_norm_clipping", action="store_true", default=HP["use_grad_norm_clipping"])
    parser.add_argument("--use_cdq", action="store_true", default=HP["use_cdq"])
    parser.add_argument("--disable_bootstrap", action="store_true", default=HP["disable_bootstrap"])

    # System & Environment Specifics
    parser.add_argument("--obs_normalization", action="store_true", default=HP["obs_normalization"])
    parser.add_argument("--reward_normalization", action="store_true", default=HP["reward_normalization"])
    parser.add_argument("--compile", action="store_true", default=HP["compile"])
    parser.add_argument("--compile_mode", type=str, default=HP["compile_mode"])
    parser.add_argument("--diagnostics", action="store_true", default=HP["diagnostics"])
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--measure_burnin", type=int, default=HP["measure_burnin"])
    parser.add_argument("--eval_interval", type=int, default=HP["eval_interval"])
    parser.add_argument("--save_interval", type=int, default=HP["save_interval"])
    parser.add_argument("--render_interval", type=int, default=HP["render_interval"])
    parser.add_argument("--render_max_steps", type=int, default=HP["render_max_steps"])
    
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


def _check_tensor_health(tensor, name, device):
    if tensor is None:
        return {}
    return {
        f"{name}_has_nan": torch.isnan(tensor).any().float(),
        f"{name}_has_inf": torch.isinf(tensor).any().float()
    }

def _get_model_health_metrics(model, prefix, device):
    has_nan_w = torch.tensor(0.0, device=device)
    has_inf_w = torch.tensor(0.0, device=device)
    has_nan_g = torch.tensor(0.0, device=device)
    has_inf_g = torch.tensor(0.0, device=device)
    
    for param in model.parameters():
        if torch.isnan(param).any(): has_nan_w.fill_(1.0)
        if torch.isinf(param).any(): has_inf_w.fill_(1.0)
        if param.grad is not None:
            if torch.isnan(param.grad).any(): has_nan_g.fill_(1.0)
            if torch.isinf(param.grad).any(): has_inf_g.fill_(1.0)
            
    return {
        f"{prefix}_weights_has_nan": has_nan_w,
        f"{prefix}_weights_has_inf": has_inf_w,
        f"{prefix}_grads_has_nan": has_nan_g,
        f"{prefix}_grads_has_inf": has_inf_g,
    }


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def main():
    args = parse_args()
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"

    # Setup AMP & Device
    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = "cuda" if amp_enabled else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    
    device = torch.device(f"cuda:{args.device_rank}" if amp_enabled else "cpu")
    
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.use_wandb:
        wandb.init(project=args.project, name=run_name, config=vars(args))

    # Initialize Environments
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
        force_mag= args.force_mag,
        force_every= args.force_every,
        force_hold= args.force_hold,
        force_viz_scale=args.force_viz_scale,
    )

    envs = BallCatcherParallelEnv(
        num_envs=args.num_envs, device=str(device), **env_kwargs)

    eval_envs = BallCatcherParallelEnv(
        num_envs=args.eval_num_envs, device=str(device), **env_kwargs)

    render_env = BallCatcherParallelEnv(
        num_envs=1, device=str(device), render_mode="rgb_array", debug=True,
        terminate_out_of_field=False, **env_kwargs)
    
    # Dimensions
    n_act = envs.action_space.shape[-1]
    n_obs = envs.observation_space.shape[-1]
    
    asymmetric_obs = getattr(envs, "asymmetric_obs", False)
    if asymmetric_obs:
        n_critic_obs = envs.num_privileged_obs if type(envs.num_privileged_obs) == int else envs.num_privileged_obs[0]
    else:
        n_critic_obs = n_obs

    action_low, action_high = -1.0, 1.0

    # Normalizers
    obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device) if args.obs_normalization else nn.Identity()
    critic_obs_normalizer = EmpiricalNormalization(shape=n_critic_obs, device=device) if args.obs_normalization else nn.Identity()
    
    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(gamma=args.gamma, device=device, g_max=min(abs(args.v_min), abs(args.v_max)))
    else:
        reward_normalizer = nn.Identity()

    # Define base callables strictly before compiling/nesting
    normalize_obs = obs_normalizer.forward
    normalize_critic_obs = critic_obs_normalizer.forward
    update_stats = reward_normalizer.update_stats if args.reward_normalization else None
    normalize_reward = reward_normalizer.forward if args.reward_normalization else None

    # Actor / Critic Instantiation
    actor = Actor(
        n_obs=n_obs, n_act=n_act, num_envs=args.num_envs, device=device,
        init_scale=args.init_scale, hidden_dim=args.actor_hidden_dim,
        std_min=args.std_min, std_max=args.std_max, sim_type=args.sim_type,
        sim_dimension=args.sim_dimension, seq_len=args.actor_seq_len
    )
    
    actor_detach = Actor(
        n_obs=n_obs, n_act=n_act, num_envs=args.num_envs, device=device,
        init_scale=args.init_scale, hidden_dim=args.actor_hidden_dim,
        std_min=args.std_min, std_max=args.std_max, sim_type=args.sim_type,
        sim_dimension=args.sim_dimension, seq_len=args.actor_seq_len
    )
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = Critic(
        n_obs=n_critic_obs, n_act=n_act, num_atoms=args.num_atoms, v_min=args.v_min,
        v_max=args.v_max, hidden_dim=args.critic_hidden_dim, device=device,
        sim_type=args.sim_type, sim_dimension=args.sim_dimension, seq_len=args.critic_seq_len
    )
    qnet_target = Critic(
        n_obs=n_critic_obs, n_act=n_act, num_atoms=args.num_atoms, v_min=args.v_min,
        v_max=args.v_max, hidden_dim=args.critic_hidden_dim, device=device,
        sim_type=args.sim_type, sim_dimension=args.sim_dimension, seq_len=args.critic_seq_len
    )
    qnet_target.load_state_dict(qnet.state_dict())

    # Optimizers
    q_optimizer = optim.AdamW(qnet.parameters(), lr=args.critic_learning_rate, weight_decay=args.weight_decay)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=args.actor_learning_rate, weight_decay=args.weight_decay)

    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(q_optimizer, T_max=args.total_timesteps, eta_min=args.critic_learning_rate_end)
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=args.total_timesteps, eta_min=args.actor_learning_rate_end)

    # Replay Buffer
    rb = SimpleReplayBuffer(
        n_env=args.num_envs, buffer_size=args.buffer_size, n_obs=n_obs, n_act=n_act,
        n_critic_obs=n_critic_obs, asymmetric_obs=asymmetric_obs,
        playground_mode=False, n_steps=args.num_steps, gamma=args.gamma, device=device,
    )

    # Core Training Functions
    def evaluate():
        episode_returns = torch.zeros(args.eval_num_envs, device=device)
        episode_lengths = torch.zeros(args.eval_num_envs, device=device)
        done_masks = torch.zeros(args.eval_num_envs, dtype=torch.bool, device=device)
        
        obs = eval_envs.reset()
        if isinstance(obs, tuple): obs = obs[0]
        
        max_steps = getattr(eval_envs, "max_episode_steps", 2000) // getattr(eval_envs, "phys_substeps", 5) 
        
        for _ in range(max_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = normalize_obs(obs)
                actions = actor(norm_obs)

            step_result = eval_envs.step(actions.float())
            next_obs, rewards, dones = step_result[0], step_result[1], step_result[2]
            
            episode_returns = torch.where(~done_masks, episode_returns + rewards, episode_returns)
            episode_lengths = torch.where(~done_masks, episode_lengths + 1, episode_lengths)
            done_masks = torch.logical_or(done_masks, dones.bool())
            
            if done_masks.all():
                break
            obs = next_obs
            
        return episode_returns.mean().item(), episode_lengths.mean().item()

    def render_with_rollout():
        obs = render_env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        renders = []
        phys_substeps = getattr(eval_envs, "phys_substeps", 5)
        max_steps = min(getattr(eval_envs, "max_episode_steps", 2000),
                        args.render_max_steps) // phys_substeps
        
        for i in range(max_steps):
            with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
                norm_obs = normalize_obs(obs)
                actions = actor(norm_obs)

            step_result = render_env.step(actions.float())
            next_obs, dones = step_result[0], step_result[2]
            
            if i % max(1, 10 // phys_substeps) == 0:
                frame = render_env.render()
                if frame is not None:
                    renders.append(frame)
            if dones.any():
                break
            obs = next_obs
            
        return renders

    def update_main(data, logs_dict):
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"] if asymmetric_obs else observations
            next_critic_observations = data["next"]["critic_observations"] if asymmetric_obs else next_observations
            
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            
            bootstrap = (~dones).float() if args.disable_bootstrap else (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions).mul(args.policy_noise).clamp(-args.noise_clip, args.noise_clip)
            next_state_actions = (actor(next_observations) + clipped_noise).clamp(action_low, action_high)
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_proj, qf2_proj = qnet_target.projection(next_critic_observations, next_state_actions, rewards, bootstrap, discount)
                qf1_val = qnet_target.get_value(qf1_proj)
                qf2_val = qnet_target.get_value(qf2_proj)
                
                if args.use_cdq:
                    qf_next_target_dist = torch.where(qf1_val.unsqueeze(1) < qf2_val.unsqueeze(1), qf1_proj, qf2_proj)
                    qf1_target_dist, qf2_target_dist = qf_next_target_dist, qf_next_target_dist
                else:
                    qf1_target_dist, qf2_target_dist = qf1_proj, qf2_proj

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(qf1_target_dist * F.log_softmax(qf1, dim=1), dim=1).mean()
            qf2_loss = -torch.sum(qf2_target_dist * F.log_softmax(qf2, dim=1), dim=1).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)
        
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm) if args.use_grad_norm_clipping else torch.tensor(0.0)

        if args.diagnostics:
            for k, v in _check_tensor_health(qf_loss, "qf_loss", device).items():
                logs_dict[k] = v
            for k, v in _get_model_health_metrics(qnet, "critic", device).items():
                logs_dict[k] = v
        
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_val.max().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            critic_observations = data["critic_observations"] if asymmetric_obs else data["observations"]
            
            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            
            qf_value = torch.minimum(qf1_value, qf2_value) if args.use_cdq else (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm) if args.use_grad_norm_clipping else torch.tensor(0.0)
        
        scaler.step(actor_optimizer)
        scaler.update()
        
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau):
        for s, t in zip(src.parameters(), tgt.parameters()):
            t.data.mul_(1.0 - tau).add_(s.data, alpha=tau)

    # Compilation
    if args.compile:
        update_main = torch.compile(update_main, mode=args.compile_mode)
        update_pol = torch.compile(update_pol, mode=args.compile_mode)
        policy = torch.compile(policy)
        normalize_obs = torch.compile(normalize_obs)
        normalize_critic_obs = torch.compile(normalize_critic_obs)
        if args.reward_normalization:
            update_stats = torch.compile(update_stats)
            normalize_reward = torch.compile(normalize_reward)

    # Initialize State
    obs_reset_out = envs.reset()

    if asymmetric_obs:
        obs, critic_obs = obs_reset_out
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = obs_reset_out[0] if isinstance(obs_reset_out, tuple) else obs_reset_out

    global_step = 0
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state_dict"])
        obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state"])
        qnet.load_state_dict(ckpt["qnet_state_dict"])
        qnet_target.load_state_dict(ckpt["qnet_target_state_dict"])
        global_step = ckpt.get("global_step", 0)

    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    dones = None

    # Main Loop
    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        
        if start_time is None and global_step >= args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        step_out = envs.step(actions.float())
        
        # Support both 4 and 5 element gym returns (step, reward, done, [truncation], info)
        next_obs, rewards, dones = step_out[0], step_out[1], step_out[2]
        infos = step_out[-1]
        
        # Robust truncation handling depending on Env interface
        if len(step_out) == 5:
            truncations = step_out[3]
        else:
            truncations = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))

        if args.reward_normalization:
            update_stats(rewards, dones.float())

        # Support Env providing 'raw' obs inside infos (common for vectorized episodic resets)
        true_next_obs = torch.where(dones[:, None] > 0, infos.get("raw_obs", next_obs), next_obs)
        
        transition = TensorDict({
            "observations": obs,
            "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
            "next": {
                "observations": true_next_obs,
                "rewards": torch.as_tensor(rewards, device=device, dtype=torch.float),
                "truncations": torch.as_tensor(truncations, device=device, dtype=torch.long),
                "dones": dones.long(),
            },
        }, batch_size=(args.num_envs,), device=device)

        if asymmetric_obs:
            next_critic_obs = infos.get("critic_obs", next_obs)
            true_next_critic_obs = torch.where(dones[:, None] > 0, infos.get("raw_critic_obs", next_critic_obs), next_critic_obs)
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
            critic_obs = next_critic_obs

        rb.extend(transition)
        obs = next_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(data["next"]["observations"])
                
                if asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(data["critic_observations"])
                    data["next"]["critic_observations"] = normalize_critic_obs(data["next"]["critic_observations"])
                
                raw_rewards = data["next"]["rewards"]

                if args.reward_normalization:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)
                else:
                    data["next"]["rewards"] = raw_rewards

                logs_dict = update_main(data, logs_dict)
                if (args.num_updates > 1 and i % args.policy_frequency == 1) or (args.num_updates == 1 and global_step % args.policy_frequency == 0):
                    logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps")
                
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict.get("actor_loss", torch.tensor(0.0)).mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "env_rewards": rewards.mean(),
                    }

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        eval_ret, eval_len = evaluate()
                        logs["eval_avg_return"] = eval_ret
                        logs["eval_avg_length"] = eval_len

                    if args.use_wandb and args.render_interval > 0 and global_step % args.render_interval == 0:
                        renders = render_with_rollout()
                        if renders:
                            # Convert array of images to (T, C, H, W) for WandB
                            logs["render_video"] = wandb.Video(np.array(renders).transpose(0, 3, 1, 2), fps=30, format="gif")

                if args.use_wandb:
                    wandb.log({"speed": speed, "frame": global_step * args.num_envs, **logs}, step=global_step)
                elif "eval_avg_return" in logs:
                    print(f"step {global_step} eval {logs['eval_avg_return']:.3f}", flush=True)

            if args.save_interval > 0 and global_step > 0 and global_step % args.save_interval == 0:
                os.makedirs("models", exist_ok=True)
                save_params(global_step, actor, qnet, qnet_target, obs_normalizer, critic_obs_normalizer, args, f"models/{run_name}_{global_step}.pt")

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

    save_params(global_step, actor, qnet, qnet_target, obs_normalizer, critic_obs_normalizer, args, f"models/{run_name}_final.pt")

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()