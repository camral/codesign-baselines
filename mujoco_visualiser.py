#!/usr/bin/env python3
"""Interactive MuJoCo viewer for trained codesign-baselines checkpoints.

Supports all MuJoCo environments in codesign-gym (classic locomotion and
Warp/CPU co-design tasks) and all baseline checkpoint formats:
  - fasttd3_cmaes  (.pt)
  - fasttd3 joint  (.pt)
  - ppo_ngopt / ppo_llm (.zip + sibling best_embodiment.npy, or .pt snapshot)
  - cmaes          (.npz or .pt snapshot)
  - loki           (.pt snapshot)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
for _extra in (REPO_ROOT, REPO_ROOT / "fasttd3_rl"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from common.codesign_adapter import make_baseline_vec
from common.embodiment_conditioning import condition_observation

MUJOCO_ENV_PREFIXES = (
    "HalfCheetah-",
    "Swimmer-",
    "Hopper-",
    "Walker2d-",
    "Ant-",
    "Humanoid-",
    "HumanoidStandup-",
    "BallCatcher-",
    "SolarCleaner-",
    "RotateInHand-",
    "HandPickUp-",
    "TruckUnload-",
    "NeroGrasp",
)


def is_mujoco_env(env_id: str) -> bool:
    """Return True if env_id belongs to a MuJoCo environment family."""
    return bool(env_id) and env_id.startswith(MUJOCO_ENV_PREFIXES)


def _load_sibling_config(checkpoint_path: Path) -> dict[str, Any]:
    for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
        cfg_path = directory / "config.json"
        if cfg_path.is_file():
            try:
                return json.loads(cfg_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def inspect_checkpoint(checkpoint_path: Path, *, env_id: str | None = None) -> dict[str, Any]:
    """Load checkpoint metadata and detect method + MuJoCo env_id."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    sibling = _load_sibling_config(path)
    suffix = path.suffix.lower()
    payload: Any = None
    method: str

    if suffix in (".pt", ".pth"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")
        if "token_schema" in payload and "ppo" in payload:
            method = "loki"
        elif "ppo" in payload:
            method = "ppo"
        elif "policy" in payload and "action_size" in payload:
            method = "cmaes"
        elif payload.get("method") == "fasttd3_cmaes" or ("actor" in payload and "design" in payload):
            method = "fasttd3_cmaes"
        elif "actor" in payload:
            method = "fasttd3"
        else:
            raise ValueError(f"Unrecognised .pt checkpoint format in {path} (keys={sorted(payload.keys())})")
    elif suffix == ".zip":
        method = "ppo"
        payload = {}
    elif suffix == ".npz":
        method = "cmaes"
        with np.load(path, allow_pickle=False) as data:
            payload = {k: data[k] for k in data.files}
    else:
        raise ValueError(f"Unsupported checkpoint extension {suffix!r}; expected .pt, .zip, or .npz")

    resolved_env_id = env_id or (payload.get("env_id") if isinstance(payload, dict) else None) or sibling.get("env_id")
    if not resolved_env_id:
        raise ValueError(
            f"Could not determine env_id from {path} or sibling config.json; pass --env-id explicitly."
        )
    if not is_mujoco_env(resolved_env_id):
        raise ValueError(
            f"Environment {resolved_env_id!r} is not a MuJoCo environment. "
            f"mujoco_visualiser.py only supports MuJoCo environments ({', '.join(MUJOCO_ENV_PREFIXES)})."
        )

    control_mode = (
        (payload.get("control_mode") if isinstance(payload, dict) else None)
        or sibling.get("control_mode")
        or "logits"
    )
    mesh_bound_m = (
        payload.get("mesh_bound_m") if isinstance(payload, dict) and payload.get("mesh_bound_m") is not None
        else sibling.get("mesh_bound_m")
    )
    use_cbd = bool(
        payload.get("use_cbd", sibling.get("use_cbd", False))
        if isinstance(payload, dict)
        else sibling.get("use_cbd", False)
    )
    embodiment_conditioned = bool(
        payload.get("embodiment_conditioned", sibling.get("embodiment_conditioned", False))
        if isinstance(payload, dict)
        else sibling.get("embodiment_conditioned", False)
    )
    return {
        "path": path,
        "method": method,
        "env_id": str(resolved_env_id),
        "control_mode": str(control_mode),
        "mesh_bound_m": mesh_bound_m,
        "use_cbd": use_cbd,
        "embodiment_conditioned": embodiment_conditioned,
        "payload": payload,
    }


def _resolve_design(spec: dict[str, Any], env, *, design_path: Path | None, use_base_design: bool) -> np.ndarray:
    if use_base_design:
        return np.asarray(env.base_embodiment, dtype=np.float32).copy()
    if design_path is not None:
        return np.asarray(np.load(Path(design_path).expanduser()), dtype=np.float32)
    payload = spec["payload"]
    if isinstance(payload, dict) and payload.get("design") is not None:
        return np.asarray(payload["design"], dtype=np.float32)
    if spec["path"] is not None:
        sibling_design = spec["path"].parent / "best_embodiment.npy"
        if sibling_design.is_file():
            return np.asarray(np.load(sibling_design), dtype=np.float32)
    return np.asarray(env.base_embodiment, dtype=np.float32).copy()


def _build_controller(spec: dict[str, Any], env, device: str, *, design_path: Path | None, use_base_design: bool):
    method = spec["method"]
    path = spec["path"]
    payload = spec["payload"]

    if method == "fasttd3":
        from fast_td3.codesign.evaluation import load_policy
        _, policy = load_policy(path, device, compile=False)
        return "joint", None, policy

    design = _resolve_design(spec, env, design_path=design_path, use_base_design=use_base_design)

    if method == "zero":
        zero_action = torch.zeros(
            (env.num_envs, *env.single_action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        return "fixed", design, lambda obs: zero_action

    if method == "fasttd3_cmaes":
        from fasttd3_cmaes.normalization import load_policy
        _, policy = load_policy(path, device, compile=False)

        def act(obs):
            conditioned = condition_observation(env, obs, design)
            return env.project_control(policy(conditioned, deterministic=True, update=False).clamp(-1, 1))

        return "fixed", design, act

    if method == "loki":
        from loki.single_cluster import TokenConditioner
        from ppo_ngopt_rlgo.timed_ppo import TimedPPO

        policy = TimedPPO.load(io.BytesIO(payload["ppo"]), device=device)
        conditioner = TokenConditioner(env, payload["token_schema"])
        norm = payload["normalizer"]
        mean = np.asarray(norm["mean"], dtype=np.float32)
        var = np.asarray(norm["var"], dtype=np.float32)
        eps = float(norm["epsilon"])
        clip = float(norm["clip_obs"])

        def act(obs):
            conditioned = conditioner(obs, design).cpu().numpy()
            normalized = np.clip((conditioned - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            return policy.predict(normalized, deterministic=True)[0]

        return "fixed", design, act

    if method == "ppo":
        from stable_baselines3 import PPO

        source = io.BytesIO(payload["ppo"]) if isinstance(payload, dict) and "ppo" in payload else path
        policy = PPO.load(source, device=device)
        conditioned_flag = spec["embodiment_conditioned"]

        def act(obs):
            inp = condition_observation(env, obs, design) if conditioned_flag else obs
            return policy.predict(inp.cpu().numpy(), deterministic=True)[0]

        return "fixed", design, act

    if method == "cmaes":
        from cmaes_evolutionary.gym_train import control

        weights = torch.as_tensor(payload["policy"], dtype=torch.float32, device=device)
        action_size = int(np.asarray(payload["action_size"]).item())
        conditioned_flag = spec["embodiment_conditioned"]

        def act(obs):
            inp = condition_observation(env, obs, design) if conditioned_flag else obs
            return control(inp, weights, action_size)

        return "fixed", design, act

    raise ValueError(f"Unsupported method: {method!r}")


def _infer_step_dt(env) -> float:
    raw = getattr(env, "env", env)
    for attr in ("dt", "step_dt", "control_dt"):
        value = getattr(raw, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    model = getattr(raw, "m", None) or getattr(raw, "model", None)
    if model is not None and hasattr(model, "opt"):
        substeps = getattr(raw, "phys_substeps", getattr(raw, "frame_skip", getattr(raw, "n_substeps", 1)))
        dt = float(model.opt.timestep) * max(1, int(substeps))
        if dt > 0:
            return dt
    fps = getattr(raw, "metadata", {}).get("render_fps")
    if isinstance(fps, (int, float)) and fps > 0:
        return 1.0 / float(fps)
    return 0.02


def _viewer_is_closed(env, saw_viewer: bool) -> tuple[bool, bool]:
    raw = getattr(env, "env", env)
    viewer = getattr(raw, "_human_viewer", None)
    if viewer is not None:
        saw_viewer = True
        if not viewer.is_running():
            return True, saw_viewer
    elif saw_viewer:
        return True, saw_viewer
    return False, saw_viewer


@contextlib.contextmanager
def _viewer_key_callback(on_key):
    """Attach a key_callback to any mujoco.viewer.launch_passive call in scope."""
    import mujoco.viewer

    orig_launch = mujoco.viewer.launch_passive

    def _wrapped_launch(*args, **kwargs):
        existing = kwargs.get("key_callback")

        def _dispatch(keycode: int) -> None:
            on_key(keycode)
            if existing is not None:
                existing(keycode)

        kwargs["key_callback"] = _dispatch
        return orig_launch(*args, **kwargs)

    mujoco.viewer.launch_passive = _wrapped_launch
    try:
        yield
    finally:
        mujoco.viewer.launch_passive = orig_launch


def visualise(
    checkpoint_path: str | Path | None = None,
    *,
    env_id: str | None = None,
    device: str = "auto",
    seed: int = 0,
    episodes: int = 0,
    max_steps: int = 0,
    speed: float = 1.0,
    design_path: str | Path | None = None,
    use_base_design: bool = False,
    save_video: str | Path | None = None,
    headless: bool = False,
    env_idx: int = 0,
    env_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load a checkpoint (or run zero actions if omitted) in a MuJoCo viewer."""
    if checkpoint_path is not None:
        spec = inspect_checkpoint(Path(checkpoint_path), env_id=env_id)
    else:
        if not env_id:
            raise ValueError("Must provide either checkpoint_path or env_id.")
        if not is_mujoco_env(env_id):
            raise ValueError(
                f"Environment {env_id!r} is not a MuJoCo environment. "
                f"mujoco_visualiser.py only supports MuJoCo environments ({', '.join(MUJOCO_ENV_PREFIXES)})."
            )
        spec = {
            "path": None,
            "method": "zero",
            "env_id": str(env_id),
            "control_mode": "logits",
            "mesh_bound_m": None,
            "use_cbd": False,
            "embodiment_conditioned": False,
            "payload": {},
        }
    resolved_device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    render_mode = None if headless else ("rgb_array" if save_video is not None else "human")

    if (
        render_mode == "human"
        and spec["env_id"].startswith("NeroGrasp")
        and not spec["env_id"].startswith(("NeroGraspAlt", "NeroGrasp-Tools"))
    ):
        raise RuntimeError(
            f"{spec['env_id']} does not support render_mode='human'; pass --save-video out.mp4 instead."
        )

    frames: list[np.ndarray] = []
    results: list[dict[str, Any]] = []
    paused = threading.Event()

    def _on_key(keycode: int) -> None:
        if keycode in (ord("p"), ord("P")):
            if paused.is_set():
                paused.clear()
                print("Resumed")
            else:
                paused.set()
                print("Paused (press 'p' in viewer to resume)")

    viewer_ctx = _viewer_key_callback(_on_key) if render_mode == "human" else contextlib.nullcontext()

    with viewer_ctx:
        env, _ = make_baseline_vec(
            spec["env_id"],
            num_envs=1,
            device=resolved_device,
            seed=seed,
            is_eval_env=True,
            render_mode=render_mode,
            control_mode=spec["control_mode"],
            mesh_bound_m=spec["mesh_bound_m"],
            **(env_kwargs or {}),
        )
        if spec["method"] == "fasttd3_cmaes" and spec["use_cbd"]:
            from fasttd3_cmaes.cbdsa_codec import apply_cbdsa_codec
            apply_cbdsa_codec(env, bound_m=spec["mesh_bound_m"])
        try:
            mode, design, controller = _build_controller(
                spec,
                env,
                resolved_device,
                design_path=Path(design_path) if design_path else None,
                use_base_design=use_base_design,
            )
            step_dt = _infer_step_dt(env)
            auto_human = bool(getattr(env.env, "_render_human_automatically", False))
            saw_viewer = False
            episode_idx = 0

            if spec["method"] == "zero":
                print(
                    f"Running zero-action viewer for {spec['env_id']} "
                    f"(device={resolved_device}, render_mode={render_mode}, dt={step_dt:.4f}s)"
                )
            else:
                print(
                    f"Loaded {spec['method']} checkpoint for {spec['env_id']} "
                    f"(device={resolved_device}, render_mode={render_mode}, dt={step_dt:.4f}s)"
                )
            if render_mode == "human":
                print("Press 'p' in the viewer window to pause/resume.")
            if mode == "fixed" and design is not None:
                design_label = (
                    "base_embodiment"
                    if (use_base_design or (spec["path"] is None and design_path is None))
                    else (str(design_path) if design_path else "checkpoint")
                )
                print(f"Design source: {design_label} (dim={design.size}, norm={float(np.linalg.norm(design)):.4f})")

            def _wait_if_paused() -> bool:
                nonlocal saw_viewer
                while render_mode == "human" and paused.is_set():
                    closed, saw_viewer = _viewer_is_closed(env, saw_viewer)
                    if closed:
                        return False
                    viewer = getattr(getattr(env, "env", env), "_human_viewer", None)
                    if viewer is not None:
                        viewer.sync()
                    time.sleep(0.02)
                return True

            def _render_step():
                nonlocal saw_viewer
                if render_mode == "human":
                    if not auto_human:
                        env.render(env_idx=env_idx)
                    closed, saw_viewer = _viewer_is_closed(env, saw_viewer)
                    return not closed
                if render_mode == "rgb_array":
                    frame = env.render(env_idx=env_idx)
                    if frame is not None:
                        frames.append(np.asarray(frame))
                return True

            with torch.inference_mode():
                while episodes <= 0 or episode_idx < episodes:
                    ep_seed = seed + episode_idx
                    ep_return = 0.0
                    ep_steps = 0
                    ep_success = False

                    if mode == "joint":
                        from fast_td3.codesign.joint import JointRollout

                        rollout = JointRollout(env, seed=ep_seed)
                        design_out = controller(rollout.state(), deterministic=True, update=False).clamp(-1, 1)
                        _, rejected, *_ = rollout.select_design(design_out)
                        if bool(rejected.any()):
                            print(f"Episode {episode_idx + 1}: design rejected, fell back to base embodiment")
                    else:
                        obs, rejected, _ = env.reset(seed=ep_seed, embodiment=design)
                        if bool(torch.as_tensor(rejected).any()):
                            raise ValueError("Checkpoint design was rejected by the environment.")

                    if not _render_step():
                        break

                    while True:
                        if not _wait_if_paused():
                            episode_idx = episodes if episodes > 0 else episode_idx + 1
                            results.append({"episode": episode_idx, "return": ep_return, "length": ep_steps, "success": ep_success})
                            return results

                        t0 = time.monotonic()
                        if mode == "joint":
                            action_out = controller(rollout.state(), deterministic=True, update=False).clamp(-1, 1)
                            _, _, reward, _, terminated, truncated, _ = rollout.step(action_out)
                        else:
                            action = controller(obs)
                            obs, reward, terminated, truncated, _ = env.step(action)

                        ep_return += float(reward.reshape(-1)[env_idx].item())
                        ep_steps += 1
                        if hasattr(env, "last_success"):
                            ep_success = ep_success or bool(env.last_success.reshape(-1)[env_idx].item())
                        elif hasattr(env, "last_held_at_goal"):
                            ep_success = ep_success or bool(env.last_held_at_goal.reshape(-1)[env_idx].item())

                        if not _render_step():
                            episode_idx = episodes if episodes > 0 else episode_idx + 1
                            results.append({"episode": episode_idx, "return": ep_return, "length": ep_steps, "success": ep_success})
                            return results

                        if render_mode == "human" and speed > 0:
                            remaining = (step_dt / speed) - (time.monotonic() - t0)
                            if remaining > 0:
                                time.sleep(remaining)

                        done = bool((terminated | truncated).reshape(-1)[env_idx].item())
                        if done or (max_steps > 0 and ep_steps >= max_steps):
                            break

                    episode_idx += 1
                    summary = {"episode": episode_idx, "return": ep_return, "length": ep_steps, "success": ep_success}
                    results.append(summary)
                    print(
                        f"Episode {episode_idx}: return={ep_return:.3f}, length={ep_steps}, "
                        f"success={ep_success}"
                    )

            if save_video is not None and frames:
                import imageio.v2 as imageio

                out_path = Path(save_video).expanduser()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fps = max(1, int(round(1.0 / step_dt)))
                imageio.mimsave(out_path, frames, fps=fps)
                print(f"Saved {len(frames)} frames to {out_path} ({fps} fps)")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            env.close()

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise a trained codesign-baselines checkpoint in a MuJoCo viewer."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Optional path to checkpoint (.pt, .zip, or .npz); omit to run with zero actions",
    )
    parser.add_argument(
        "--env-id",
        "--env_name",
        "--env-name",
        dest="env_id",
        default=None,
        help="MuJoCo env ID (required when no checkpoint is given; otherwise auto-detected from checkpoint or config.json)",
    )
    parser.add_argument("--device", default="auto", help="Torch/Warp device ('auto', 'cuda:0', 'cpu')")
    parser.add_argument("--seed", type=int, default=0, help="Initial episode seed")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to play (0 = loop until viewer window is closed or Ctrl-C)",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Optional cap on steps per episode (0 = full episode)")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for human viewer (e.g. 0.5 = half speed, 0 = unthrottled)",
    )
    parser.add_argument("--design", default=None, help="Optional .npy file to override the checkpoint's embodiment")
    parser.add_argument(
        "--base-design",
        action="store_true",
        help="Visualise the policy on the undeformed base_embodiment instead of the evolved design",
    )
    parser.add_argument("--save-video", default=None, help="Render offscreen rgb_array video to this .mp4 path")
    parser.add_argument("--headless", action="store_true", help="Run without opening a viewer or rendering frames")
    parser.add_argument("--env-idx", type=int, default=0, help="Lane index to display/log for multi-world envs")
    parser.add_argument(
        "--env-kwargs",
        default=None,
        help='Optional JSON dict of extra kwargs passed to make_baseline_vec (e.g. \'{"debug": true}\')',
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None and not args.env_id:
        parser.error("Provide a checkpoint path or --env_name / --env-id.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_kwargs = json.loads(args.env_kwargs) if args.env_kwargs else None
    visualise(
        args.checkpoint,
        env_id=args.env_id,
        device=args.device,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        speed=args.speed,
        design_path=args.design,
        use_base_design=args.base_design,
        save_video=args.save_video,
        headless=args.headless,
        env_idx=args.env_idx,
        env_kwargs=env_kwargs,
    )


if __name__ == "__main__":
    main()
