"""Interactive MuJoCo viewer helpers for a batched env.

Simplified from ``drafts/gpu_env/utils/mjview_utils.py``. The viewer shows one
world at a time (``env.viz_env_id``); the physics for every world still runs on
the GPU. Keys: Space/P pause-resume, ``]``/``[`` switch which world is shown.

The cage-wireframe / structural-affinity debug layers of the original are gone;
a generic ``env._add_debug_visualizations`` hook remains.
"""

import time

import mujoco
import mujoco.viewer
import mujoco_warp as mjw


def make_viewer_key_callback(pause_state: dict):
    def _on_key(keycode):
        if keycode in (ord(" "), ord("P"), ord("p")):
            pause_state["paused"] = not pause_state["paused"]
            print("[viewer] " + ("paused" if pause_state["paused"] else "resumed"),
                  flush=True)
        elif keycode == ord("]"):
            pause_state["viz_env_delta"] = pause_state.get("viz_env_delta", 0) + 1
        elif keycode == ord("["):
            pause_state["viz_env_delta"] = pause_state.get("viz_env_delta", 0) - 1
    return _on_key


def sync_viewer(env, viewer) -> None:
    """Copy the selected world off the GPU and repaint the viewer.

    ``mujoco_warp`` only mirrors state (qpos/qvel) into the CPU MjData; world
    frame geom transforms are not copied, so we must re-run kinematics before
    syncing or body-attached geoms render at the origin.
    """
    with env._wp_scope():
        mjw.get_data_into(env._render_d, env._render_m, env.gpu_data, env.viz_env_id)
    mujoco.mj_kinematics(env._render_m, env._render_d)
    if env.debug:
        viewer.user_scn.ngeom = 0
        env._add_debug_visualizations(viewer.user_scn, env.viz_env_id)
    viewer.sync()


def wait_while_paused(viewer, pause_state: dict, env) -> None:
    while viewer and pause_state["paused"] and viewer.is_running():
        if pause_state.get("viz_env_delta"):
            return  # let the caller apply the env switch, then re-enter
        sync_viewer(env, viewer)
        time.sleep(0.05)


def refresh_viewer(env, viewer, current_render_model, pause_state):
    """Launch the passive viewer, or relaunch it if the shown world's model
    changed (each variant is a distinct compiled ``MjModel``)."""
    target = env.variants[env.viz_env_id]
    first_launch = viewer is None
    if viewer is None or current_render_model is not target:
        if viewer is not None:
            viewer.close()
            del viewer
            time.sleep(0.3)
        if env._render_m is not target:
            env._render_d = mujoco.MjData(target)
            env._render_m = target
        viewer = mujoco.viewer.launch_passive(
            target, env._render_d,
            key_callback=make_viewer_key_callback(pause_state))
        current_render_model = target
        env._setup_camera(viewer.cam)
        if hasattr(mujoco.mjtVisFlag, "mjVIS_REFLECTION"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_REFLECTION] = 0
        if first_launch:
            print("[viewer] Space/P pause | ] next world | [ prev world", flush=True)
    return viewer, current_render_model


def apply_pending_env_switch(env, viewer, current_render_model, pause_state):
    """Apply a queued ``[``/``]`` world switch. Returns (viewer, model, switched)."""
    delta = pause_state.pop("viz_env_delta", 0)
    if not delta or viewer is None or not viewer.is_running():
        return viewer, current_render_model, False
    env.viz_env_id = (env.viz_env_id + delta) % env.num_envs
    print(f"[viewer] showing world {env.viz_env_id}", flush=True)
    viewer, current_render_model = refresh_viewer(
        env, viewer, current_render_model, pause_state)
    sync_viewer(env, viewer)
    return viewer, current_render_model, True
