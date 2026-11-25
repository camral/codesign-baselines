"""Task descriptions from live configuration and first-episode diagnostics."""

import numpy as np
import torch


def task_context(env, parameters):
    native = getattr(env, "env", env)
    context = {"rejection_reward": getattr(env, "rejection_reward", None), "parameters": parameters}
    if not hasattr(native, "pitch_from"):
        return context
    names = ("pitch_from", "pitch_to", "pitch_to_radius", "pitch_vel", "pitch_vel_spread", "pitch_arrival_time", "pitch_arrival_spread", "pitch_target_half_widths", "max_episode_steps", "goal_radius", "drop_radius", "goal_dwell_steps", "at_goal_rew", "grounded_goal_rew", "prog_gain", "terminate_out_of_field", "ball_size", "ball_mass", "max_offset", "glove_max_offset")
    context["configuration"] = {name: np.asarray(getattr(native, name)).tolist() for name in names if hasattr(native, name)}
    context["launch"] = ("Target independently uniform in the world-coordinate box centered at pitch_to with pitch_target_half_widths; arrival time uniform around pitch_arrival_time +/- pitch_arrival_spread. Velocity solves the ballistic trajectory under gravity." if native.pitch_arrival_time > 0 else "Aim at a random point on the sphere centered at pitch_to with radius pitch_to_radius; initial speed is pitch_vel + Uniform(-pitch_vel_spread, pitch_vel_spread). Gravity bends the trajectory.")
    context["reward"] = "Distance is ball-center to glove goal. Each of the first goal_dwell_steps accumulated steps inside goal_radius pays at_goal_rew; these steps need not be consecutive. After dwell, retaining within drop_radius and all four feet grounded pays grounded_goal_rew and terminates. Leaving drop_radius after dwell terminates as a drop. Progress reward is prog_gain * positive improvement in best distance / initial distance. Diverged steps pay zero. Out-of-field termination follows configuration; max_episode_steps truncates."
    context["units"] = "Positions and radii in meters, velocity in meters/second, arrival time in seconds. Embodiment coordinates are normalized offsets."
    for body, adr, num in zip(native._segs["body"], native._segs["axis_adr"], native._segs["axis_num"]):
        for slot in native._segs["axis_act"][adr:adr+num]:
            parameter = parameters[int(slot)]
            name = native.m.body(int(body)).name
            bodies = parameter.setdefault("bodies", [])
            if name not in bodies:
                bodies.append(name)
            parameter["meaning"] = "Link deformation along its configured local axis; shared-slot bodies deform together."
            parameter["meters_per_unit"] = native.max_offset
    if getattr(native, "n_glove_params", 0):
        for j, name in enumerate(("glove width", "glove height", "glove wall height")):
            parameters[native.n_link_slots+j].update(meaning=name, meters_per_unit=native.glove_max_offset)
    if getattr(native, "disable_legs", False):
        parameters[-1]["meaning"] = "Leg-chain switch: positive keeps the configured leg chains; zero or negative removes them."
    return context


class EpisodeDiagnostics:
    signals = {"goal_entry_rate": "last_at_goal", "catch_rate": "last_caught", "drop_rate": "last_dropped", "grounded_success_rate": "last_held_at_goal", "ever_grounded_rate": "last_grounded", "ball_left_field_rate": "last_left_field", "time_limit_rate": "last_time_limit", "divergence_rate": "last_diverged"}

    def __init__(self, env):
        self.env = env
        self.native = getattr(env, "env", env)
        self.events = {}

    def add(self, finished):
        active = ~finished
        for name, attribute in self.signals.items():
            value = getattr(self.native, attribute, None)
            if value is not None:
                event = torch.as_tensor(value, device=finished.device, dtype=torch.bool).reshape(-1)
                self.events.setdefault(name, torch.zeros_like(finished)).logical_or_(event & active)

    def values(self, lengths):
        values = {**self.events, "episode_length_mean": lengths}
        result = {}
        for name, value in values.items():
            value = value.float()
            if hasattr(self.env, "collapse_returns"):
                value = self.env.collapse_returns(value)
            result[name] = value.cpu().numpy().reshape(-1)
        return result
