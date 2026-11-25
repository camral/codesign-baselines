"""Executed joint actions for reset-only embodiment/control co-design."""

import torch


def project_action(output, embodiment, design_phase, env=None):
    output = output.clamp(-1, 1)
    width = embodiment.shape[-1]
    proposed_design = env.project_design(output[:, :width]) if env is not None and hasattr(env, "project_design") else output[:, :width]
    proposed_control = env.project_control(output[:, width:]) if env is not None and hasattr(env, "project_control") else output[:, width:]
    design = torch.where(design_phase[:, None], proposed_design, embodiment)
    control = torch.where(design_phase[:, None], torch.zeros_like(proposed_control), proposed_control)
    return torch.cat([design, control], dim=-1)


class JointRollout:
    """Separate zero-time design transitions from physical control transitions.

    Embodiment actions use normalized [-1,1] actor coordinates and are mapped
    into each environment's embodiment bounds only when passed to reset.
    """

    def __init__(self, env, seed=None):
        self.env = env
        self.design_dim = len(env.base_embodiment)
        self.low = torch.as_tensor(env.embodiment_space.low, device=env.device)
        self.high = torch.as_tensor(env.embodiment_space.high, device=env.device)
        base = torch.as_tensor(env.base_embodiment, device=env.device)
        self.base_design = torch.where(self.high > self.low, 2 * (base - self.low) / (self.high - self.low) - 1, torch.zeros_like(base))
        self.design = torch.zeros((env.num_envs, self.design_dim), device=env.device)
        self.observation, rejected, _ = env.reset(seed=seed)
        if bool(rejected.any()):
            raise ValueError("base embodiment rejected")
        self.design_phase = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def state(self):
        return torch.cat([self.observation, self.design, self.design_phase[:, None].float()], dim=-1)

    def select_design(self, output):
        selected = self.design_phase.clone()
        state = self.state()
        executed = project_action(output, self.design, selected, self.env)
        if self.env.worlds_per_design > 1:
            proposed = executed[:, :self.design_dim].reshape(
                self.env.num_designs, self.env.worlds_per_design, self.design_dim).mean(dim=1)
            executed[:, :self.design_dim] = self.env.expand_design_values(proposed)
        self.design[selected] = executed[selected, :self.design_dim]
        actual = self.low + (self.design + 1) * 0.5 * (self.high - self.low)
        candidate_actual = actual.reshape(
            self.env.num_designs, self.env.worlds_per_design, self.design_dim)[:, 0]
        self.observation, rejected, _ = self.env.reset(embodiment=candidate_actual, options={"reset_mask": selected})
        rejected = torch.as_tensor(rejected, dtype=torch.bool, device=self.env.device).reshape(-1)
        self.design[rejected] = self.base_design
        self.design_phase.zero_()
        return selected, rejected, state, executed, self.state()

    def step(self, output):
        if bool(self.design_phase.any()):
            raise RuntimeError("select episode designs before advancing physics")
        state = self.state()
        executed = project_action(output, self.design, self.design_phase, self.env)
        self.observation, reward, terminated, truncated, info = self.env.step(executed[:, self.design_dim:])
        # Preserve terminal control observations for timeout bootstrapping.
        next_state = self.state()
        done = (terminated | truncated).reshape(-1)
        grouped_done = self.env.collapse_rejections(done)
        self.design_phase.copy_(self.env.expand_design_values(grouped_done))
        return state, executed, reward, next_state, terminated, truncated, info
