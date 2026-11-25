"""Compiled FastTD3 rollout inference with explicit normalization updates."""

import torch


class RolloutInference:
    def __init__(self, actor, normalization, *, compile=True, amp=True):
        self.actor, self.normalization = actor, normalization
        self.device_type = next(actor.parameters()).device.type
        self.amp = amp and self.device_type == 'cuda'
        self.normalize = (lambda state, update=True: state) if normalization is None else normalization.forward

        def policy(observation, dones=None, deterministic=False):
            # The caller supplies dones only when a design reset is needed.
            # This removes explore()'s tensor-dependent branch without changing
            # its per-episode noise resampling or its random draw shapes.
            if dones is not None:
                scales = torch.rand(actor.n_envs, 1, device=observation.device) * (actor.std_max - actor.std_min) + actor.std_min
                actor.noise_scales.copy_(torch.where(dones[:, None], scales, actor.noise_scales))
            action = actor(observation)
            return action if deterministic else action + torch.randn_like(action) * actor.noise_scales

        self.policy = policy
        if compile:
            if normalization is not None:
                self.normalize = torch.compile(self.normalize, mode=None)
            self.policy = torch.compile(self.policy, mode=None)

    @torch.no_grad()
    def __call__(self, state, dones=None, *, deterministic=False, update=True):
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.amp):
            observation = self.normalize(state, update=update)
            return self.policy(observation, dones, deterministic).float()
