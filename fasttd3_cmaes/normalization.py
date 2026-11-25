"""Task-only observation normalization and snapshot loading for FastTD3 + CMA-ES."""

import torch

from fast_td3.codesign.evaluation import load_policy as _load_fasttd3_policy
from fast_td3.codesign.inference import RolloutInference
from fast_td3.fast_td3 import Actor
from fast_td3.fast_td3_utils import EmpiricalNormalization


class PrefixEmpiricalNormalization(EmpiricalNormalization):
    """Normalize only the leading `obs_dim` task-observation features.

    Trailing features (e.g. already-bounded [-1, 1] embodiment encodings) pass
    through unchanged, preventing 1/eps amplification spikes when designs are
    constant during warmup and then perturbed by CMA-ES.
    """

    def __init__(self, obs_dim: int, device, eps: float = 1e-2, until=None):
        super().__init__(obs_dim, device, eps=eps, until=until)
        self.obs_dim = int(obs_dim)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        head = super().forward(x[:, : self.obs_dim], center=center, update=update)
        return torch.cat((head, x[:, self.obs_dim :]), dim=-1)


def load_policy(path, device, compile=True):
    """Load a FastTD3 + CMA-ES policy snapshot, supporting prefix observation normalization."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hp = payload["hp"]
    norm_state = payload.get("normalization")
    if (
        norm_state is None
        or not hp.get("obs_normalization", True)
        or int(norm_state["_mean"].shape[-1]) >= int(payload["state_dim"])
    ):
        return _load_fasttd3_policy(path, device, compile=compile)

    actor = Actor(
        payload["state_dim"],
        payload["action_dim"],
        payload["num_envs"],
        hp["init_scale"],
        hp["actor_hidden_dim"],
        std_min=hp["std_min"],
        std_max=hp["std_max"],
        device=device,
    )
    actor.load_state_dict(payload["actor"])
    actor.eval()
    norm_dim = int(norm_state["_mean"].shape[-1])
    normalization = PrefixEmpiricalNormalization(norm_dim, device)
    normalization.load_state_dict(norm_state)
    normalization.eval()
    return payload, RolloutInference(actor, normalization, compile=compile, amp=payload["amp"])
