"""SoftWalker per-cell designs: uniform native samples repaired into valid bodies."""

from common.soft_walker_design import repair

from .base import BaseCollector


class SoftWalkerCollector(BaseCollector):
    presets = ("SoftWalker-Beam3D",)

    def make_env(self):
        from codesign_gym import make_vec
        # SoftWalker's physics has no randomness and its builder rejects seed=.
        kwargs = {"device": self.device, **self.env_kwargs}
        return make_vec(self.env_id, 1, is_eval_env=True, **kwargs)[0]

    def sample_embodiment(self):
        # A uniform draw is almost never a valid body (disconnected / corner-joined cells, fractional kind and
        # channel), so it goes through the same repair every search method's designs go through.
        return repair(self.sample_native_space(), self.env.grid, self.env.channels).astype(self.space.dtype)

    def sampler_metadata(self):
        return {"distribution": "independent uniform native per-cell bounds, then common.soft_walker_design.repair",
                "grid": list(self.env.grid), "channels": int(self.env.channels)}
