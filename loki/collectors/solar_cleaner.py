"""SolarCleaner per-link offsets and optional leg-presence switch."""

import numpy as np

from .base import BaseCollector


class SolarCleanerCollector(BaseCollector):
    presets = ("SolarCleaner-Reach", "SolarCleaner-Stability", "SolarCleaner-Traverse")

    def make_env(self):
        from codesign_gym import make_vec
        kwargs = {"device": self.device, "seed": self.seed, **self.env_kwargs}
        return make_vec(self.env_id, 1, is_eval_env=True, **kwargs)[0]

    def sample_embodiment(self):
        design = self.sample_native_space()
        if getattr(self.env, "disable_legs", False):
            start = self.env.n_link_slots
            design[start:] = np.where(design[start:] > 0, 1, -1)
        return design

    def sampler_metadata(self):
        return {"distribution": "independent uniform native offset bounds; leg switch canonicalized to sign"}
