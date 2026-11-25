"""Native offset-vector corpus for the single-line truck unloading task."""

from .base import BaseCollector


class TruckUnloadCollector(BaseCollector):
    presets = ("TruckUnload-Single",)

    def make_env(self):
        from codesign_gym import make_vec

        kwargs = {"device": self.device, "seed": self.seed, **self.env_kwargs}
        return make_vec(self.env_id, 1, is_eval_env=True, **kwargs)[0]

    def sample_embodiment(self):
        return self.sample_native_space()

    def sampler_metadata(self):
        return {
            "distribution": "independent uniform native normalized-offset bounds",
            "max_offset_m": float(self.env.max_offset),
        }
