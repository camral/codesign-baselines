"""BallCatcher offset embodiments for all pitch presets."""

from .base import BaseCollector


class BallCatcherCollector(BaseCollector):
    presets = ("BallCatcher-Pitch", "BallCatcher-Arc", "BallCatcher-Fielding")

    def make_env(self):
        from codesign_gym import make_vec
        kwargs = {"device": self.device, "seed": self.seed, **self.env_kwargs}
        return make_vec(self.env_id, 1, is_eval_env=True, **kwargs)[0]

    def sample_embodiment(self):
        return self.sample_native_space()

    def sampler_metadata(self):
        return {"distribution": "independent uniform native offset bounds"}
