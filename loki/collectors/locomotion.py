"""Native offset-vector corpora for the MuJoCo locomotion co-design tasks."""

from .base import BaseCollector


class LocomotionCollector(BaseCollector):
    presets = (
        "HalfCheetah-v4",
        "Swimmer-v4",
        "Hopper-v4",
        "Walker2d-v4",
        "Ant-v4",
        "Humanoid-v4",
        "HumanoidStandup-v4",
    )

    def make_env(self):
        from codesign_gym import make_vec

        kwargs = {"device": self.device, "seed": self.seed, **self.env_kwargs}
        return make_vec(self.env_id, 1, is_eval_env=True, **kwargs)[0]

    def sample_embodiment(self):
        return self.sample_native_space()

    def sampler_metadata(self):
        return {"distribution": "independent uniform native normalized-offset bounds"}
