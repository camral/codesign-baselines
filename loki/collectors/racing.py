"""F1TENTH vehicle parameters within each track preset's native bounds."""

from .base import BaseCollector


class RacingCollector(BaseCollector):
    presets = ("Racing-Spielberg", "Racing-Levine")

    def sample_embodiment(self):
        return self.sample_native_space()

    def sampler_metadata(self):
        return {"distribution": "independent uniform native vehicle-design bounds"}
