"""Microgrid capacities using each preset's load-scaled bounds."""

from .base import BaseCollector


class MicrogridCollector(BaseCollector):
    presets = ("Microgrid-Random", "Microgrid-Forecast", "Microgrid-Campus",
               "Microgrid-OffGridSingle", "Microgrid-OffGridCampus")

    def sample_embodiment(self):
        return self.sample_native_space()

    def sampler_metadata(self):
        return {"distribution": "independent uniform native capacity bounds"}
