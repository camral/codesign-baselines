"""LQ chain parameters with preset-specific placement and budget constraints."""

import numpy as np

from .base import BaseCollector


class LQCollector(BaseCollector):
    presets = ("LQStructure-LQR", "LQStructure-LQG", "LQStructure-QP")

    def configure(self):
        self.cfg = self.env.cfg
        self.actuator_cap = self.cfg.n
        if self.cfg.actuator_budget is not None:
            self.actuator_cap = min(self.actuator_cap, self.cfg.actuator_budget)
        if self.actuator_cap < 1:
            raise ValueError("No feasible actuator allocation")

    def _mask(self, maximum):
        mask = np.zeros(self.cfg.n, dtype=np.int8)
        count = int(self.rng.integers(1, maximum + 1))
        mask[self.rng.choice(self.cfg.n, count, replace=False)] = 1
        return mask

    def sample_embodiment(self):
        design = self.sample_native_space()
        design["actuators"] = (self._mask(self.actuator_cap) if self.cfg.design_actuators
                               else self.base_embodiment["actuators"].copy())
        design["sensors"] = (self._mask(self.cfg.n) if self.cfg.design_sensors
                             else self.base_embodiment["sensors"].copy())
        return design

    def sampler_metadata(self):
        return {"distribution": "uniform continuous bounds; uniform feasible placement count then subset",
                "frozen_placements": "native base embodiment", "mass_budget": "native rejection"}
