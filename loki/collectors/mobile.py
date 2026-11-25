"""Network station counts, positions and powers with canonical inactive rows."""

from .base import BaseCollector


class MobileCollector(BaseCollector):
    presets = ("Network-Urban", "Network-Suburban", "Network-Motorway")

    def configure(self):
        self.cfg = self.env.cfg

    def sample_embodiment(self):
        design = self.sample_native_space()
        design["num_bs"] = int(self.rng.integers(self.cfg.min_bs, self.cfg.max_bs + 1))
        return self.canonicalize(design)

    def canonicalize(self, design):
        design["stations"][int(design["num_bs"]):] = (0, 0, self.cfg.tx_min)
        return design

    def sampler_metadata(self):
        return {"distribution": "uniform feasible station count; uniform positions and dBm powers",
                "inactive_rows": [0, 0, self.cfg.tx_min],
                "power_and_separation_constraints": "native rejection"}
