"""Warehouse layouts spanning the feasible shelf-count range."""

import numpy as np

from .base import BaseCollector


class WarehouseCollector(BaseCollector):
    presets = ("Warehouse-Small", "Warehouse-Medium", "Warehouse-Congested")

    def configure(self):
        from envs.robot_warehouse.embodiment import grid_size
        self.cfg = self.env.cfg
        height, width = grid_size(self.cfg)
        self.min_shelves = self.cfg.num_shelves
        self.max_shelves = min(self.space.shape[0], height * width - self.cfg.num_agents)
        if not 0 <= self.min_shelves <= self.max_shelves:
            raise ValueError("No feasible shelf count for this warehouse")

    def sample_embodiment(self):
        count = int(self.rng.integers(self.min_shelves, self.max_shelves + 1))
        design = np.zeros(self.space.shape, dtype=self.space.dtype)
        design[self.rng.choice(design.size, count, replace=False)] = 1
        return design

    def sampler_metadata(self):
        return {"distribution": "uniform feasible shelf count, then uniform subset of eligible cells",
                "minimum_shelves": self.min_shelves, "maximum_shelves": self.max_shelves,
                "quality_filter": None}
