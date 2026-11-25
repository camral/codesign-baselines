"""Repair SoftWalker per-cell designs into structurally valid ones for search methods.

Codesign Gym validates SoftWalker designs structurally only (integer kind/channel, a fibre for every directional
actuator, a non-empty face-connected body without corner/edge-only junctions), so a design drawn uniformly from the
embodiment Box is almost never valid. Search methods (CMA-ES, NGOpt, FastTD3's design actor, LOKI's sampler) propose
unconstrained vectors; this repair maps each one to a valid design deterministically:

1. round ``kind``/``channel`` to integers and give every zero fibre the base body's direction ``(1, 0, ..., 0)``;
2. keep the largest face-connected component of solid cells (ties: the first in C order);
3. fill every 2^dim block whose local pattern is not face-connected (clipped to the grid) until none remain.

Step 3 only adds cells and each filled block contains a solid cell, so the body stays connected; it terminates
because the full block is valid. Removed cells get occupancy 0, added cells occupancy 1, everything else is kept.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from envs.soft_walker import embodiment as E


def repair(vector, grid, channels):
    """A structurally valid copy of the flat per-cell ``vector`` (float64, same shape)."""
    grid = tuple(int(n) for n in grid)
    dim, cells = len(grid), int(np.prod(grid))
    low, high = E.bounds(dim, cells, channels)
    table = np.clip(np.asarray(vector, dtype=np.float64).reshape(-1), low, high).reshape(cells, E.width(dim))
    s = E._slices(dim)
    for name in ("kind", "channel"):
        table[:, s[name]] = np.rint(table[:, s[name]])
    direction = table[:, s["direction"]]
    zero = np.linalg.norm(direction, axis=1) == 0
    direction[zero] = 0.0
    direction[zero, 0] = 1.0
    table[:, s["direction"]] = direction

    occupancy = table[:, s["occupancy"].start]
    solid = (occupancy >= E.OCCUPANCY_THRESHOLD).reshape(grid)
    if solid.any():
        labels, count = ndimage.label(solid, structure=ndimage.generate_binary_structure(dim, 1))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        keep = labels == int(np.argmax(sizes))
    else:
        keep = np.zeros(grid, bool)
        keep[(0,) * dim] = True
    while True:
        bad = np.argwhere(~E._local_table(dim)[E._local_pattern(keep)])
        if not len(bad):
            break
        # Block b of the padded pattern covers grid cells b-1 .. b along every axis.
        for block in bad:
            keep[tuple(slice(max(b - 1, 0), min(b + 1, n)) for b, n in zip(block, grid))] = True
    keep = keep.reshape(-1)
    occupancy[~keep & (occupancy >= E.OCCUPANCY_THRESHOLD)] = 0.0
    occupancy[keep & (occupancy < E.OCCUPANCY_THRESHOLD)] = 1.0
    table[:, s["occupancy"].start] = occupancy
    return table.reshape(-1)
