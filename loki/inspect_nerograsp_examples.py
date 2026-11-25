"""Record native rejection details for the plotted NeroGrasp examples."""

import argparse
import json

import numpy as np

from .collectors import make_collector
from .nerograsp_pilot import save_meshes


def main():
    from pathlib import Path
    from envs.nerograsp.spaces import Mesh

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output / "visuals"
    rows = []
    with make_collector("NeroGrasp-SetScrew", seed=0, device="cuda:0") as collector:
        save_meshes(root / "template_processed.npz", collector.env._prepare_direct_meshes([collector.base_embodiment])[0])
        for mm in (0.1, 0.5, 1, 2):
            with np.load(root / f"{mm:g}mm_sample0.npz") as arrays:
                design = {name: Mesh(arrays[name + "__vertices"], arrays[name + "__faces"])
                          for name in collector.base_embodiment}
            _, meshes, rejected, records, reasons = collector.env._assess_embodiment([design])
            row = {"amplitude_mm": mm, "rejected": bool(rejected.any()), "reasons": reasons,
                   "dimensions_mm": {name: (np.ptp(mesh.vertices, axis=0) * 1000).tolist()
                                     for name, mesh in design.items()}}
            if meshes[0]:
                save_meshes(root / f"{mm:g}mm_sample0_assessed.npz", meshes[0])
            rows.append(row)
            print(json.dumps(row), flush=True)
        (root / "rejection_details.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
