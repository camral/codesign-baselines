"""Compare native NeroGrasp perturbation amplitudes with fixed proposal budgets."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .collectors import CorpusReader, make_collector
from .collectors.storage import json_value


def save_meshes(path, meshes):
    arrays = {}
    for name, mesh in meshes.items():
        vertices, faces = (mesh.vertices, mesh.faces) if hasattr(mesh, "vertices") else mesh
        arrays[name + "__vertices"] = vertices
        arrays[name + "__faces"] = faces
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=int, default=1000)
    parser.add_argument("--preset", default="NeroGrasp-SetScrew")
    parser.add_argument("--amplitudes", type=float, nargs="+", default=(0.1, 0.5, 1.0, 2.0))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for mm in args.amplitudes:
        root = args.output / f"{mm:g}mm"
        root.mkdir(exist_ok=True)
        start = time.monotonic()
        with make_collector(args.preset, seed=0, device="cuda:0",
                            env_kwargs={"sample_perturbation_m": mm / 1000}) as collector:
            save_meshes(root / "template.npz", collector.base_embodiment)
            result = collector.collect(
                root / "corpus", count=args.proposals, max_attempts=args.proposals,
                max_stalled_attempts=args.proposals + 1, audit_every=0, shard_size=25,
                resume=(root / "corpus/manifest.json").exists(),
                progress=lambda value: print(json.dumps({"amplitude_mm": mm, **value}), flush=True))
            examples = []
            for design_id, design in CorpusReader(root / "corpus").iter_embodiments():
                save_meshes(root / f"accepted_{design_id}.npz", design)
                processed = collector.env._prepare_direct_meshes([design])[0]
                save_meshes(root / f"processed_{design_id}.npz", processed)
                examples.append({"design_id": design_id,
                                 "reset_passed": collector.audit_reset(design, design_id)})
                if len(examples) == 3:
                    break
            base = collector.base_embodiment
            geometry = {name: {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                              "bbox_mm": (np.ptp(mesh.vertices, axis=0) * 1000).tolist(),
                              "mounting_vertices": collector.space.mounting_vertex_counts[name]}
                        for name, mesh in base.items()}
            row = {"amplitude_mm": mm, "result": asdict(result), "examples": examples,
                   "geometry": geometry, "config": json_value(collector.env.codesign_config),
                   "wall_seconds": time.monotonic() - start}
            results.append(row)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print("PILOT_RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
