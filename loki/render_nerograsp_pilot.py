"""Render paired native proposals, without exaggerating mesh displacements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .collectors import make_collector
from .nerograsp_pilot import save_meshes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    panels = []
    metadata = []
    with make_collector("NeroGrasp-SetScrew", seed=0, device="cuda:0") as collector:
        base = collector.base_embodiment
        key = next(iter(base))
        count = collector.space.mounting_vertex_counts[key]
        panels.append(("Template", base, True))
        for mm in (0.1, 0.5, 1.0, 2.0):
            collector.space.sample_perturbation_m = mm / 1000
            collector.rng = np.random.default_rng(0)
            for sample in range(3):
                design = collector.sample_embodiment()
                valid, reason = collector.validate_embodiment(design)
                prefix = f"{mm:g}mm_sample{sample}"
                save_meshes(root / (prefix + ".npz"), design)
                row = {"amplitude_mm": mm, "sample": sample, "native_valid": valid,
                       "reason": reason, "meshes": {}}
                for name, mesh in design.items():
                    path = root / f"{prefix}_{name}.obj"
                    with path.open("w") as stream:
                        for vertex in mesh.vertices:
                            stream.write("v " + " ".join(map(str, vertex)) + "\n")
                        for face in mesh.faces:
                            stream.write("f " + " ".join(map(str, face + 1)) + "\n")
                    row["meshes"][name] = path.name
                if valid:
                    processed = collector.env._prepare_direct_meshes([design])[0]
                    save_meshes(root / (prefix + "_processed.npz"), processed)
                    if sample == 0:
                        row["processed_keys"] = list(processed)
                if sample == 0:
                    panels.append((f"±{mm:g} mm\n{'Accepted' if valid else 'Rejected'}", design, valid))
                metadata.append(row)
                (root / "examples.json").write_text(json.dumps(metadata, indent=2))
                print(json.dumps(row), flush=True)
        # Same limits and camera for every panel. Two views show thickness and profile.
        vertices = np.concatenate([design[key].vertices for _, design, _ in panels]) * 1000
        center = (vertices.max(0) + vertices.min(0)) / 2
        radius = np.ptp(vertices, axis=0).max() * 0.55
        fig = plt.figure(figsize=(18, 8), facecolor="#f5f6f8")
        for row, (elev, azim) in enumerate(((22, -55), (10, 35))):
            for col, (title, design, valid) in enumerate(panels):
                ax = fig.add_subplot(2, 5, row * 5 + col + 1, projection="3d")
                mesh = design[key]
                faces = mesh.faces
                colors = np.where(np.all(faces < count, axis=1)[:, None],
                                  np.array([[0.48, 0.54, 0.62, 1]]),
                                  np.array([[0.94, 0.61, 0.24, 1]]))
                poly = Poly3DCollection(mesh.vertices[faces] * 1000, facecolors=colors,
                                       edgecolors="#463e35", linewidths=0.3, shade=True,
                                       lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=40))
                ax.add_collection3d(poly)
                for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                    setter(center[dim] - radius, center[dim] + radius)
                ax.set_box_aspect((1, 1, 1))
                ax.view_init(elev=elev, azim=azim)
                ax.set_axis_off()
                ax.set_facecolor("#f5f6f8")
                if row == 0:
                    ax.set_title(title, fontsize=14, color="#243447" if valid else "#a72b20")
        fig.suptitle(f"NeroGrasp: native vertex-noise proposals · {key}", fontsize=20, y=0.99)
        fig.text(0.5, 0.04, "Same seed, camera and scale · True displacement (no magnification)\n"
                 "Orange: mutable surface · Grey: fixed mounting · Two views of the first proposal at each amplitude",
                 ha="center", fontsize=12)
        fig.subplots_adjust(left=0, right=1, top=0.88, bottom=0.1, wspace=0, hspace=0)
        fig.savefig(root / "comparison.png", dpi=180)
        fig.savefig(root / "comparison.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
