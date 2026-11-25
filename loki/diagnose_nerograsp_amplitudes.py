"""Measure NeroGrasp rejectors and render matched proposals by amplitude."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .collectors import make_collector
from .nerograsp_pilot import save_meshes


AMPLITUDES_MM = (0.2, 0.3, 0.4, 0.5, 1.0, 2.0)


def category(detail):
    text = detail.lower()
    if " volume " in text and " < " in text:
        return "min volume ratio"
    for phrase in ("min thickness", "volume ratio", "backplate area", "minimum bbox",
                   "degenerate mesh", "topology", "convex"):
        if phrase in text:
            return phrase
    return detail.split(":", 1)[0].strip()[:100]


def reason_events(reason):
    events = []
    for line in reason.splitlines():
        match = re.search(r"fingertip ([^:]+): (.*)", line)
        if match:
            events.append((match.group(1), category(match.group(2)), match.group(2)))
    if reason and not events:
        events.append(("design", category(reason), reason))
    return events


def write_obj(path, mesh):
    with path.open("w") as stream:
        for vertex in mesh.vertices:
            stream.write("v " + " ".join(map(str, vertex)) + "\n")
        for face in mesh.faces:
            stream.write("f " + " ".join(map(str, face + 1)) + "\n")


def render(root, base, examples, mounting_counts):
    keys = tuple(base)
    all_vertices = np.concatenate(
        [mesh.vertices for mesh in base.values()]
        + [mesh.vertices for design, _ in examples.values() for mesh in design.values()]
    ) * 1000
    center = (all_vertices.max(0) + all_vertices.min(0)) / 2
    radius = np.ptp(all_vertices, axis=0).max() * 0.54
    fig = plt.figure(figsize=(22, 7), facecolor="#f5f6f8")
    columns = (("Template", base, True),) + tuple(
        (f"±{mm:g} mm\n{'Accepted' if examples[mm][1] else 'Rejected'}",
         examples[mm][0], examples[mm][1]) for mm in AMPLITUDES_MM
    )
    for row, key in enumerate(keys):
        for col, (title, design, valid) in enumerate(columns):
            ax = fig.add_subplot(2, 7, row * 7 + col + 1, projection="3d")
            mesh = design[key]
            fixed = np.all(mesh.faces < mounting_counts[key], axis=1)
            colors = np.where(fixed[:, None], np.array([[0.48, 0.54, 0.62, 1]]),
                              np.array([[0.94, 0.61, 0.24, 1]]))
            ax.add_collection3d(Poly3DCollection(
                mesh.vertices[mesh.faces] * 1000, facecolors=colors,
                edgecolors="#463e35", linewidths=0.25, shade=True,
                lightsource=matplotlib.colors.LightSource(azdeg=315, altdeg=40)))
            for dim, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                setter(center[dim] - radius, center[dim] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=18, azim=-55)
            ax.set_axis_off()
            ax.set_facecolor("#f5f6f8")
            if row == 0:
                ax.set_title(title, fontsize=13, color="#243447" if valid else "#a72b20")
            if col == 0:
                ax.text2D(-0.08, 0.5, key.capitalize(), transform=ax.transAxes,
                          rotation=90, va="center", fontsize=14)
    fig.suptitle("NeroGrasp matched-seed fingertip proposals", fontsize=20, y=0.98)
    fig.text(0.5, 0.025,
             "Same random directions, camera and scale · no displacement magnification\n"
             "Orange: mutable surface · Grey: fixed mounting · validity applies to the two-fingertip design",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.10, wspace=0, hspace=0)
    fig.savefig(root / "fingertips.png", dpi=180)
    fig.savefig(root / "fingertips.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    plotted = {}
    with make_collector("NeroGrasp-SetScrew", seed=0, device="cuda:0") as collector:
        base = collector.base_embodiment
        save_meshes(args.output / "template.npz", base)
        for key, mesh in base.items():
            write_obj(args.output / f"template_{key}.obj", mesh)
        for mm in AMPLITUDES_MM:
            collector.space.sample_perturbation_m = mm / 1000
            collector.rng = np.random.default_rng(0)
            design_rejectors = Counter()
            event_counts = Counter()
            fingertip_counts = Counter()
            example_details = []
            accepted = 0
            started = time.monotonic()
            for proposal in range(args.proposals):
                design = collector.sample_embodiment()
                _, _, rejected, _, reasons = collector.env._assess_embodiment([design])
                events = [event for reason in reasons for event in reason_events(reason)]
                valid = not bool(rejected.any())
                if valid:
                    accepted += 1
                categories = {event[1] for event in events}
                design_rejectors.update(categories)
                event_counts.update(event[1] for event in events)
                fingertip_counts.update(f"{event[0]}: {event[1]}" for event in events)
                if proposal == 0:
                    plotted[mm] = (design, valid)
                if proposal == 0 or (valid and not any(row["kind"] == "first_accepted" for row in example_details)):
                    kind = "matched_first" if proposal == 0 else "first_accepted"
                    save_meshes(args.output / f"{mm:g}mm_{kind}.npz", design)
                    for key, mesh in design.items():
                        write_obj(args.output / f"{mm:g}mm_{kind}_{key}.obj", mesh)
                    example_details.append({"kind": kind, "proposal": proposal,
                                            "valid": valid, "reasons": list(reasons)})
            result = {
                "amplitude_mm": mm, "proposals": args.proposals, "accepted": accepted,
                "rejected": args.proposals - accepted, "acceptance_rate": accepted / args.proposals,
                "elapsed_seconds": time.monotonic() - started,
                "rejected_designs_by_cause": dict(design_rejectors.most_common()),
                "reason_events": dict(event_counts.most_common()),
                "fingertip_reason_events": dict(fingertip_counts.most_common()),
                "examples": example_details,
            }
            results.append(result)
            (args.output / "diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        render(args.output, base, plotted, collector.space.mounting_vertex_counts)


if __name__ == "__main__":
    main()
