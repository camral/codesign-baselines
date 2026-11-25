"""Create a standalone interactive mesh viewer and downloadable mesh archive."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output / "visuals"
    examples = json.loads((root / "examples.json").read_text())
    entries = []
    for name in ("index", "thumb"):
        entries.append(("Template · " + name, args.output / "0.1mm/template.npz", name))
    for example in examples:
        mm, sample = example["amplitude_mm"], example["sample"]
        status = "ACCEPTED" if example["native_valid"] else "REJECTED"
        for name in ("index", "thumb"):
            entries.append((f"±{mm:g} mm · sample {sample} · {name} · {status}",
                            root / f"{mm:g}mm_sample{sample}.npz", name))
            processed = root / f"{mm:g}mm_sample{sample}_processed.npz"
            if processed.exists():
                entries.append((f"Processed ±{mm:g} mm · sample {sample} · {name} · {status}",
                                processed, "leap_if_tip" if name == "index" else "leap_th_tip"))
    fig = go.Figure()
    for index, (title, path, name) in enumerate(entries):
        with np.load(path) as arrays:
            v = arrays[name + "__vertices"] * 1000
            f = arrays[name + "__faces"]
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                               i=f[:, 0], j=f[:, 1], k=f[:, 2],
                               facecolor=["#82909e" if fixed else "#e7a246" for fixed in np.all(f < 25, axis=1)],
                               flatshading=True, visible=index == 0, name=title,
                               hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm<extra></extra>"))
    buttons = []
    for index, (title, _, _) in enumerate(entries):
        buttons.append(dict(label=title, method="update", args=[
            {"visible": [i == index for i in range(len(entries))]},
            {"title": {"text": "NeroGrasp pilot — " + title}}]))
    fig.update_layout(title="NeroGrasp pilot — Template · index", height=850,
                      scene=dict(aspectmode="data", xaxis_title="x (mm)", yaxis_title="y (mm)",
                                 zaxis_title="z (mm)", uirevision="same-camera"),
                      updatemenus=[dict(buttons=buttons, x=0, y=1.08, xanchor="left")],
                      annotations=[dict(text="Drag to rotate · Scroll to zoom · Grey: mounting · Orange: mutable surface<br>"
                                        "Exact meshes, no displacement magnification. REJECTED examples are not valid corpus entries.",
                                        x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False)],
                      margin=dict(t=130, b=100))
    fig.write_html(root / "viewer.html", include_plotlyjs=True, auto_open=False)
    with zipfile.ZipFile(root / "meshes.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.glob("*.obj")):
            archive.write(path, path.name)
        for name in ("examples.json", "rejection_details.json"):
            if (root / name).exists():
                archive.write(root / name, name)
    print(root / "viewer.html")


if __name__ == "__main__":
    main()
