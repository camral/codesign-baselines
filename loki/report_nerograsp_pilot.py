"""Summarize saved native-validation pilot manifests and mesh artifacts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    complete = True
    for mm in (0.1, 0.5, 1, 2):
        manifest = json.loads((args.output / f"{mm:g}mm/corpus/manifest.json").read_text())
        stats = manifest["state"]["stats"]
        attempts = stats["attempted"]
        complete &= attempts == 1000 and manifest["status"] != "running"
        rows.append(f"| ±{mm:g} mm | {attempts:,} | {stats['accepted']:,} | "
                    f"{100 * stats['accepted'] / max(1, attempts):.1f}% | {stats['invalid']:,} |")
    text = """# NeroGrasp perturbation-amplitude pilot

Status: STATUS

Historical timing note: this report may describe a run before Codesign Gym commit `328d8f5`; current validation defers convex decomposition to reset.

## Experiment

- Preset: `NeroGrasp-SetScrew`; seed 0, matched proposal directions across amplitudes.
- 1,000 proposals per amplitude; accepted count is an outcome, not a stopping target.
- Native `MeshSpace.sample`: independent uniform XYZ noise around the template.
- Two fingertip meshes (index/thumb), each 98 vertices and 192 triangles; 25 fixed mounting vertices.
- Template bounding box: 16 × 14 × 16 mm per fingertip.
- Full native `validate_embodiment`, including enabled geometry rejectors. Convex decomposition is deferred to reset in the current environment.
- Minimum thickness 0.3 mm and minimum volume ratio 0.9.
- Proposal corpus collection has reset audits disabled to isolate validity. The pilot separately resets up to three accepted examples per amplitude.
- No controller training or grasp-performance evaluation was performed.

## Native validation

| Coordinate noise | Proposals | Accepted | Acceptance | Rejected |
| --- | ---: | ---: | ---: | ---: |
ROWS

`attempt_limit` in a corpus manifest is expected here: a fixed proposal budget
was used to measure acceptance, rather than generating additional proposals until
1,000 designs passed. Zero observed acceptance is not a proof that no valid designs exist.

## Meshes

![Paired proposals at true scale](visuals/comparison.png)

- [Interactive viewer: both fingertips, three proposals per amplitude, and processed accepted meshes](visuals/viewer.html)
- [OBJ mesh archive](visuals/meshes.zip) — coordinates in metres; includes rejected examples with explicit validity labels in `examples.json`.
- [PDF comparison](visuals/comparison.pdf)
- [Native rejection details for the displayed first proposal](visuals/rejection_details.json)

All panels use the same scale and camera; displacements are not magnified.
The plotted 0.1 mm proposal is accepted. The plotted larger-amplitude proposals
are rejected and are shown to explain what the sampler produces, not as valid corpus entries.
For those examples, native rejection identifies minimum thickness below 0.3 mm:
0.021–0.053 mm at 0.5 mm noise and approximately 0.011–0.022 mm at 1–2 mm noise.

## Interpretation

0.1 mm produces small local changes to a 16 mm template. Larger independent
vertex perturbations produce visible surface irregularity but no accepted designs
in these 1,000-proposal trials. Do not increase the production amplitude to 0.5–2 mm
with this sampler. A useful next experiment is smooth, coordinated deformation of
the mutable region while preserving the mounting and retaining the same validators.
This pilot does not establish whether the accepted 0.1 mm designs differ in grasp performance,
nor does it measure validity across other object presets.
"""
    text = text.replace("STATUS", "Complete: 4,000 proposals assessed." if complete else "Running; table reflects committed progress.")
    text = text.replace("ROWS", "\n".join(rows))
    (args.output / "report.md").write_text(text)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
