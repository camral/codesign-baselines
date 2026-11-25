"""Plot local baseline histories for one or more Codesign Gym presets."""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("eval/reward", "Evaluation reward"),
    ("eval/success_rate", "Evaluation success rate"),
    ("train/reward", "Training reward"),
    ("embodiment_rejection_rate", "Embodiment rejection rate"),
    ("eval/divergence_rate", "Evaluation divergence rate"),
    ("train/fps", "Training throughput (world FPS)"),
)
METHOD_LABELS = {"fasttd3": "FastTD3", "ppo_ngopt": "PPO + NGOpt", "cmaes": "CMA-ES", "ppo_llm": "PPO + LLM", "fasttd3_cmaes": "FastTD3 + CMA-ES"}
METHOD_COLORS = {"fasttd3": "#2878B5", "ppo_ngopt": "#E07B39", "cmaes": "#3A923A", "ppo_llm": "#8E5EA2", "fasttd3_cmaes": "#C23B5A"}


def read_rows(path):
    rows = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from error
    return rows


def rejection_rate(row, previous_evaluations=None):
    if "embodiment_rejection_rate" in row:
        return row["embodiment_rejection_rate"]
    rejected = row.get("embodiment_rejections")
    if rejected is not None:
        candidates = row.get("embodiment_candidates", row.get("candidates"))
        if candidates is None and previous_evaluations is not None and "evaluations" in row:
            candidates = row["evaluations"] - previous_evaluations
        if candidates:
            return rejected / candidates
    if row.get("evaluations"):
        return row.get("rejections", 0) / row["evaluations"]
    return None


def metric_points(rows, metric):
    points, previous_evaluations = {}, None
    for row in rows:
        value = rejection_rate(row, previous_evaluations) if metric == "embodiment_rejection_rate" else row.get(metric)
        if "evaluations" in row:
            previous_evaluations = row["evaluations"]
        if value is None or isinstance(value, bool):
            continue
        x = row.get("evaluation_boundary_seconds") if metric.startswith("eval/") else row.get("elapsed_seconds")
        if x is None or not math.isfinite(float(x)) or not math.isfinite(float(value)):
            continue
        points[float(x) / 3600] = float(value)
    return tuple(sorted(points.items()))


def run_env_id(manifest, run):
    return run.get("env_id") or manifest.get("config", {}).get("env_id")


def run_metrics_path(root, run):
    if "directory" not in run:
        return None
    directory = Path(run["directory"])
    if not directory.is_absolute():
        directory = root / directory
    return directory / "artifacts" / "metrics.jsonl"


def collect_runs(root, env_id):
    manifest = json.loads((root / "manifest.json").read_text())
    collected = []
    for run in manifest["runs"]:
        if run_env_id(manifest, run) != env_id:
            continue
        metrics_path = run_metrics_path(root, run)
        collected.append({**run, "rows": read_rows(metrics_path) if metrics_path else []})
    return collected


def interpolate(points, grid):
    if not points:
        return np.full(grid.shape, np.nan)
    x, y = (np.asarray(values, dtype=float) for values in zip(*points))
    if len(x) == 1:
        result = np.full(grid.shape, np.nan)
        result[np.isclose(grid, x[0])] = y[0]
        return result
    result = np.interp(grid, x, y)
    result[(grid < x[0]) | (grid > x[-1])] = np.nan
    return result


def plot_metric(axis, runs, metric, title):
    methods = sorted({run["method"] for run in runs})
    found = False
    for method_index, method in enumerate(methods):
        method_runs = [run for run in runs if run["method"] == method]
        histories = [(run, metric_points(run["rows"], metric)) for run in method_runs]
        histories = [(run, points) for run, points in histories if points]
        if not histories:
            continue
        found = True
        all_x = np.unique(np.concatenate([np.asarray([x for x, _ in points]) for _, points in histories]))
        if len(all_x) > 500:
            all_x = np.linspace(all_x[0], all_x[-1], 500)
        values = np.stack([interpolate(points, all_x) for _, points in histories])
        count = np.sum(np.isfinite(values), axis=0)
        mean = np.divide(np.nansum(values, axis=0), count, out=np.full(len(all_x), np.nan), where=count > 0)
        centered = np.where(np.isfinite(values), values - mean, 0)
        std = np.sqrt(np.divide(np.sum(centered ** 2, axis=0), count - 1,
            out=np.zeros(len(all_x)), where=count > 1))
        ci = 1.96 * std / np.sqrt(np.maximum(count, 1))
        color = METHOD_COLORS.get(method, f"C{method_index}")
        complete = sum(run.get("status") == "complete" for run, _ in histories)
        label = f"{METHOD_LABELS.get(method, method)} (n={len(histories)}, {complete} complete)"
        for _, points in histories:
            x, y = zip(*points)
            axis.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
        axis.plot(all_x, mean, color=color, linewidth=2.4, label=label)
        axis.fill_between(all_x, mean - ci, mean + ci, color=color, alpha=0.16, linewidth=0)
    axis.set_title(title, loc="left", fontweight="semibold")
    axis.set_xlabel("Wall-clock time (hours)")
    axis.grid(True, color="#D8DCE3", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    if metric.endswith("_rate"):
        axis.set_ylim(-0.02, 1.02)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    if not found:
        axis.text(0.5, 0.5, "No logged data", ha="center", va="center", color="#777777", transform=axis.transAxes)


def plot_preset(root, env_id, output):
    runs = collect_runs(root, env_id)
    if not runs:
        raise ValueError(f"manifest contains no runs for {env_id}")
    plt.rcParams.update({"font.size": 10, "axes.facecolor": "#FAFBFC", "figure.facecolor": "white",
        "axes.titlepad": 10, "savefig.bbox": "tight"})
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, METRICS):
        plot_metric(axis, runs, metric, title)
    legend = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    figure.suptitle(env_id, fontsize=18, fontweight="bold", y=1.04)
    if legend:
        figure.legend(legend.values(), legend.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=min(len(legend), 4), frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    stem = env_id.lower().replace("-v0", "").replace("-", "_")
    paths = [output / f"{stem}_training_curves.{extension}" for extension in ("png", "svg")]
    for path in paths:
        figure.savefig(path, dpi=220 if path.suffix == ".png" else None)
    plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Suite or single-preset result directory containing manifest.json")
    parser.add_argument("--env-ids", nargs="+", help="Exact presets to plot; defaults to every preset in the manifest")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DIRECTORY/plots")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    available = sorted({run_env_id(manifest, run) for run in manifest["runs"] if run_env_id(manifest, run)})
    env_ids = args.env_ids or available
    unknown = sorted(set(env_ids) - set(available))
    if unknown:
        parser.error(f"presets not found in manifest: {', '.join(unknown)}")
    if not env_ids:
        parser.error("manifest contains no environment preset IDs")
    output = args.output_dir or args.directory / "plots"
    for env_id in env_ids:
        paths = plot_preset(args.directory, env_id, output)
        print(f"{env_id}: " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
