import json

import matplotlib.pyplot as plt

from plot_comparison import collect_runs, metric_points, plot_preset


def write_run(root, method, seed, rows, status="complete"):
    directory = root / f"{method}_seed_{seed}"
    artifacts = directory / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"env_id": "ExamplePreset", "method": method, "seed": seed, "status": status,
        "directory": str(directory)}


def test_plot_preset_writes_png_and_svg_with_seed_histories(tmp_path):
    runs = [
        write_run(tmp_path, "fasttd3", 0, [
            {"elapsed_seconds": 10, "train/reward": 1, "embodiment_candidates": 10, "embodiment_rejections": 2},
            {"elapsed_seconds": 20, "train/reward": 3, "embodiment_rejection_rate": 0.1},
            {"elapsed_seconds": 21, "evaluation_boundary_seconds": 20, "eval/reward": 4,
                "eval/success_rate": 0.5, "eval/divergence_rate": 0.1, "train/fps": 100}],
        ),
        write_run(tmp_path, "fasttd3", 1, [
            {"elapsed_seconds": 10, "train/reward": 2, "embodiment_candidates": 10, "embodiment_rejections": 4},
            {"elapsed_seconds": 21, "evaluation_boundary_seconds": 20, "eval/reward": 6,
                "eval/success_rate": 0.75, "eval/divergence_rate": 0, "train/fps": 120}],
            status="failed"),
        write_run(tmp_path, "cmaes", 0, [
            {"elapsed_seconds": 10, "evaluations": 4, "rejections": 1, "embodiment_rejections": 1},
            {"elapsed_seconds": 20, "evaluations": 8, "rejections": 3, "embodiment_rejections": 2}],
        ),
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    loaded = collect_runs(tmp_path, "ExamplePreset")
    fast_points = metric_points(loaded[0]["rows"], "embodiment_rejection_rate")
    cma_points = metric_points(loaded[2]["rows"], "embodiment_rejection_rate")
    assert [value for _, value in fast_points] == [0.2, 0.1]
    assert [value for _, value in cma_points] == [0.25, 0.5]
    paths = plot_preset(tmp_path, "ExamplePreset", tmp_path / "figures")
    assert {path.suffix for path in paths} == {".png", ".svg"}
    assert all(path.stat().st_size > 1000 for path in paths)
    assert not plt.get_fignums()
