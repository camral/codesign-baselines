"""Collate run status and available local metrics without hiding failures."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    rows = []
    for run in manifest["runs"]:
        row = {key: run.get(key) for key in ("env_id", "method", "seed", "gpu", "status", "returncode", "elapsed_seconds")}
        if "directory" not in run:
            rows.append(row)
            continue
        directory = Path(run["directory"])
        result = directory / "artifacts" / "result.json"
        if result.exists():
            payload = json.loads(result.read_text())
            row.update({key: payload.get(key) for key in ("frames", "best_score", "best_eval_reward", "evaluations", "rounds", "updates")})
            if payload.get("status") != "complete":
                row["status"] = payload.get("status", "failed")
            row["error"] = payload.get("error")
        metrics_path = directory / "artifacts" / "metrics.jsonl"
        if metrics_path.exists():
            evaluation_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if "evaluation_boundary_seconds" in line]
            completed_rows = [metrics for metrics in evaluation_rows
                if metrics.get("eval/status", "complete") == "complete"]
            if completed_rows:
                latest = max(completed_rows, key=lambda metrics: metrics["evaluation_boundary_seconds"])
                for key in ("evaluation_boundary_seconds", "train/reward", "eval/reward", "eval/return_std", "eval/length_mean", "eval/success_rate", "eval/divergence_rate", "train/fps", "eval/best_video_path", "eval/worst_video_path", "eval/replay_paths"):
                    if key in latest:
                        row[key] = latest[key]
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (args.directory / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collated {len(rows)} runs; {sum(row['status'] != 'complete' for row in rows)} incomplete/failed")


if __name__ == "__main__":
    main()
