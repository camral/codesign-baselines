import json
import sys
from types import SimpleNamespace

import numpy as np

from common.run_log import RunLog
from common.wall_clock import WallClockBudget
from launch_comparison import METHODS, command_for
import collate_comparison
import launch_environment_suite


def test_launcher_commands(tmp_path):
    args = SimpleNamespace(hours=1, num_envs=128, env_id="BallCatcher-Arc", project="test", group="group", entity="entity")
    for method in METHODS:
        command = command_for(args, method, 17, 3, tmp_path)
        assert command[0] == sys.executable
        assert command[command.index("--hours") + 1] == "1"
        assert command[command.index("--output-dir") + 1] == str(tmp_path)
        assert command[command.index("--env-id") + 1] == "BallCatcher-Arc"


def test_fasttd3_cmaes_command_carries_replay_cma_and_search_settings(tmp_path):
    args = SimpleNamespace(hours=1, num_envs=128, env_id="BallCatcher-Arc", project=None, group=None, entity=None,
        replay_transitions=1000, replay_memory_gb=2.0, cma_population_size=32, cma_covariance="full",
        search_frequency=250)
    command = command_for(args, "fasttd3_cmaes", 0, 0, tmp_path)
    assert command[command.index("-m") + 1] == "fasttd3_cmaes.gym_train"
    for flag, value in (("--replay-transitions", "1000"), ("--replay-memory-gb", "2.0"),
                        ("--population-size", "32"), ("--covariance", "full"), ("--search-frequency", "250")):
        assert command[command.index(flag) + 1] == value
    args.search_frequency = None
    assert "--search-frequency" not in command_for(args, "fasttd3_cmaes", 0, 0, tmp_path)


def test_inherited_start_includes_import_time(monkeypatch):
    monkeypatch.setenv("CODESIGN_RUN_STARTED", "123")
    budget = WallClockBudget.start(1)
    assert budget.started_at == 123
    budget = WallClockBudget.start(1, clock=lambda: 456)
    assert budget.started_at == 456


def test_environment_suite_expands_family_and_qualifies_run_names(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["launch_environment_suite.py", "--families", "lq",
        "--seeds", "3", "--gpus", "0", "--hours", "1", "--num-envs", "64",
        "--output-dir", str(tmp_path / "runs"), "--dry-run"])
    launch_environment_suite.main()
    commands = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == 3 * len(launch_environment_suite.DEFAULT_METHODS)
    assert {command[command.index("--env-id") + 1] for command in commands} == set(launch_environment_suite.FAMILIES["lq"])
    assert all("lqstructure_" in command[command.index("--run-name") + 1] for command in commands)


def test_evaluation_video_omits_unavailable_initial_frame(tmp_path):
    log = RunLog(tmp_path / "run", WallClockBudget.start(1), {})
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    log.log_evaluation([None, frame], [None, frame], evaluation_boundary_seconds=1.0)
    log.finish("complete")
    assert len(list((tmp_path / "run" / "evaluation_videos").glob("*.mp4"))) == 2


def test_collate_queued_and_failed_runs(tmp_path, monkeypatch):
    artifact = tmp_path / "failed" / "artifacts"
    artifact.mkdir(parents=True)
    (artifact / "result.json").write_text(json.dumps({"status": "failed", "error": "test failure", "best_eval_reward": 2.0}))
    (artifact / "metrics.jsonl").write_text(json.dumps({"evaluation_boundary_seconds": 180, "train/reward": 1.0, "eval/reward": 2.0, "eval/best_video_path": "best.mp4", "eval/worst_video_path": "worst.mp4", "eval/replay_paths": ["battle.html"]}) + "\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [{"method": "cmaes", "seed": 1, "status": "queued"}, {"method": "fasttd3", "seed": 2, "status": "complete", "directory": str(artifact.parent)}]}))
    monkeypatch.setattr(sys, "argv", ["collate_comparison.py", str(tmp_path)])
    collate_comparison.main()
    summary = (tmp_path / "summary.csv").read_text()
    assert "queued" in summary and "failed" in summary and "test failure" in summary
    assert "best_eval_reward" in summary and "eval/reward" in summary and "best.mp4" in summary and "worst.mp4" in summary and "battle.html" in summary


def test_collate_keeps_completed_score_after_new_boundary_status(tmp_path, monkeypatch):
    artifact = tmp_path / "run" / "artifacts"
    artifact.mkdir(parents=True)
    metrics = [
        {"evaluation_boundary_seconds": 720, "eval/status": "scheduled"},
        {"evaluation_boundary_seconds": 720, "eval/status": "complete", "eval/reward": 12.5},
        {"evaluation_boundary_seconds": 1440, "eval/status": "started"},
        {"evaluation_boundary_seconds": 1440, "eval/status": "budget_exhausted"},
    ]
    (artifact / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": [
        {"method": "fasttd3", "status": "complete", "directory": str(artifact.parent)}]}))
    monkeypatch.setattr(sys, "argv", ["collate_comparison.py", str(tmp_path)])
    collate_comparison.main()
    import csv
    with (tmp_path / "summary.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["eval/reward"] == "12.5" and row["evaluation_boundary_seconds"] == "720"


def test_resource_options_reach_only_the_relevant_trainers(tmp_path):
    args = SimpleNamespace(hours=1, num_envs=16384, env_id="Warehouse-Congested",
        project=None, group=None, entity=None, replay_transitions=1000000,
        replay_memory_gb=6, cma_population_size=64, cma_covariance="full")
    fast = command_for(args, "fasttd3", 0, 0, tmp_path)
    cma = command_for(args, "cmaes", 0, 0, tmp_path)
    ppo = command_for(args, "ppo_ngopt", 0, 0, tmp_path)
    assert fast[fast.index("--replay-memory-gb") + 1] == "6"
    assert fast[fast.index("--replay-transitions") + 1] == "1000000"
    assert cma[cma.index("--population-size") + 1] == "64"
    assert cma[cma.index("--covariance") + 1] == "full"
    assert "--population-size" not in fast
    assert "--replay-memory-gb" not in cma + ppo
    for command in (fast, cma, ppo):
        assert command[command.index("--num-envs") + 1] == "16384"
