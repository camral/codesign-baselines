"""Master launcher grouping, dispatch, and failure handling tests."""

import json

import pytest

from loki import collect_all


def test_default_dry_run_generates_36_corpora_covering_every_preset(tmp_path, capsys):
    root = tmp_path / "corpora"
    assert len(collect_all.CORPUS_GROUPS) == 36
    assert set(collect_all.PRESET_TO_CORPUS) == set(collect_all.COLLECTORS)
    assert sum(map(len, collect_all.CORPUS_GROUPS.values())) == len(collect_all.COLLECTORS)
    assert collect_all.main(["--dry-run", "--output-dir", str(root)]) == 0
    output = capsys.readouterr().out
    assert output.count("--env-id") == 36
    for corpus_id, presets in collect_all.CORPUS_GROUPS.items():
        assert f"--env-id {corpus_id} " in output
        assert corpus_id + " serves " + ", ".join(presets) in output
    assert not root.exists()


def test_shared_preset_selection_is_deduplicated(tmp_path, capsys):
    assert collect_all.main([
        "--dry-run", "--output-dir", str(tmp_path), "--presets",
        "BallCatcher-Arc", "BallCatcher-Fielding", "SolarCleaner-Traverse",
    ]) == 0
    output = capsys.readouterr().out
    assert output.count("--env-id") == 2
    assert "--env-id BallCatcher-Pitch " in output
    assert "--env-id SolarCleaner-Reach " in output


def test_humanoid_standup_reuses_humanoid_preparation(tmp_path, capsys):
    assert collect_all.PRESET_TO_CORPUS["HumanoidStandup-v4"] == "Humanoid-v4"
    assert collect_all.main([
        "--dry-run", "--output-dir", str(tmp_path), "--presets",
        "Humanoid-v4", "HumanoidStandup-v4",
    ]) == 0
    output = capsys.readouterr().out
    assert output.count("--env-id") == 1
    assert "Humanoid-v4 serves Humanoid-v4, HumanoidStandup-v4" in output


def test_truck_unload_single_has_its_own_preparation(tmp_path, capsys):
    assert collect_all.PRESET_TO_CORPUS["TruckUnload-Single"] == "TruckUnload-Single"
    assert collect_all.PRESET_TO_CORPUS["RotateInHand-Cube"] == "HandPickUp-Cube"
    assert collect_all.PRESET_TO_CORPUS["SoftWalker-Beam3D"] == "SoftWalker-Beam3D"
    assert "TruckUnload-Multi" not in collect_all.COLLECTORS
    assert collect_all.main([
        "--dry-run", "--output-dir", str(tmp_path), "--presets",
        "TruckUnload-Single",
    ]) == 0
    output = capsys.readouterr().out
    assert output.count("--env-id") == 1
    assert "TruckUnload-Single serves TruckUnload-Single" in output


def test_resume_and_continue_after_failure(tmp_path, monkeypatch):
    existing = tmp_path / "BallCatcher-Pitch"
    existing.mkdir()
    (existing / "manifest.json").write_text("{}")
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self):
            return 2 if len(calls) == 1 else 0

    monkeypatch.setattr(collect_all.subprocess, "Popen", Process)
    assert collect_all.main(["--output-dir", str(tmp_path), "--count", "4",
                             "--presets", "BallCatcher-Arc", "Pokenv-Random"]) == 1
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index("--env-id") + 1] == "BallCatcher-Pitch"
    assert "--resume" in calls[0][0]
    assert "--resume" not in calls[1][0]
    assert all(call[1]["cwd"] == collect_all.REPO for call in calls)
    summary = json.loads((tmp_path / "collection_summary.json").read_text())
    assert [result["returncode"] for result in summary["results"]] == [2, 0]
    assert summary["target_per_corpus"] == 4
    assert summary["requested_corpora"] == ["BallCatcher-Pitch", "Pokenv-Random"]
    assert summary["results"][0]["serves_presets"] == list(
        collect_all.CORPUS_GROUPS["BallCatcher-Pitch"])


def test_alias_override_is_applied_to_canonical_corpus(tmp_path, monkeypatch):
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"NeroGrasp-Tiny": {"sample_perturbation_m": 0.0002}}))
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(collect_all.subprocess, "Popen", Process)
    assert collect_all.main(["--output-dir", str(tmp_path), "--count", "1",
                             "--presets", "NeroGrasp-Tiny",
                             "--overrides", str(overrides)]) == 0
    command = commands[0]
    assert command[command.index("--env-id") + 1] == "NeroGrasp-All"
    assert json.loads(command[command.index("--env-kwargs") + 1]) == {
        "sample_perturbation_m": 0.0002}


def test_conflicting_shared_overrides_are_rejected(tmp_path):
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({
        "BallCatcher-Pitch": {"seed": 1},
        "BallCatcher-Arc": {"seed": 2},
    }))
    with pytest.raises(SystemExit):
        collect_all.main(["--dry-run", "--presets", "BallCatcher-Pitch",
                          "BallCatcher-Arc", "--overrides", str(overrides)])


def test_jax_children_pin_first_visible_gpu(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_CUDA_VISIBLE_DEVICES", "all")

    class Process:
        def __init__(self, command, **kwargs):
            calls.append(kwargs["env"])

        def wait(self):
            return 0

    monkeypatch.setattr(collect_all.subprocess, "Popen", Process)
    for device, backend in (("cuda:0", "cuda"), ("cpu", "cpu")):
        assert collect_all.main(["--output-dir", str(tmp_path), "--device", device,
                                 "--presets", "Racing-Levine", "Warehouse-Small"]) == 0
        for env in calls[-2:]:
            assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
            assert env["JAX_CUDA_VISIBLE_DEVICES"] == "0"
            assert env["JAX_PLATFORMS"] == backend


def test_corpora_keep_the_directories_they_were_collected_under():
    """Corpus / artifact dirs keep the pre-rename env ids; every canonical id is a registered env."""
    from codesign_gym import REGISTRY
    assert collect_all.corpus_dir("Microgrid-Random") == "MicrogridRandom"
    assert collect_all.corpus_dir("Pokenv-LeagueG1") == "PokenvLeague-G1"
    assert collect_all.corpus_dir("BallCatcher-Pitch") == "BallCatcher-Pitch"
    assert all(corpus in REGISTRY for corpus in collect_all.CORPUS_GROUPS)
    assert all(old not in REGISTRY for old in collect_all.CORPUS_DIRS.values())
