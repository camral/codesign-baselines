"""Tests for mujoco_visualiser.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco_visualiser as mv


def test_is_mujoco_env():
    assert mv.is_mujoco_env("HandPickUp-SmallHammer")
    assert mv.is_mujoco_env("RotateInHand-Cube")
    assert mv.is_mujoco_env("Ant-v4")
    assert mv.is_mujoco_env("HalfCheetah-v4")
    assert mv.is_mujoco_env("BallCatcher-Pitch")
    assert mv.is_mujoco_env("SolarCleaner-Reach")
    assert mv.is_mujoco_env("TruckUnload-Single")
    assert mv.is_mujoco_env("NeroGrasp-Tiny")
    assert not mv.is_mujoco_env("Pokenv-Random")
    assert not mv.is_mujoco_env("SoftWalker-Spring2D")
    assert not mv.is_mujoco_env("LQStructure-LQR")
    assert not mv.is_mujoco_env("Racing-Spielberg")


def test_inspect_checkpoint_rejects_non_mujoco(tmp_path):
    ckpt = tmp_path / "best_checkpoint.pt"
    torch.save({"actor": {}, "design": np.zeros(4, dtype=np.float32), "env_id": "Pokenv-Random"}, ckpt)
    with pytest.raises(ValueError, match="not a MuJoCo environment"):
        mv.inspect_checkpoint(ckpt)


def test_inspect_checkpoint_detects_formats_and_sibling_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"env_id": "Ant-v4", "embodiment_conditioned": True}))

    npz_ckpt = tmp_path / "best_checkpoint.npz"
    np.savez(npz_ckpt, design=np.zeros(4, dtype=np.float32), policy=np.zeros(10, dtype=np.float32), action_size=2)
    spec_npz = mv.inspect_checkpoint(npz_ckpt)
    assert spec_npz["method"] == "cmaes"
    assert spec_npz["env_id"] == "Ant-v4"
    assert spec_npz["embodiment_conditioned"] is True

    zip_ckpt = tmp_path / "best_checkpoint.zip"
    zip_ckpt.write_bytes(b"placeholder")
    spec_zip = mv.inspect_checkpoint(zip_ckpt)
    assert spec_zip["method"] == "ppo"
    assert spec_zip["env_id"] == "Ant-v4"

    pt_cmaes = tmp_path / "fasttd3_cmaes.pt"
    torch.save(
        {"method": "fasttd3_cmaes", "actor": {}, "design": np.zeros(4, dtype=np.float32), "env_id": "HandPickUp-Cube"},
        pt_cmaes,
    )
    spec_cmaes = mv.inspect_checkpoint(pt_cmaes)
    assert spec_cmaes["method"] == "fasttd3_cmaes"
    assert spec_cmaes["env_id"] == "HandPickUp-Cube"

    pt_joint = tmp_path / "fasttd3.pt"
    torch.save({"actor": {}, "env_id": "BallCatcher-Pitch"}, pt_joint)
    spec_joint = mv.inspect_checkpoint(pt_joint)
    assert spec_joint["method"] == "fasttd3"
    assert spec_joint["env_id"] == "BallCatcher-Pitch"


def test_visualise_headless_cmaes_ant(tmp_path):
    from common.codesign_adapter import make_baseline_vec

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env, base_design = make_baseline_vec("Ant-v4", num_envs=1, device=device, seed=0, is_eval_env=True)
    obs_dim = int(env.single_observation_space.shape[0])
    act_dim = int(env.single_action_space.shape[0])
    env.close()

    hidden = 32
    weights = np.zeros(obs_dim * hidden + hidden + hidden * act_dim + act_dim, dtype=np.float32)
    ckpt = tmp_path / "best_checkpoint.npz"
    np.savez(ckpt, design=base_design, policy=weights, action_size=act_dim, embodiment_conditioned=False)

    results = mv.visualise(
        ckpt,
        env_id="Ant-v4",
        device=device,
        seed=0,
        episodes=1,
        max_steps=5,
        headless=True,
    )
    assert len(results) == 1
    assert results[0]["length"] == 5
    assert np.isfinite(results[0]["return"])


def test_viewer_key_callback_chains_and_restores(monkeypatch):
    import mujoco.viewer

    captured = {}

    def fake_launch(*args, **kwargs):
        captured["cb"] = kwargs.get("key_callback")
        return "fake_viewer"

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_launch)
    seen = []

    with mv._viewer_key_callback(lambda k: seen.append(("ours", k))):
        res = mujoco.viewer.launch_passive("m", "d", key_callback=lambda k: seen.append(("orig", k)))
        assert res == "fake_viewer"
        captured["cb"](ord("P"))

    assert seen == [("ours", ord("P")), ("orig", ord("P"))]
    assert mujoco.viewer.launch_passive is fake_launch
