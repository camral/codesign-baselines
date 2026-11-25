"""Real Network-Urban rollouts from the snapshot format used by each trainer."""

import io
import threading
import time
from dataclasses import asdict

import numpy as np
import pytest
import torch

from common.codesign_adapter import make_baseline_vec
from common.evaluation_worker import SnapshotEvaluator
from common.gym_batch import GymBatch
from common.sb3_env import SB3GymBatch
from cmaes_evolutionary.gym_train import policy_size


@pytest.mark.parametrize("method", ["fasttd3", "ppo_ngopt", "cmaes", "fasttd3_cmaes"])
@pytest.mark.parametrize("rejected_design", [False, True])
def test_networkurban_snapshot_video_and_matching_checkpoint(tmp_path, method, rejected_design):
    torch.set_num_threads(1)
    config = dict(env_id="Network-Urban", evaluation_worlds=2, seed=0, device="cpu",
        method=method, output_dir=str(tmp_path), video_stride=25, video_fps=20,
        rejection_reward=None)
    backend = SnapshotEvaluator(config)
    env = backend.env
    obs_size = env.single_observation_space.shape[0]
    act_size = env.single_action_space.shape[0]
    base = env.base_embodiment
    try:
        if method == "fasttd3":
            from fast_td3.fast_td3 import Actor
            from fast_td3.fast_td3_utils import EmpiricalNormalization
            from fast_td3.hyperparams import BaseArgs
            from fast_td3.codesign.evaluation import policy_snapshot, load_policy
            hp = asdict(BaseArgs())
            hp["actor_hidden_dim"] = 32
            state_dim = obs_size + len(base) + 1
            actor = Actor(state_dim, len(base) + act_size, 2, hp["init_scale"], 32,
                std_min=hp["std_min"], std_max=hp["std_max"], device="cpu")
            if rejected_design:
                with torch.no_grad():
                    actor.fc_mu[0].weight[0].zero_()
                    actor.fc_mu[0].bias[0] = -20
            payload = policy_snapshot(actor, EmpiricalNormalization(state_dim, "cpu"),
                env_id="Network-Urban", seed=0, state_dim=state_dim, action_dim=len(base) + act_size,
                design_dim=len(base), hp=hp, amp=False)
        elif method == "fasttd3_cmaes":
            from fast_td3.fast_td3 import Actor
            from fast_td3.fast_td3_utils import EmpiricalNormalization
            from fast_td3.hyperparams import BaseArgs
            from fast_td3.codesign.evaluation import policy_snapshot
            hp = asdict(BaseArgs())
            hp["actor_hidden_dim"] = 32
            state_dim = obs_size + len(base)
            actor = Actor(state_dim, act_size, 2, hp["init_scale"], 32,
                std_min=hp["std_min"], std_max=hp["std_max"], device="cpu")
            design = base.copy()
            if rejected_design:
                design[0] = 0
            payload = {**policy_snapshot(actor, EmpiricalNormalization(state_dim, "cpu"),
                env_id="Network-Urban", seed=0, state_dim=state_dim, action_dim=act_size,
                design_dim=len(base), hp=hp, amp=False), "design": design, "embodiment_conditioned": True}
        elif method == "ppo_ngopt":
            from ppo_ngopt_rlgo.timed_ppo import TimedPPO
            vector = SB3GymBatch(GymBatch(np.stack([base] * 2), env=env))
            model = TimedPPO("MlpPolicy", vector, n_steps=2, batch_size=4, device="cpu")
            data = io.BytesIO()
            model.save(data)
            design = base.copy()
            if rejected_design:
                design[0] = 0
            payload = {"ppo": data.getvalue(), "design": design, "embodiment_conditioned": True}
        else:
            design = base.copy()
            if rejected_design:
                design[0] = 0
            payload = {"design": design, "policy": np.zeros(policy_size(obs_size + len(base), act_size)),
                "observation_size": obs_size, "action_size": act_size,
                "embodiment_conditioned": True}
        payload["_evaluation"] = dict(version=1, elapsed_seconds=0, kind="current", metrics={"updates": 0})
        path = tmp_path / "boundary_01.pt"
        torch.save(payload, path)
        metrics, videos = backend.evaluate(path, deadline=time.monotonic() + 60, stop=threading.Event())
        assert metrics["eval/status"] == "complete"
        if rejected_design:
            assert metrics["eval/reward"] == env.rejection_reward
            assert metrics["eval/length_mean"] == 0
            assert metrics["eval/embodiment_rejections"] == 2
            assert metrics["eval/embodiment_rejection_rate"] == 1
            assert metrics["eval/video_complete"] is False
            assert videos == {}
            return
        assert metrics["eval/embodiment_rejections"] == 0
        assert metrics["eval/embodiment_rejection_rate"] == 0
        assert metrics["eval/length_mean"] == 100
        assert metrics["eval/video_complete"]
        assert set(videos) == {"best", "worst"}
        from pathlib import Path
        assert all(Path(video).stat().st_size > 0 for video in videos.values())
        backend.save_best(path, metrics["eval/reward"])
        if method == "fasttd3_cmaes":
            from fast_td3.codesign.evaluation import load_policy
            saved, _ = load_policy(tmp_path / "best_checkpoint.pt", "cpu", compile=False)
            assert saved["eval_reward"] == metrics["eval/reward"]
            for key in payload["actor"]:
                torch.testing.assert_close(saved["actor"][key], payload["actor"][key])
            np.testing.assert_equal(np.load(tmp_path / "best_embodiment.npy"), base)
        elif method == "fasttd3":
            saved, _ = load_policy(tmp_path / "best_checkpoint.pt", "cpu", compile=False)
            assert saved["eval_reward"] == metrics["eval/reward"]
            for key in payload["actor"]:
                torch.testing.assert_close(saved["actor"][key], payload["actor"][key])
        elif method == "ppo_ngopt":
            from stable_baselines3 import PPO
            saved = PPO.load(tmp_path / "best_checkpoint.zip", device="cpu")
            for key in model.policy.state_dict():
                torch.testing.assert_close(saved.policy.state_dict()[key], model.policy.state_dict()[key])
            np.testing.assert_equal(np.load(tmp_path / "best_embodiment.npy"), base)
        else:
            saved = np.load(tmp_path / "best_checkpoint.npz")
            np.testing.assert_equal(saved["policy"], payload["policy"])
            assert saved["eval_reward"] == metrics["eval/reward"]
    finally:
        backend.close()


def test_video_failure_does_not_discard_evaluation(tmp_path):
    backend = SnapshotEvaluator.__new__(SnapshotEvaluator)
    backend.method = "cmaes"
    backend.seed = 0
    backend.device = "cpu"
    backend.output = tmp_path
    backend.config = {"video_stride": 1, "video_fps": 20}
    backend.replay_dir = None                      # as __init__ sets it for a non-Pokemon env
    summary = {"return_mean": 3.0, "return_std": 0.0, "length_mean": 4.0,
        "success_rate": 0.0, "divergence_rate": 0.0,
        "episode_returns": [3.0], "episode_lengths": [4]}

    class Complete:
        complete = True

        def summary(self):
            return summary

    calls = 0

    def rollout(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Complete(), [], 0, 0.0
        raise ValueError("render geometry failed")

    backend._rollout = rollout
    payload = {"design": np.zeros(1), "policy": np.zeros(1), "action_size": 1,
        "_evaluation": {"version": 1, "elapsed_seconds": 0, "kind": "current", "metrics": {}}}
    path = tmp_path / "boundary_01.pt"
    torch.save(payload, path)

    metrics, videos = backend.evaluate(path, deadline=time.monotonic() + 1, stop=threading.Event())

    assert metrics["eval/status"] == "complete"
    assert metrics["eval/reward"] == 3.0
    assert metrics["eval/video_complete"] is False
    assert metrics["eval/video_error"] == "ValueError: render geometry failed"
    assert videos == {}
