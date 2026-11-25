"""LLM proposals share PPO+NGOpt's scheduler, scoring and launcher contract."""

import argparse
import json
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from common.codesign_adapter import SpaceCodec
from launch_comparison import command_for
from ppo_llm_rlgo.sampler import LLMSampler, add_arguments, parse_design
from ppo_ngopt_rlgo.gym_train import (
    ask_candidate_batch, repeat_designs, replace_rejected_designs,
    tell_candidate_batch,
)


def arguments():
    p = argparse.ArgumentParser()
    add_arguments(p)
    return SimpleNamespace(**vars(p.parse_args([])), env_id="TestEnv", seed=7, device="cpu")


def sampler(tmp_path, backend):
    codec = SpaceCodec(gym.spaces.Box(0, 1, (2,), dtype=np.float32))
    env = SimpleNamespace(embodiment_space=codec.space_flat, design_codec=codec, env=object())
    return LLMSampler(env, np.zeros(2), arguments(), tmp_path, backend=backend)


@pytest.mark.parametrize("text", ['{"embodiment":[true,0]}', '{"embodiment":[0]}', '{"embodiment":[NaN,0]}', '{"embodiment":[2,0]}', '{"embodiment":[[0],0]}', '{"embodiment":["0",0]}', '__import__("os")', '{"embodiment":[0,0],"extra":1}'])
def test_invalid_completion(text):
    with pytest.raises((ValueError, TypeError)):
        parse_design(text, np.zeros(2), np.ones(2), np.zeros(2, dtype=bool))


def test_discrete_and_fenced_completion():
    np.testing.assert_equal(parse_design('```json\n{"embodiment":[1,0.3]}\n```', np.zeros(2), np.ones(2), np.array([True,False])), np.array([1,.3],dtype=np.float32))
    with pytest.raises(ValueError, match="integral"):
        parse_design('{"embodiment":[0.3,0]}', np.zeros(2), np.ones(2), np.array([True,False]))


def test_retry_scoring_feedback_and_fallback(tmp_path):
    calls = []
    def backend(prompt, count, budget):
        calls.append(count)
        return ['bad']*count if len(calls)==1 else ['{"embodiment":[0.2,0.8]}']*count
    opt = sampler(tmp_path, backend)
    candidates = ask_candidate_batch(opt, 3, SimpleNamespace(expired=False))
    assert len(candidates)==3 and calls==[3,3] and opt.invalid==3
    scores, score, design = tell_candidate_batch(opt, candidates, [1,2,3], [[False],[True],[False]], -10)
    np.testing.assert_equal(scores, [1,-10,3])
    assert score==3 and opt.round==1
    assert any(row['rejected'] and row['reward']==-10 for row in opt.history)
    assert 'previous_round_results' in opt.prompt()[1]['content']
    fixed = replace_rejected_designs(np.stack([c.value for c in candidates]), [[False],[True],[False]], [0,0])
    np.testing.assert_equal(fixed[1], [0,0])
    opt.dump(tmp_path/'optimizer.pkl')
    assert (tmp_path/'llm_sampler.json').exists()
    assert len((tmp_path/'llm_feedback.jsonl').read_text().splitlines())==3


def test_searched_designs_repeat_over_larger_ppo_batch():
    designs = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
    np.testing.assert_equal(
        repeat_designs(designs, 5),
        [[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]],
    )


def test_invalid_output_retries_until_deadline_without_random_fill(tmp_path):
    budget = SimpleNamespace(expired=False)
    def backend(prompt,count,b):
        b.expired=True
        return ['not json']*count
    opt = sampler(tmp_path,backend)
    assert ask_candidate_batch(opt,4,budget)==[]
    assert opt.accepted==0 and opt.invalid==4


def test_progress_hook_and_microbatching(tmp_path):
    opt = sampler(tmp_path,lambda p,n,b: ['{"embodiment":[0,1]}']*n)
    opt.args.llm_batch_size=2
    calls=[]
    assert len(ask_candidate_batch(opt,5,SimpleNamespace(expired=False),progress=lambda:calls.append(1)))==5
    assert len(calls)==3


def test_launcher_routes_to_shared_runner(tmp_path):
    args = SimpleNamespace(num_envs=16384, env_id="BallCatcher-Pitch", hours=4, project="test", entity="test", group="test")
    command = command_for(args,"ppo_llm",3,7,tmp_path)
    assert "ppo_llm_rlgo.gym_train" in command
    assert command[command.index("--num-envs")+1]=="16384"
    assert command[command.index("--run-name")+1]=="ballcatcher_pitch_ppo_llm_seed_3"


def test_launcher_forwards_llm_options_only_to_llm(tmp_path):
    args = SimpleNamespace(num_envs=2, env_id="TestEnv", hours=1, project=None, entity=None, group=None, llm_batch_size=3, llm_local_files_only=True)
    command = command_for(args, "ppo_llm", 0, 0, tmp_path)
    assert command[command.index("--llm-batch-size")+1] == "3"
    assert "--llm-local-files-only" in command
    assert "--llm-batch-size" not in command_for(args, "ppo_ngopt", 0, 0, tmp_path)


def test_shared_runner_complete_cycle(tmp_path, monkeypatch):
    """Real CPU PPO update, injected proposals, snapshot publication and saves."""
    tmp_path = tmp_path / "run"
    from common.run_log import RunLog
    from common.wall_clock import WallClockBudget
    from ppo_ngopt_rlgo import gym_train
    from ppo_llm_rlgo import sampler as sampler_module

    clock = [0.0]
    budget = WallClockBudget(100, 0, lambda: clock[0])
    monkeypatch.setattr(WallClockBudget, "start", lambda hours: budget)
    instances = []

    class ToyEnv:
        def __init__(self, n):
            self.num_envs, self.device, self.render_mode = n, "cpu", "rgb_array"
            self.design_codec = SpaceCodec(gym.spaces.Box(0, 1, (2,), dtype=np.float32))
            self.embodiment_space = self.design_codec.space_flat
            self.single_observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            self.single_action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            self.base_embodiment = np.zeros(2, dtype=np.float32)
            self.rejection_reward, self.env = -10, object()
            self.steps = torch.zeros(n, dtype=torch.long)
            self.designs = torch.zeros(n, 2)
            self.closed = False

        def reset(self, embodiment, seed=None, options=None):
            mask = torch.ones(self.num_envs, dtype=torch.bool) if options is None else options["reset_mask"].reshape(-1)
            designs = torch.as_tensor(embodiment).expand(self.num_envs, 2)
            self.designs[mask] = designs[mask]
            self.steps[mask] = 0
            return self.designs.clone(), torch.zeros(self.num_envs, 1, dtype=torch.bool), {}

        def step(self, action):
            self.steps += 1
            return self.designs.clone(), torch.ones(self.num_envs), self.steps >= 2, torch.zeros(self.num_envs, dtype=torch.bool), {}

        def render(self, env_idx=0):
            return np.zeros((32, 32, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    def make_env(env_id, n, **kwargs):
        env = ToyEnv(n)
        instances.append(env)
        return env, env.base_embodiment

    class InjectedSampler(LLMSampler):
        def __init__(self, env, base, args, output):
            super().__init__(env, base, args, output, backend=lambda p,n,b: ['{"embodiment":[0.2,0.8]}']*n)

        def end_round(self):
            super().end_round()
            clock[0] = 6.0

    snapshots = []
    class RecordingEvaluator:
        best_eval_reward = None
        def __init__(self, *args, **kwargs):
            self.last = float("-inf")
        def publish(self, snapshot, *, force=False, **metrics):
            if force or clock[0] - self.last >= 1:
                snapshots.append(snapshot())
                self.last = clock[0]
                if clock[0] >= 6:
                    clock[0] = 100
        def close(self, **kwargs):
            pass

    monkeypatch.setattr(gym_train, "make_baseline_vec", make_env)
    monkeypatch.setattr(sampler_module, "LLMSampler", InjectedSampler)
    monkeypatch.setattr(gym_train, "AsyncEvaluator", RecordingEvaluator)
    monkeypatch.setattr(gym_train, "PPO_SETTINGS", {**gym_train.PPO_SETTINGS, "n_epochs": 1, "batch_size": 512})
    monkeypatch.setattr(sys, "argv", ["ppo_llm", "--hours", "1", "--num-envs", "2", "--evaluation-worlds", "2", "--device", "cpu", "--output-dir", str(tmp_path), "--env-id", "TestEnv"])
    gym_train.main(method="ppo_llm")
    result = json.loads((tmp_path/"result.json").read_text())
    assert result["status"] == "complete" and result["rounds"] == 1
    assert result["completed_ppo_iterations"] == 1 and result["search_frames"] == 4
    rows = [json.loads(line) for line in (tmp_path/"metrics.jsonl").read_text().splitlines()]
    assert snapshots[0]["ppo"] and snapshots[-1]["ppo"]
    np.testing.assert_allclose(snapshots[0]["design"], [0, 0])
    np.testing.assert_allclose(snapshots[-1]["design"], [.2, .8])
    assert (tmp_path/"best_checkpoint.zip").exists() and (tmp_path/"llm_sampler.json").exists()
    np.testing.assert_allclose(np.load(tmp_path/"best_embodiment.npy"), [.2,.8])
    assert all(env.closed for env in instances)
