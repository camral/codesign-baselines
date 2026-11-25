"""Frozen-policy design scoring that runs alongside live PPO training."""

from __future__ import annotations

import io
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import numpy as np
from stable_baselines3 import PPO


class FrozenPolicySearchBackend:
    """Own a separate vector environment used only for comparable scoring."""

    def __init__(self, config):
        from common.codesign_adapter import make_baseline_vec

        self.device = config["device"]
        self.seed = config["seed"]
        self.env, _ = make_baseline_vec(
            config["env_id"], config["num_envs"], device=self.device,
            seed=self.seed,
            control_mode="native", rejection_reward=config["rejection_reward"])

    def evaluate(self, policy_bytes, designs, stop, deadline):
        from common.embodiment_conditioning import condition_observation
        from common.gym_batch import EvaluationAccumulator

        # Loading a fresh immutable copy at the round boundary is intentional:
        # every candidate in this batch sees exactly the same controller while
        # the original controller continues PPO updates in the parent process.
        model = PPO.load(io.BytesIO(policy_bytes), device=self.device)
        observation, rejected, _ = self.env.reset(
            seed=self.seed, embodiment=designs)
        result = EvaluationAccumulator(self.env)
        result.reject(rejected, self.env.rejection_reward)
        while not result.complete and not stop.is_set() and time.monotonic() < deadline:
            conditioned = condition_observation(self.env, observation, designs)
            action, _ = model.predict(conditioned.cpu().numpy(), deterministic=True)
            observation, reward, terminated, truncated, _ = self.env.step(action)
            result.add(reward, terminated, truncated)
        if not result.complete:
            return None
        returns = (self.env.collapse_returns(result.returns).cpu().numpy()
            if hasattr(self.env, "collapse_returns") else result.returns.cpu().numpy())
        rejected = (self.env.collapse_rejections(rejected).cpu().numpy()
            if hasattr(self.env, "collapse_rejections") else rejected.cpu().numpy())
        return returns, rejected, result.frames

    def close(self):
        self.env.close()


def _worker(config, jobs, events, stop, deadline, backend_factory):
    backend = None
    try:
        # Three processes can initialize Warp for a PPO+NGOpt run (training,
        # held-out evaluation and design search). Never let them share compiler
        # cache files.
        root = Path(config["output_dir"])
        warp_cache = Path(os.environ.get("WARP_CACHE_PATH", root.parent / ".warp_cache"))
        os.environ["WARP_CACHE_PATH"] = str(warp_cache / "design_search")
        backend = (backend_factory or FrozenPolicySearchBackend)(config)
        events.put(("ready", None))
        while not stop.is_set():
            try:
                job = jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            generation, policy_bytes, designs = job
            value = backend.evaluate(policy_bytes, designs, stop, deadline)
            events.put(("result", (generation, value)))
    except Exception:
        events.put(("error", traceback.format_exc()))
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                events.put(("error", traceback.format_exc()))
        events.put(("done", None))


class AsyncDesignSearch:
    """One in-flight frozen-policy candidate batch in a spawned process."""

    def __init__(self, args, budget, *, backend_factory=None):
        context = mp.get_context("spawn")
        self.jobs = context.Queue(maxsize=1)
        self.events = context.Queue()
        self.stop = context.Event()
        self.error = None
        self.done = False
        self.closed = False
        self.pending = False
        self.generation = 0
        config = {**vars(args), "output_dir": str(args.output_dir)}
        self.process = context.Process(
            target=_worker,
            args=(config, self.jobs, self.events, self.stop,
                budget.started_at + budget.seconds, backend_factory),
            name="codesign-design-search")
        self.process.start()

    def submit(self, policy_bytes, designs):
        self.check()
        if self.pending:
            raise RuntimeError("a frozen-policy design search is already running")
        self.generation += 1
        self.pending = True
        self.jobs.put((self.generation, policy_bytes,
            np.asarray(designs, dtype=np.float32)))
        return self.generation

    def poll(self):
        result = None
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self.error = value
            elif kind == "done":
                self.done = True
            elif kind == "result":
                generation, payload = value
                if generation == self.generation:
                    self.pending = False
                    result = payload
        self.check()
        return result

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"independent design search failed:\n{self.error}")
        if not self.closed and self.process.exitcode not in (None, 0):
            raise RuntimeError(
                f"independent design search exited with code {self.process.exitcode}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        # Surface a failure unless shutdown deliberately interrupted an active
        # rollout, in which case SIGTERM is the expected bounded shutdown path.
        self.poll()
