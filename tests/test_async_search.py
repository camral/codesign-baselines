import time
from types import SimpleNamespace

import numpy as np

from common.wall_clock import WallClockBudget
from ppo_ngopt_rlgo.async_search import AsyncDesignSearch


class FakeSearchBackend:
    def __init__(self, config):
        self.scale = config["seed"]

    def evaluate(self, policy_bytes, designs, stop, deadline):
        assert policy_bytes == b"frozen-policy"
        return designs.sum(axis=1) + self.scale, np.zeros(len(designs), bool), 17

    def close(self):
        pass


def test_async_design_search_returns_tagged_frozen_policy_result(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, device="cpu", env_id="test",
        num_envs=2, seed=3, rejection_reward=-10)
    budget = WallClockBudget.start(1)
    search = AsyncDesignSearch(args, budget, backend_factory=FakeSearchBackend)
    try:
        designs = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        assert search.submit(b"frozen-policy", designs) == 1
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            result = search.poll()
            time.sleep(0.01)
        assert result is not None
        returns, rejected, frames = result
        np.testing.assert_equal(returns, [6, 10])
        np.testing.assert_equal(rejected, [False, False])
        assert frames == 17 and not search.pending
    finally:
        search.close()
