"""Regression for uint32 audit seeds overflowing BallCatcher's int32 tick."""

import copy
import json

import numpy as np
import pytest
import torch

from test_loki_collectors import StubCollector, StubEnv, hashes


class Int32ResetEnv(StubEnv):
    def __init__(self):
        super().__init__()
        self.seeds = []
        self.tick = torch.zeros(1, dtype=torch.int32)

    def reset(self, *, seed, embodiment):
        self.tick.fill_(int(seed) - 1)  # BallCatcher's seed=0 constructor.
        self.seeds.append(seed)
        return super().reset(seed=seed, embodiment=embodiment)


def test_audit_seed_is_deterministic_safe_and_independent():
    env = Int32ResetEnv()
    collector = StubCollector("stub", seed=0, env=env)
    state = copy.deepcopy(collector.rng.bit_generator.state)
    for design_id in range(0, 500_000, 10_000):
        assert collector.audit_reset(env.base_embodiment, design_id)
        first = env.seeds[-1]
        assert 0 <= first <= np.iinfo(np.int32).max
        assert collector.audit_reset(env.base_embodiment, design_id)
        assert env.seeds[-1] == first
    assert collector.rng.bit_generator.state == state


def test_resume_after_original_second_audit_overflow(tmp_path):
    collector = StubCollector("stub", seed=0, env=Int32ResetEnv())

    def old_audit(design, design_id):
        seed = int(np.random.SeedSequence([0, design_id, 0x10C1]).generate_state(1)[0])
        collector.env.reset(seed=seed, embodiment=design)
        return True

    collector.audit_reset = old_audit
    with pytest.raises(RuntimeError, match="overflow"):
        collector.collect(tmp_path, count=10_002, audit_every=10_000)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["state"]["stats"]["accepted"] == 10_000
    previous = hashes(tmp_path).copy()

    resumed = StubCollector("stub", seed=0, env=Int32ResetEnv())
    result = resumed.collect(tmp_path, count=10_002, audit_every=10_000, resume=True)
    assert result.complete
    assert result.stats["errors"] == 1  # Historical failure remains recorded.
    assert len(resumed.env.seeds) == 1
    np.testing.assert_array_equal(hashes(tmp_path)[:10_000], previous)
