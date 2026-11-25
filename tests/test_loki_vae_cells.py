"""Per-cell VAE tokens (SoftWalker-style sampler grid) and the on-device corpus cache used by distributed training."""

from gymnasium import spaces
import numpy as np
import torch

from loki.collectors.base import BaseCollector
from loki.train_vae import CorpusBatches, TrainingConfig, _rank_slice, run_epoch
from loki.vae import CorpusTokenSchema, TransformerVAE, VAEConfig

CELLS, WIDTH = 4, 3


class CellEnv:
    def __init__(self):
        low = np.tile(np.array([0.0, -1.0, 10.0]), CELLS)
        high = np.tile(np.array([1.0, 1.0, 20.0]), CELLS)
        self.embodiment_space = spaces.Box(low, high, dtype=np.float64)
        self.base_embodiment = (low + high) / 2

    def validate_embodiment(self, design):
        return self.embodiment_space.contains(design)

    def reset(self, *, seed, embodiment):
        return None, np.array([False]), {}

    def close(self):
        pass


class CellCollector(BaseCollector):
    presets = ("cells",)

    def sample_embodiment(self):
        return self.sample_native_space()

    def sampler_metadata(self):
        return {"grid": [2, 2], "channels": 2}

    def identity(self, audit_every):
        return {"collector": "cells", "seed": self.seed, "audit_every": audit_every, "sampler": self.sampler_metadata()}


def make_corpus(path, count=40):
    # shard_size 7 over 40 designs leaves a short final shard, as time-based flushes do in real corpora.
    assert CellCollector("cells", seed=3, env=CellEnv()).collect(path, count=count, shard_size=7, audit_every=0).complete


def test_sampler_grid_gives_one_token_per_cell(tmp_path):
    make_corpus(tmp_path)
    schema = CorpusTokenSchema.from_corpus(tmp_path)
    assert (schema.token_count, schema.feature_width) == (CELLS, WIDTH)
    assert [(group.count, group.width) for group in schema.groups] == [(CELLS, WIDTH)]


def test_device_cache_yields_the_streamed_batches(tmp_path):
    make_corpus(tmp_path)
    schema = CorpusTokenSchema.from_corpus(tmp_path)
    config = TrainingConfig(batch_size=9)
    streamed, cached = CorpusBatches(tmp_path, schema, config), CorpusBatches(tmp_path, schema, config)
    cached.cache_on_device(torch.device("cpu"))
    for split in ("train", "validation"):
        for epoch in (0, 3):
            a, b = list(streamed.iter_batches(split, epoch)), list(cached.iter_batches(split, epoch))
            assert len(a) == len(b)
            for x, y in zip(a, b):
                for key in x:
                    np.testing.assert_array_equal(x[key], y[key].numpy())
    # Rank slices partition every global batch in order.
    parts = [list(cached.iter_batches("train", 1, rank=r, world=3)) for r in range(3)]
    whole = list(cached.iter_batches("train", 1))
    for index, batch in enumerate(whole):
        torch.testing.assert_close(torch.cat([part[index]["continuous"] for part in parts]), batch["continuous"])


def test_rank_slice_skips_batches_smaller_than_world():
    rows = {"x": np.arange(2)}
    assert _rank_slice(rows, 0, 3) is None
    assert [len(_rank_slice({"x": np.arange(10)}, r, 3)["x"]) for r in range(3)] == [3, 3, 4]


def test_cached_epoch_trains(tmp_path):
    make_corpus(tmp_path)
    schema = CorpusTokenSchema.from_corpus(tmp_path)
    batches = CorpusBatches(tmp_path, schema, TrainingConfig(batch_size=8))
    batches.cache_on_device(torch.device("cpu"))
    model = TransformerVAE(schema, VAEConfig(layers=1, hidden_dim=8, heads=2))
    metrics = run_epoch(model, batches, torch.device("cpu"), 1e-2, optimizer=torch.optim.Adam(model.parameters()))
    assert metrics["samples"] == batches.training_count and np.isfinite(metrics["loss"])
