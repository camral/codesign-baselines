"""Transformer VAE tokenization, training, artifacts, and resume contracts."""

import json
from dataclasses import replace

from gymnasium import spaces
import numpy as np
import pytest
import torch

from loki.collectors.base import BaseCollector
from loki.collectors.storage import CorpusReader
from loki.train_vae import CorpusBatches, TrainingConfig, run_epoch, train
from loki.vae import (BINARY, CATEGORICAL, CONTINUOUS, CorpusTokenSchema,
                      TransformerVAE, VAEConfig, batch_to_device, vae_loss)


class TypedEnv:
    def __init__(self):
        self.embodiment_space = spaces.Dict({
            "binary": spaces.MultiBinary(3),
            "category": spaces.Discrete(3, start=2),
            "continuous": spaces.Box(
                np.array([-2.0, 10.0], dtype=np.float32),
                np.array([2.0, 20.0], dtype=np.float32)),
        })
        self.base_embodiment = {
            "binary": np.zeros(3, dtype=np.int8),
            "category": 2,
            "continuous": np.array([0.0, 15.0], dtype=np.float32),
        }

    def validate_embodiment(self, design):
        return self.embodiment_space.contains(design)

    def reset(self, *, seed, embodiment):
        return None, np.array([False]), {}

    def close(self):
        pass


class TypedCollector(BaseCollector):
    presets = ("typed",)

    def sample_embodiment(self):
        return self.sample_native_space()

    def identity(self, audit_every):
        return {"collector": "typed", "seed": self.seed,
                "audit_every": audit_every}


def make_corpus(path, count=40):
    result = TypedCollector("typed", seed=7, env=TypedEnv()).collect(
        path, count=count, shard_size=7, audit_every=0)
    assert result.complete


def test_typed_schema_and_loss_cover_all_native_value_kinds(tmp_path):
    make_corpus(tmp_path)
    schema = CorpusTokenSchema.from_corpus(tmp_path)
    assert schema.token_count == 6
    assert list(schema.kinds).count(CONTINUOUS) == 2
    assert list(schema.kinds).count(CATEGORICAL) == 1
    assert list(schema.kinds).count(BINARY) == 3
    shard = next(CorpusReader(tmp_path).iter_shards())
    fields = {name: shard[name] for name in schema.corpus_schema["fields"]}
    batch = batch_to_device(schema.encode_arrays(fields), "cpu")
    assert torch.all(batch["continuous"].abs() <= 1.000001)
    model = TransformerVAE(
        schema, VAEConfig(layers=1, hidden_dim=8, heads=2, ffn_factor=2,
                          positional_dropout=0.0))
    outputs, mean, logvar = model(batch)
    metrics = vae_loss(model, batch, outputs, mean, logvar, beta=0.01)
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()


def test_training_writes_converged_reloadable_artifacts(tmp_path):
    corpus, output = tmp_path / "corpus", tmp_path / "vae"
    make_corpus(corpus)
    model_config = VAEConfig(
        layers=1, hidden_dim=8, heads=2, ffn_factor=2,
        positional_dropout=0.0)
    training_config = TrainingConfig(
        batch_size=8, max_epochs=1, min_epochs=2, patience=1,
        min_delta=1e9, beta_patience=1, scheduler_patience=1,
        validation_interval=1, seed=3)
    first = train(corpus, output, model_config=model_config,
                  training_config=training_config, device="cpu")
    assert not first["converged"]
    report = train(corpus, output, model_config=model_config,
                   training_config=replace(training_config, max_epochs=2),
                   device="cpu", resume=True)
    assert report["converged"]
    assert report["stop_reason"] == "validation_plateau"
    assert report["epochs_completed"] == 2
    for name in ("config.json", "best.pt", "encoder.pt", "last.pt",
                 "history.json", "convergence.json"):
        assert (output / name).is_file()
    checkpoint = torch.load(output / "best.pt", map_location="cpu",
                            weights_only=False)
    schema = CorpusTokenSchema.from_corpus(corpus)
    restored = TransformerVAE(schema, model_config)
    restored.load_state_dict(checkpoint["model"])


def test_incomplete_corpus_requires_explicit_testing_override(tmp_path):
    make_corpus(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    schema = CorpusTokenSchema.from_corpus(tmp_path)
    config = TrainingConfig(batch_size=8, max_epochs=1, min_epochs=1)
    with pytest.raises(ValueError, match="requires complete"):
        CorpusBatches(tmp_path, schema, config)
    batches = CorpusBatches(tmp_path, schema, config, allow_incomplete=True)
    assert batches.training_count + batches.validation_count == 40


def test_released_recipe_defaults():
    config = TrainingConfig()
    assert config.batch_size == 4096
    assert config.max_epochs == 200
    assert config.min_delta == 0.0
    assert config.gradient_clip == 0.0
    assert config.validation_interval == 10


def test_validation_samples_posterior(tmp_path):
    make_corpus(tmp_path)
    schema = CorpusTokenSchema.from_corpus(tmp_path)
    config = TrainingConfig(batch_size=8)
    batches = CorpusBatches(tmp_path, schema, config)
    model = TransformerVAE(
        schema, VAEConfig(layers=1, hidden_dim=8, heads=2, ffn_factor=2,
                          positional_dropout=0.0))
    samples = []
    original_forward = model.forward

    def recording_forward(batch, *, sample=True, generator=None):
        samples.append(sample)
        return original_forward(batch, sample=sample, generator=generator)

    model.forward = recording_forward
    run_epoch(model, batches, "cpu", beta=0.01)
    assert samples
    assert all(samples)
