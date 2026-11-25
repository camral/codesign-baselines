"""Frozen posterior extraction and six-cluster artifact contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from loki.cluster import ClusteringConfig, cluster
from loki.train_vae import TrainingConfig, train
from loki.vae import VAEConfig

from test_loki_vae import make_corpus


def make_vae(corpus, output):
    model = VAEConfig(layers=1, hidden_dim=8, heads=2, ffn_factor=2,
                      positional_dropout=0.0)
    training = TrainingConfig(
        batch_size=8, max_epochs=1, min_epochs=2, patience=1,
        min_delta=1e9, beta_patience=1, scheduler_patience=1,
        validation_interval=1, seed=3)
    train(corpus, output, model_config=model, training_config=training,
          device="cpu")
    report = train(
        corpus, output, model_config=model,
        training_config=replace(training, max_epochs=2),
        device="cpu", resume=True)
    assert report["converged"]


def test_six_clusters_cover_every_design_and_are_reproducible(tmp_path):
    corpus, vae = tmp_path / "corpus", tmp_path / "vae"
    make_corpus(corpus, count=40)
    make_vae(corpus, vae)
    config = ClusteringConfig(
        n_clusters=6, batch_size=7, posterior_seed=11, kmeans_seed=13,
        min_cluster_size=1, save_latents=True)
    first = cluster(corpus, vae, tmp_path / "clusters-a", config=config,
                    device="cpu")
    second = cluster(corpus, vae, tmp_path / "clusters-b", config=config,
                     device="cpu")

    assert first["design_count"] == 40
    assert first["latent_shape_per_design"] == [6, 8]
    assert sum(first["diagnostics"]["cluster_sizes"]) == 40
    assert not first["diagnostics"]["empty_clusters"]
    labels_a = np.load(tmp_path / "clusters-a" / "labels.npy")
    labels_b = np.load(tmp_path / "clusters-b" / "labels.npy")
    latents_a = np.load(tmp_path / "clusters-a" / "latents.npy")
    latents_b = np.load(tmp_path / "clusters-b" / "latents.npy")
    np.testing.assert_array_equal(labels_a, labels_b)
    np.testing.assert_array_equal(latents_a, latents_b)
    members = [np.load(tmp_path / "clusters-a" / "clusters" / f"{i}.npy")
               for i in range(6)]
    np.testing.assert_array_equal(np.sort(np.concatenate(members)), np.arange(40))
    assert np.load(tmp_path / "clusters-a" / "centers.npy").shape == (6, 6, 8)


def test_posterior_seed_changes_frozen_codes(tmp_path):
    corpus, vae = tmp_path / "corpus", tmp_path / "vae"
    make_corpus(corpus, count=40)
    make_vae(corpus, vae)
    base = ClusteringConfig(n_clusters=6, batch_size=8, posterior_seed=1,
                            kmeans_seed=0, min_cluster_size=1,
                            save_latents=True)
    cluster(corpus, vae, tmp_path / "one", config=base, device="cpu")
    cluster(corpus, vae, tmp_path / "two",
            config=replace(base, posterior_seed=2), device="cpu")
    one = np.load(tmp_path / "one" / "latents.npy")
    two = np.load(tmp_path / "two" / "latents.npy")
    assert not np.array_equal(one, two)


def test_clustering_rejects_changed_corpus_snapshot(tmp_path):
    corpus, vae = tmp_path / "corpus", tmp_path / "vae"
    make_corpus(corpus, count=40)
    make_vae(corpus, vae)
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different corpus snapshot"):
        cluster(corpus, vae, tmp_path / "clusters",
                config=ClusteringConfig(min_cluster_size=1), device="cpu",
                allow_incomplete=True)


def test_clustering_requires_converged_report(tmp_path):
    corpus, vae = tmp_path / "corpus", tmp_path / "vae"
    make_corpus(corpus, count=40)
    make_vae(corpus, vae)
    report_path = vae / "convergence.json"
    report = json.loads(report_path.read_text())
    report["converged"] = False
    report["complete"] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="requires a completed VAE recipe"):
        cluster(corpus, vae, tmp_path / "clusters",
                config=ClusteringConfig(min_cluster_size=1), device="cpu")
