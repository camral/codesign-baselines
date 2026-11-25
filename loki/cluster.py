"""Freeze LOKI posterior codes and form fixed embodiment clusters.

The representation and estimator follow upstream ``vae/latent_cluster.py``:
sample one latent per VAE token, flatten ``[tokens, hidden]``, then fit sklearn
``KMeans(n_clusters=..., random_state=...)``.  This port streams Codesign Gym
corpus shards into a disk-backed array so the full 500K corpus need not reside
in accelerator memory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from sklearn.cluster import KMeans
import torch

from .collectors.storage import CorpusReader, file_digest, write_json
from .vae import CorpusTokenSchema, TransformerVAE, VAEConfig, batch_to_device


UPSTREAM_LOKI_COMMIT = "24aaac7eb2ecd8f1aada1e1ed0e2d91577fb578a"


@dataclass(frozen=True)
class ClusteringConfig:
    n_clusters: int = 6
    batch_size: int = 4096
    posterior_seed: int = 0
    kmeans_seed: int = 0
    min_cluster_size: int = 128
    save_latents: bool = False

    def validate(self):
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least two")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be positive")


def _resolve_device(device):
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return resolved


def _corpus_identity(corpus, manifest):
    accepted = int(manifest.get("state", {}).get("stats", {}).get("accepted", 0))
    return {
        "manifest_sha256": file_digest(corpus / "manifest.json"),
        "constants_sha256": file_digest(corpus / "constants.npz"),
        "accepted": accepted,
        "status": manifest["status"],
    }


def _load_model(corpus, vae_dir, device, *, allow_unconverged=False):
    report_path = vae_dir / "convergence.json"
    checkpoint_path = vae_dir / "best.pt"
    if not report_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Clustering requires convergence.json and best.pt")
    report = json.loads(report_path.read_text())
    training_complete = report.get("complete", report.get("converged", False))
    if not training_complete and not allow_unconverged:
        raise ValueError("Production clustering requires a completed VAE recipe")
    expected_hash = report.get("best_checkpoint_sha256")
    actual_hash = file_digest(checkpoint_path)
    if expected_hash != actual_hash:
        raise ValueError("best.pt does not match convergence.json")

    reader = CorpusReader(corpus)
    identity = _corpus_identity(corpus, reader.manifest)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported VAE checkpoint format")
    if checkpoint.get("corpus_identity") != identity:
        raise ValueError("VAE checkpoint was trained on a different corpus snapshot")
    schema = CorpusTokenSchema.from_corpus(corpus)
    if checkpoint.get("token_schema") != schema.to_dict():
        raise ValueError("VAE token schema differs from the corpus")
    model_config = VAEConfig(**checkpoint["model_config"])
    model = TransformerVAE(schema, model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return reader, schema, model, model_config, identity, checkpoint_path, actual_hash, report


def _extract_latents(reader, schema, model, device, path, count, config):
    latent_shape = (schema.token_count, model.config.hidden_dim)
    flat_width = math.prod(latent_shape)
    latents = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(count, flat_width))
    fields = tuple(dict.fromkeys(group.field for group in schema.groups))
    generator = torch.Generator(device=device)
    generator.manual_seed(config.posterior_seed)
    cursor = 0
    with torch.inference_mode():
        for shard in reader.iter_shards():
            ids = shard["ids"]
            for start in range(0, len(ids), config.batch_size):
                stop = min(start + config.batch_size, len(ids))
                expected = np.arange(cursor, cursor + stop - start, dtype=np.int64)
                if not np.array_equal(ids[start:stop], expected):
                    raise ValueError("Corpus design IDs are not contiguous during extraction")
                arrays = {field: shard[field][start:stop] for field in fields}
                batch = batch_to_device(schema.encode_arrays(arrays), device)
                mean, logvar = model.posterior(batch)
                # Literal upstream convention: sample z and flatten all token codes.
                latent = model.reparameterize(mean, logvar, generator=generator)
                values = latent.reshape(len(expected), flat_width).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("VAE produced non-finite posterior codes")
                latents[cursor:cursor + len(expected)] = values
                cursor += len(expected)
    if cursor != count:
        raise ValueError(f"Corpus contains {cursor} rows but manifest records {count}")
    latents.flush()
    return latents, latent_shape


def _cluster_diagnostics(latents, labels, centers, n_clusters):
    sizes = np.bincount(labels, minlength=n_clusters).astype(np.int64)
    # Avoid a second full-size temporary while still catching exact collapse.
    sample_count = min(len(latents), 10_000)
    indices = np.linspace(0, len(latents) - 1, sample_count, dtype=np.int64)
    sample = np.asarray(latents[indices], dtype=np.float32)
    feature_std = sample.std(axis=0, dtype=np.float64)
    return {
        "cluster_sizes": sizes.tolist(),
        "empty_clusters": np.flatnonzero(sizes == 0).astype(int).tolist(),
        "sampled_rows_for_code_diagnostics": sample_count,
        "sample_code_std": float(sample.std(dtype=np.float64)),
        "constant_latent_dimensions": int(np.count_nonzero(feature_std == 0)),
        "latent_dimensions": int(latents.shape[1]),
        "finite_centers": bool(np.isfinite(centers).all()),
    }


def cluster(corpus, vae, output, *, config=ClusteringConfig(), device="auto",
            allow_incomplete=False, allow_unconverged=False):
    """Extract one frozen posterior sample per design and fit K-means."""
    started = time.monotonic()
    config.validate()
    corpus, vae, output = Path(corpus).resolve(), Path(vae).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Clustering output already exists: {output}")
    torch_device = _resolve_device(device)
    reader, schema, model, model_config, identity, checkpoint_path, checkpoint_hash, convergence = _load_model(
        corpus, vae, torch_device, allow_unconverged=allow_unconverged)
    if identity["status"] != "complete" and not allow_incomplete:
        raise ValueError("Production clustering requires a complete corpus")
    count = identity["accepted"]
    if count < config.n_clusters:
        raise ValueError("Corpus has fewer designs than requested clusters")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary)
        latent_path = staging / "latents.npy"
        latents, latent_shape = _extract_latents(
            reader, schema, model, torch_device, latent_path, count, config)
        extraction_seconds = time.monotonic() - started

        # Keep the released LOKI estimator call and defaults intact.
        kmeans = KMeans(
            n_clusters=config.n_clusters,
            random_state=config.kmeans_seed).fit(latents)
        labels = np.asarray(kmeans.labels_, dtype=np.int32)
        centers_flat = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        centers = centers_flat.reshape(config.n_clusters, *latent_shape)
        diagnostics = _cluster_diagnostics(
            latents, labels, centers_flat, config.n_clusters)
        too_small = [index for index, size in enumerate(diagnostics["cluster_sizes"])
                     if size < config.min_cluster_size]
        diagnostics["clusters_below_minimum"] = too_small
        diagnostics["minimum_cluster_size"] = config.min_cluster_size
        if diagnostics["empty_clusters"] or not diagnostics["finite_centers"]:
            raise ValueError(f"Invalid K-means result: {diagnostics}")
        if diagnostics["sample_code_std"] == 0:
            raise ValueError("All sampled posterior codes collapsed to one value")
        if too_small:
            raise ValueError(
                f"Clusters {too_small} contain fewer than {config.min_cluster_size} designs")

        design_ids = np.arange(count, dtype=np.int64)
        np.save(staging / "design_ids.npy", design_ids, allow_pickle=False)
        np.save(staging / "labels.npy", labels, allow_pickle=False)
        np.save(staging / "centers.npy", centers, allow_pickle=False)
        torch.save(torch.from_numpy(centers),
                   staging / f"cluster{config.n_clusters}_means.pt")
        clusters_dir = staging / "clusters"
        clusters_dir.mkdir()
        for index in range(config.n_clusters):
            np.save(clusters_dir / f"{index}.npy", design_ids[labels == index],
                    allow_pickle=False)
        write_json(staging / f"cluster{config.n_clusters}_label_distribution.json",
                   {str(index): size for index, size in
                    enumerate(diagnostics["cluster_sizes"])})

        if not config.save_latents:
            del latents
            latent_path.unlink()
        elapsed = time.monotonic() - started
        manifest = {
            "format_version": 1,
            "status": "complete",
            "upstream_loki_commit": UPSTREAM_LOKI_COMMIT,
            "upstream_source": "vae/latent_cluster.py",
            "corpus": str(corpus),
            "corpus_identity": identity,
            "vae": str(vae),
            "vae_checkpoint": str(checkpoint_path),
            "vae_checkpoint_sha256": checkpoint_hash,
            "vae_best_epoch": convergence.get("best_epoch"),
            "vae_converged": bool(convergence.get("converged", False)),
            "vae_training_complete": bool(
                convergence.get("complete", convergence.get("converged", False))),
            "vae_fixed_epoch_recipe": bool(
                convergence.get("fixed_epoch_recipe", False)),
            "vae_epochs_completed": convergence.get("epochs_completed"),
            "token_schema": schema.to_dict(),
            "model_config": asdict(model_config),
            "clustering_config": asdict(config),
            "latent_shape_per_design": list(latent_shape),
            "flattened_latent_width": int(math.prod(latent_shape)),
            "design_count": count,
            "kmeans": {
                "class": "sklearn.cluster.KMeans",
                "parameters": kmeans.get_params(deep=False),
                "inertia": float(kmeans.inertia_),
                "iterations": int(kmeans.n_iter_),
            },
            "diagnostics": diagnostics,
            "extraction_seconds": extraction_seconds,
            "elapsed_seconds": elapsed,
            "artifacts": {
                "design_ids": "design_ids.npy",
                "labels": "labels.npy",
                "centers": "centers.npy",
                "upstream_compatible_centers": f"cluster{config.n_clusters}_means.pt",
                "cluster_indexes": "clusters/{cluster_id}.npy",
                "latents": "latents.npy" if config.save_latents else None,
            },
        }
        write_json(staging / "manifest.json", manifest)
        # TemporaryDirectory owns the staging parent, so move its contents as
        # one directory only after every artifact is complete.
        shutil.move(str(staging), str(output))
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True,
                        help="Directory containing best.pt and convergence.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--posterior-seed", type=int, default=0)
    parser.add_argument("--kmeans-seed", type=int, default=0)
    parser.add_argument("--min-cluster-size", type=int, default=128)
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Testing only: accept a corpus snapshot not marked complete")
    parser.add_argument("--allow-unconverged", action="store_true",
                        help="Testing only: cluster a VAE not marked converged")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = ClusteringConfig(
        n_clusters=args.n_clusters, batch_size=args.batch_size,
        posterior_seed=args.posterior_seed, kmeans_seed=args.kmeans_seed,
        min_cluster_size=args.min_cluster_size,
        save_latents=args.save_latents)
    result = cluster(
        args.corpus, args.vae, args.output, config=config, device=args.device,
        allow_incomplete=args.allow_incomplete,
        allow_unconverged=args.allow_unconverged)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
