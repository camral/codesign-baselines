"""Train the Codesign Gym Transformer VAE on one LOKI corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .collectors.storage import file_digest, write_json
from .vae import (CorpusTokenSchema, TransformerVAE, VAEConfig,
                  batch_to_device, vae_loss)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4096
    max_epochs: int = 200
    min_epochs: int = 50
    patience: int = 20
    min_delta: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_beta: float = 1e-2
    min_beta: float = 1e-5
    beta_decay: float = 0.7
    beta_patience: int = 10
    scheduler_factor: float = 0.95
    scheduler_patience: int = 10
    validation_interval: int = 10
    gradient_clip: float = 0.0
    validation_modulus: int = 10
    validation_remainder: int = 0
    seed: int = 42

    def validate(self):
        counts = (self.batch_size, self.max_epochs, self.min_epochs,
                  self.patience, self.beta_patience, self.scheduler_patience,
                  self.validation_interval,
                  self.validation_modulus)
        if min(counts) < 1:
            raise ValueError("Training counts must be positive")
        if not 0 <= self.validation_remainder < self.validation_modulus:
            raise ValueError("Invalid validation remainder")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration")
        if not 0 <= self.min_beta <= self.max_beta:
            raise ValueError("Require 0 <= min_beta <= max_beta")
        if not 0 < self.beta_decay <= 1:
            raise ValueError("beta_decay must be in (0, 1]")
        if not 0 < self.scheduler_factor < 1:
            raise ValueError("scheduler_factor must be in (0, 1)")
        if self.min_delta < 0 or self.gradient_clip < 0:
            raise ValueError("min_delta and gradient_clip must be nonnegative")


class CorpusBatches:
    """Deterministic shard streaming with a design-ID train/validation split."""

    def __init__(self, root, schema, config, *, allow_incomplete=False):
        self.root, self.schema, self.config = Path(root), schema, config
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported corpus format")
        status = self.manifest["status"]
        if status != "complete" and not allow_incomplete:
            raise ValueError(
                f"Corpus status is {status!r}; production VAE training requires complete")
        shard_count = int(self.manifest.get("shards", 0))
        self.paths = [self.root / "shards" / f"{index:06d}" / "data.npz"
                      for index in range(shard_count)]
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Corpus manifest references missing shard {missing[0]}")
        accepted = int(self.manifest.get("state", {}).get(
            "stats", {}).get("accepted", 0))
        if accepted < 2:
            raise ValueError("VAE training requires at least two corpus rows")
        ids = np.arange(accepted, dtype=np.int64)
        validation = ids % config.validation_modulus == config.validation_remainder
        self.validation_count = int(validation.sum())
        self.training_count = accepted - self.validation_count
        if min(self.training_count, self.validation_count) < 1:
            raise ValueError("Fixed split produced an empty train or validation partition")

    def cache_on_device(self, device, rank=0, world=1):
        """Encode the whole corpus once into ``device`` memory; each rank encodes every world-th shard and broadcasts
        it. ``iter_batches`` then yields exactly the streamed batches, gathered on the device."""
        import torch.distributed as dist
        accepted = self.training_count + self.validation_count
        # Shards flush on size or time, so read every shard's row count (on every rank, so all agree before any
        # collective; a rank raising mid-broadcast would deadlock the others).
        counts = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as shard:
                counts.append(len(shard["ids"]))
        if sum(counts) != accepted or min(counts) < 1:
            raise ValueError("Corpus shards do not add up to the manifest's accepted count")
        fields = tuple(dict.fromkeys(group.field for group in self.schema.groups))
        shape = {"continuous": (self.schema.token_count, self.schema.feature_width), "continuous_mask":
                 (self.schema.token_count, self.schema.feature_width), "categorical": (self.schema.token_count,),
                 "binary": (self.schema.token_count,)}
        dtype = {"continuous": torch.float32, "continuous_mask": torch.float32, "categorical": torch.int64,
                 "binary": torch.float32}
        self.cache = {key: torch.empty((accepted, *shape[key]), dtype=dtype[key], device=device) for key in shape}
        self.cache_ids = torch.empty(accepted, dtype=torch.int64, device=device)
        self.offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        self.counts = counts
        # Encode this rank's shards first (all ranks in parallel), then share them; broadcasting after every shard
        # would serialise the ranks on each owner's CPU work.
        for index, path in enumerate(self.paths):
            if index % world != rank:
                continue
            start, stop = int(self.offsets[index]), int(self.offsets[index]) + counts[index]
            with np.load(path, allow_pickle=False) as shard:
                encoded = self.schema.encode_arrays({field: shard[field] for field in fields})
                self.cache_ids[start:stop] = torch.as_tensor(shard["ids"], device=device)
            for key in shape:
                self.cache[key][start:stop] = torch.as_tensor(encoded[key], device=device)
        if world > 1:
            for index in range(len(self.paths)):
                start, stop = int(self.offsets[index]), int(self.offsets[index]) + counts[index]
                for key in shape:
                    dist.broadcast(self.cache[key][start:stop], src=index % world)
                dist.broadcast(self.cache_ids[start:stop], src=index % world)
        self.cache_ids_host = self.cache_ids.cpu().numpy()

    def _iter_cached(self, split, epoch, rank, world):
        # Same RNG calls, in the same order, as the streamed path: shard order, then rows within each shard.
        order = list(range(len(self.paths)))
        rng = np.random.default_rng(self.config.seed + int(epoch))
        if split == "train":
            rng.shuffle(order)
        pending = np.zeros(0, dtype=np.int64)
        for index in order:
            start = int(self.offsets[index])
            ids = self.cache_ids_host[start:start + self.counts[index]]
            validation = ids % self.config.validation_modulus == self.config.validation_remainder
            selected = np.flatnonzero(validation if split == "validation" else ~validation)
            if split == "train":
                rng.shuffle(selected)
            if not len(selected):
                continue
            pending = np.concatenate([pending, start + selected])
            while len(pending) >= self.config.batch_size:
                rows, pending = pending[:self.config.batch_size], pending[self.config.batch_size:]
                yield from self._gather(rows, rank, world)
        if len(pending):
            yield from self._gather(pending, rank, world)

    def _gather(self, rows, rank, world):
        local = _rank_slice({"rows": rows}, rank, world)
        if local is not None:
            index = torch.as_tensor(local["rows"], device=self.cache_ids.device)
            yield {key: value.index_select(0, index) for key, value in self.cache.items()}

    def iter_batches(self, split, epoch=0, rank=0, world=1):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        if getattr(self, "cache", None) is not None:
            yield from self._iter_cached(split, epoch, rank, world)
            return
        paths = list(self.paths)
        rng = np.random.default_rng(self.config.seed + int(epoch))
        if split == "train":
            rng.shuffle(paths)
        pending = None
        fields = tuple(dict.fromkeys(group.field for group in self.schema.groups))
        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                ids = shard["ids"]
                validation = (ids % self.config.validation_modulus
                              == self.config.validation_remainder)
                selected = np.flatnonzero(validation if split == "validation"
                                          else ~validation)
                if split == "train":
                    rng.shuffle(selected)
                if not len(selected):
                    continue
                current = {field: shard[field][selected] for field in fields}
            pending = (current if pending is None else
                       {field: np.concatenate([pending[field], current[field]])
                        for field in fields})
            while len(next(iter(pending.values()))) >= self.config.batch_size:
                take = {field: values[:self.config.batch_size]
                        for field, values in pending.items()}
                pending = {field: values[self.config.batch_size:]
                           for field, values in pending.items()}
                local = _rank_slice(take, rank, world)
                if local is not None:
                    yield self.schema.encode_arrays(local)
        if pending is not None and len(next(iter(pending.values()))):
            local = _rank_slice(pending, rank, world)
            if local is not None:
                yield self.schema.encode_arrays(local)


def _rank_slice(arrays, rank, world):
    """This rank's contiguous share of one global batch; None (on every rank) if the batch is smaller than world."""
    count = len(next(iter(arrays.values())))
    if world == 1:
        return arrays
    if count < world:
        return None
    start, stop = rank * count // world, (rank + 1) * count // world
    return {field: values[start:stop] for field, values in arrays.items()}


class MetricAverage:
    def __init__(self):
        self.total, self.count = {}, 0
        self.latent_sum = self.latent_square_sum = 0.0
        self.latent_count = 0
        self.posterior_std_sum = 0.0

    def add(self, metrics, mean, logvar, batch_size):
        values = {name: getattr(metrics, name) for name in (
            "loss", "reconstruction", "continuous_mse", "categorical_loss",
            "binary_loss", "kl", "categorical_accuracy", "binary_accuracy")}
        for name, value in values.items():
            self.total[name] = self.total.get(name, 0.0) + float(value.detach()) * batch_size
        self.count += batch_size
        latent = mean.detach().float()
        self.latent_sum += float(latent.sum())
        self.latent_square_sum += float(latent.square().sum())
        self.latent_count += latent.numel()
        self.posterior_std_sum += float(
            torch.exp(0.5 * logvar.detach()).mean()) * batch_size

    def all_reduce(self, device):
        """Sum every statistic over the distributed ranks (identical results on all of them)."""
        import torch.distributed as dist
        names = sorted(self.total)
        values = torch.tensor([self.total[name] for name in names] + [
            self.count, self.latent_sum, self.latent_square_sum, self.latent_count, self.posterior_std_sum],
            dtype=torch.float64, device=device)
        dist.all_reduce(values)
        values = values.tolist()
        self.total = dict(zip(names, values[:len(names)]))
        self.count, self.latent_sum, self.latent_square_sum, self.latent_count, self.posterior_std_sum = values[len(names):]
        self.count, self.latent_count = int(self.count), int(self.latent_count)

    def result(self):
        if not self.count:
            raise RuntimeError("No batches were produced")
        result = {name: value / self.count for name, value in self.total.items()}
        latent_mean = self.latent_sum / self.latent_count
        variance = max(self.latent_square_sum / self.latent_count
                       - latent_mean ** 2, 0.0)
        result.update(latent_mean=latent_mean, latent_std=math.sqrt(variance),
                      posterior_std_mean=self.posterior_std_sum / self.count,
                      samples=self.count)
        return result


def run_epoch(model, batches, device, beta, *, optimizer=None,
              gradient_clip=1.0, epoch=0, network=None, rank=0, world=1):
    """``network`` is the DDP wrapper of ``model`` when training is distributed (gradients averaged over ranks)."""
    network = model if network is None else network
    training = optimizer is not None
    network.train(training)
    average = MetricAverage()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        split = "train" if training else "validation"
        for numpy_batch in batches.iter_batches(split, epoch=epoch, rank=rank, world=world):
            batch = batch_to_device(numpy_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Released LOKI samples from q(z|x) during both training and
            # validation rather than decoding the posterior mean for validation.
            outputs, mean, logvar = network(batch, sample=True)
            metrics = vae_loss(model, batch, outputs, mean, logvar, beta)
            if training:
                metrics.loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            average.add(metrics, mean, logvar, len(batch["continuous"]))
    if world > 1:
        average.all_reduce(device)
    return average.result()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed, *, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def rng_state(*, use_cuda=False):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if use_cuda else None}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def encoder_state(model):
    prefixes = ("continuous_input.", "category_input.", "binary_input.",
                "kind_embedding.", "position", "encoder_mu.", "token_kinds",
                "category_cardinalities")
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if key.startswith(prefixes)}


def compatible_training_config(value):
    value = dict(value)
    value.pop("max_epochs", None)
    return value


def train(corpus, output, *, model_config=VAEConfig(),
          training_config=TrainingConfig(), device="auto", resume=False,
          allow_incomplete=False, distributed=False, cache_on_device=False):
    """``distributed``: one torchrun rank per GPU; each global batch of ``batch_size`` is split across the ranks and
    gradients are averaged, so the optimisation is the single-GPU recipe's. Only rank 0 writes files."""
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    model_config.validate()
    training_config.validate()
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rank, world = 0, 1
    if distributed:
        import torch.distributed as dist
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    primary = rank == 0
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    schema = CorpusTokenSchema.from_corpus(corpus)
    batches = CorpusBatches(corpus, schema, training_config,
                            allow_incomplete=allow_incomplete)
    corpus_identity = {
        "manifest_sha256": file_digest(corpus / "manifest.json"),
        "constants_sha256": file_digest(corpus / "constants.npz"),
        "accepted": batches.training_count + batches.validation_count,
        "status": batches.manifest["status"],
    }
    configuration = {
        "format_version": 1,
        "upstream_loki_commit": "24aaac7eb2ecd8f1aada1e1ed0e2d91577fb578a",
        "corpus": str(corpus), "corpus_identity": corpus_identity,
        "token_schema": schema.to_dict(), "model": asdict(model_config),
        "training": asdict(training_config),
        "split": {
            "rule": "design_id modulo validation_modulus equals validation_remainder",
            "training_count": batches.training_count,
            "validation_count": batches.validation_count,
        },
        "allow_incomplete": bool(allow_incomplete),
    }

    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Training output already exists: {output}; use --resume")
    if primary:
        output.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    config_path, last_path = output / "config.json", output / "last.pt"
    best_path, encoder_path = output / "best.pt", output / "encoder.pt"
    history_path = output / "history.json"
    report_path = output / "convergence.json"

    seed_everything(training_config.seed, use_cuda=torch_device.type == "cuda")
    if distributed or cache_on_device:
        batches.cache_on_device(torch_device, rank, world)
    model = TransformerVAE(schema, model_config).to(torch_device)
    network = model
    if distributed:
        # Categorical/binary heads get no gradient on all-continuous corpora (e.g. SoftWalker).
        network = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch_device.index], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate,
                                 weight_decay=training_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=training_config.scheduler_factor,
        patience=training_config.scheduler_patience)
    start_epoch, beta = 0, training_config.max_beta
    best_reconstruction, best_epoch = math.inf, None
    no_improvement, history, elapsed_before = 0, [], 0.0

    if resume:
        if not config_path.is_file() or not last_path.is_file():
            raise FileNotFoundError("Resume requires config.json and last.pt")
        previous = json.loads(config_path.read_text())
        for key in ("format_version", "upstream_loki_commit", "corpus_identity",
                    "token_schema", "model", "split", "allow_incomplete"):
            if previous[key] != configuration[key]:
                raise ValueError(f"Resume configuration differs in {key}")
        if (compatible_training_config(previous["training"])
                != compatible_training_config(configuration["training"])):
            raise ValueError("Resume training hyperparameters differ")
        checkpoint = torch.load(last_path, map_location=torch_device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["next_epoch"])
        beta = float(checkpoint["beta"])
        best_reconstruction = float(checkpoint["best_reconstruction"])
        best_epoch = checkpoint["best_epoch"]
        no_improvement = int(checkpoint["no_improvement"])
        history = checkpoint["history"]
        elapsed_before = float(checkpoint["elapsed_seconds"])
        restore_rng_state(checkpoint["rng"])
        if start_epoch >= training_config.max_epochs:
            raise ValueError("max_epochs must exceed the resumed epoch")
    if primary:
        write_json(config_path, configuration)

    started, converged, stop_reason = time.monotonic(), False, "fixed_epoch_budget"
    for epoch in range(start_epoch, training_config.max_epochs):
        train_metrics = run_epoch(
            model, batches, torch_device, beta, optimizer=optimizer,
            gradient_clip=training_config.gradient_clip, epoch=epoch, network=network, rank=rank, world=world)
        should_validate = epoch % training_config.validation_interval == 0
        validation_metrics, improved = None, None
        if should_validate:
            validation_metrics = run_epoch(
                model, batches, torch_device, beta, epoch=epoch, network=network, rank=rank, world=world)
            scheduler.step(validation_metrics["reconstruction"])
            improved = (best_reconstruction - validation_metrics["reconstruction"]
                        > training_config.min_delta)
            if improved:
                best_reconstruction = validation_metrics["reconstruction"]
                best_epoch, no_improvement = epoch, 0
            if improved and primary:
                atomic_torch_save({
                    "format_version": 1, "epoch": epoch,
                    "model": model.state_dict(),
                    "model_config": asdict(model_config),
                    "token_schema": schema.to_dict(),
                    "validation": validation_metrics,
                    "corpus_identity": corpus_identity,
                }, best_path)
                atomic_torch_save({
                    "format_version": 1, "epoch": epoch,
                    "state": encoder_state(model),
                    "model_config": asdict(model_config),
                    "token_schema": schema.to_dict(),
                    "corpus_identity": corpus_identity,
                }, encoder_path)
            if not improved:
                no_improvement += 1
                if (no_improvement == training_config.beta_patience
                        and beta > training_config.min_beta):
                    beta = max(training_config.min_beta,
                               beta * training_config.beta_decay)

        record = {
            "epoch": epoch, "beta": beta,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved, "no_improvement": no_improvement,
            "train": train_metrics, "validation": validation_metrics,
        }
        history.append(record)
        elapsed = elapsed_before + time.monotonic() - started
        if primary:
            atomic_torch_save({
                "format_version": 1, "next_epoch": epoch + 1,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "beta": beta,
                "best_reconstruction": best_reconstruction,
                "best_epoch": best_epoch, "no_improvement": no_improvement,
                "history": history, "elapsed_seconds": elapsed, "rng": rng_state(use_cuda=torch_device.type == "cuda"),
            }, last_path)
            write_json(history_path, history)
            print(json.dumps(record, sort_keys=True), flush=True)
        if (should_validate and epoch + 1 >= training_config.min_epochs
                and no_improvement >= training_config.patience):
            converged, stop_reason = True, "validation_plateau"
            break

    elapsed = elapsed_before + time.monotonic() - started
    fixed_recipe_complete = len(history) >= training_config.max_epochs
    report = {
        "format_version": 1, "complete": bool(converged or fixed_recipe_complete),
        "converged": converged,
        "fixed_epoch_recipe": fixed_recipe_complete,
        "stop_reason": stop_reason, "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_reconstruction": best_reconstruction,
        "final_beta": beta, "elapsed_seconds": elapsed,
        "training_count": batches.training_count,
        "validation_count": batches.validation_count,
        "token_count": schema.token_count,
        "latent_shape_per_design": [schema.token_count, model_config.hidden_dim],
        "best_checkpoint_sha256": (file_digest(best_path)
                                   if best_path.exists() else None),
        "encoder_checkpoint_sha256": (file_digest(encoder_path)
                                      if encoder_path.exists() else None),
    }
    if primary:
        write_json(report_path, report)
    if distributed:
        dist.barrier()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--distributed", action="store_true",
                        help="Data-parallel over torchrun ranks (one GPU each); --batch-size stays the global batch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Testing only: snapshot the currently committed shards")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-beta", type=float, default=1e-2)
    parser.add_argument("--min-beta", type=float, default=1e-5)
    parser.add_argument("--beta-decay", type=float, default=0.7)
    parser.add_argument("--beta-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.95)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--validation-modulus", type=int, default=10)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-factor", type=int, default=8)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument("--ffn-dropout", type=float, default=0.0)
    parser.add_argument("--positional-dropout", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_config = VAEConfig(
        layers=args.layers, hidden_dim=args.hidden_dim, heads=args.heads,
        ffn_factor=args.ffn_factor, attention_dropout=args.attention_dropout,
        ffn_dropout=args.ffn_dropout,
        positional_dropout=args.positional_dropout)
    training_config = TrainingConfig(
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        min_epochs=args.min_epochs, patience=args.patience,
        min_delta=args.min_delta, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, max_beta=args.max_beta,
        min_beta=args.min_beta, beta_decay=args.beta_decay,
        beta_patience=args.beta_patience,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        validation_interval=args.validation_interval,
        gradient_clip=args.gradient_clip,
        validation_modulus=args.validation_modulus,
        validation_remainder=args.validation_remainder, seed=args.seed)
    if args.distributed:
        import torch.distributed as dist
        dist.init_process_group("nccl")
    try:
        report = train(args.corpus, args.output, model_config=model_config,
                       training_config=training_config, device=args.device,
                       resume=args.resume, allow_incomplete=args.allow_incomplete, distributed=args.distributed)
    finally:
        if args.distributed:
            dist.destroy_process_group()
    if not args.distributed or int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
