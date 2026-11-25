"""Typed embodiment tokens and the Transformer VAE used by the LOKI port.

The architecture follows the released LOKI VAE at commit
24aaac7eb2ecd8f1aada1e1ed0e2d91577fb578a: independent Transformer stacks
for posterior means and log variances, sampled per-token latents, and a causal
Transformer decoder. The original UNIMAL-only limb representation is replaced
by a versioned mapping from the lossless Codesign Gym corpus schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .collectors.storage import CorpusReader, json_value


CONTINUOUS = 0
CATEGORICAL = 1
BINARY = 2
_KIND_NAMES = {CONTINUOUS: "continuous", CATEGORICAL: "categorical", BINARY: "binary"}


@dataclass(frozen=True)
class TokenGroup:
    field: str
    kind: int
    count: int
    width: int
    shape: tuple[int, ...]
    labels: tuple[str, ...]
    low: tuple[float | int, ...]
    high: tuple[float | int, ...]

    def to_dict(self):
        value = asdict(self)
        value["kind"] = _KIND_NAMES[self.kind]
        return json_value(value)


class CorpusTokenSchema:
    """Fixed typed tokens derived from one lossless corpus schema."""

    version = 1

    def __init__(self, groups, *, corpus_schema, corpus_constants_digest):
        self.groups = tuple(groups)
        self.corpus_schema = corpus_schema
        self.corpus_constants_digest = corpus_constants_digest
        self.token_count = sum(group.count for group in self.groups)
        self.feature_width = max(
            (group.width for group in self.groups if group.kind == CONTINUOUS),
            default=1)
        self.max_categories = max(
            (int(max(group.high)) for group in self.groups
             if group.kind == CATEGORICAL), default=1)
        self.max_categories = max(self.max_categories, 2)
        kinds, cardinalities, labels = [], [], []
        for group in self.groups:
            kinds.extend([group.kind] * group.count)
            labels.extend(group.labels)
            cardinalities.extend(
                (int(value) for value in group.high) if group.kind == CATEGORICAL
                else ([0] * group.count))
        self.kinds = np.asarray(kinds, dtype=np.int64)
        self.category_cardinalities = np.asarray(cardinalities, dtype=np.int64)
        self.labels = tuple(labels)
        if not self.token_count:
            raise ValueError("Corpus schema contains no embodiment tokens")

    @classmethod
    def from_corpus(cls, root):
        reader = CorpusReader(root)
        codec = reader.codec
        manifest = reader.manifest
        groups = []
        sampler = manifest.get("identity", {}).get("sampler", {})
        amplitude = sampler.get("sample_perturbation_m")
        # Per-cell designs (SoftWalker: sampler grid + channels) are one token per cell carrying all of that cell's
        # fields, instead of one token per value: 288 tokens of width 10 rather than 2880 scalar tokens.
        cells = int(np.prod(sampler["grid"])) if "grid" in sampler and "channels" in sampler else 0

        def component_labels(path, count):
            base = ".".join(path) if path else "embodiment"
            return tuple(base if count == 1 else f"{base}[{index}]"
                         for index in range(count))

        def array_group(field, path, force_rows=False, low=None, high=None):
            spec = codec.fields[field]
            shape = tuple(spec["shape"])
            dtype = np.dtype(spec["dtype"])
            if low is None:
                low = codec.constants[field + "_low"]
                high = codec.constants[field + "_high"]
            low, high = np.asarray(low), np.asarray(high)
            if np.issubdtype(dtype, np.floating):
                rows = force_rows or (len(shape) >= 2 and shape[-1] <= 4)
                width = shape[-1] if rows else 1
                count = int(np.prod(shape[:-1] if rows else shape, dtype=int)) or 1
                size = int(np.prod(shape, dtype=int))
                if not rows and cells > 1 and size % cells == 0 and size // cells > 1:
                    width, count = size // cells, cells
                groups.append(TokenGroup(
                    field, CONTINUOUS, count, width, shape,
                    component_labels(path, count),
                    tuple(float(value) for value in low.reshape(-1)),
                    tuple(float(value) for value in high.reshape(-1))))
                return
            low_flat = np.broadcast_to(low, shape or ()).reshape(-1)
            high_flat = np.broadcast_to(high, shape or ()).reshape(-1)
            if np.all(low_flat == 0) and np.all(high_flat == 1):
                groups.append(TokenGroup(
                    field, BINARY, len(low_flat), 1, shape,
                    component_labels(path, len(low_flat)),
                    tuple([0] * len(low_flat)), tuple([1] * len(low_flat))))
                return
            cardinalities = high_flat.astype(np.int64) - low_flat.astype(np.int64) + 1
            if np.any(cardinalities < 2):
                raise ValueError(f"Categorical field {field} has fewer than two classes")
            groups.append(TokenGroup(
                field, CATEGORICAL, len(low_flat), 1, shape,
                component_labels(path, len(low_flat)),
                tuple(int(value) for value in low_flat),
                tuple(int(value) for value in cardinalities)))

        def visit(node, path=()):
            kind = node["kind"]
            if kind == "dict":
                for key, child in node["items"].items():
                    visit(child, (*path, key))
            elif kind == "categorical":
                cardinality = len(node["values"])
                groups.append(TokenGroup(
                    node["field"], CATEGORICAL, 1, 1, (),
                    component_labels(path, 1), (0,), (cardinality,)))
            elif kind == "meshes":
                if amplitude is None or float(amplitude) <= 0:
                    raise ValueError("Mesh tokenization requires sample_perturbation_m metadata")
                for key, item in node["items"].items():
                    field = item["field"]
                    mounting = int(item["mounting_count"])
                    base = codec.constants[field + "_vertices"][mounting:]
                    delta = float(amplitude)
                    array_group(field, (*path, key), force_rows=True,
                                low=base - delta, high=base + delta)
            elif kind in ("array", "int"):
                array_group(node["field"], path)
            else:
                raise ValueError(f"Unsupported corpus node kind {kind!r}")

        visit(codec.tree)
        return cls(groups, corpus_schema=manifest["schema"],
                   corpus_constants_digest=manifest["schema"]["constants_digest"])

    def to_dict(self):
        return {
            "version": self.version,
            "token_count": self.token_count,
            "feature_width": self.feature_width,
            "max_categories": self.max_categories,
            "labels": list(self.labels),
            "groups": [group.to_dict() for group in self.groups],
            "corpus_schema": self.corpus_schema,
            "corpus_constants_digest": self.corpus_constants_digest,
        }

    def encode_arrays(self, arrays):
        first = next(iter(arrays.values()))
        batch = len(first)
        continuous = np.zeros(
            (batch, self.token_count, self.feature_width), dtype=np.float32)
        continuous_mask = np.zeros_like(continuous)
        categorical = np.zeros((batch, self.token_count), dtype=np.int64)
        binary = np.zeros((batch, self.token_count), dtype=np.float32)
        cursor = 0
        for group in self.groups:
            stop = cursor + group.count
            values = np.asarray(arrays[group.field])
            if len(values) != batch:
                raise ValueError("Corpus fields have inconsistent batch lengths")
            if group.kind == CONTINUOUS:
                values = values.reshape(batch, group.count, group.width).astype(np.float64)
                low = np.asarray(group.low, dtype=np.float64).reshape(group.count, group.width)
                high = np.asarray(group.high, dtype=np.float64).reshape(group.count, group.width)
                span = high - low
                normalized = np.divide(
                    2.0 * (values - low), span,
                    out=np.zeros_like(values), where=span > 0) - (span > 0)
                continuous[:, cursor:stop, :group.width] = normalized.astype(np.float32)
                continuous_mask[:, cursor:stop, :group.width] = (span > 0).astype(np.float32)
            elif group.kind == CATEGORICAL:
                low = np.asarray(group.low, dtype=np.int64)
                target = values.reshape(batch, group.count).astype(np.int64) - low
                cardinality = np.asarray(group.high, dtype=np.int64)
                if np.any(target < 0) or np.any(target >= cardinality):
                    raise ValueError(f"Categorical values outside schema for {group.field}")
                categorical[:, cursor:stop] = target
            else:
                target = values.reshape(batch, group.count)
                if np.any((target != 0) & (target != 1)):
                    raise ValueError(f"Nonbinary values in {group.field}")
                binary[:, cursor:stop] = target.astype(np.float32)
            cursor = stop
        return {"continuous": continuous, "continuous_mask": continuous_mask,
                "categorical": categorical, "binary": binary}


@dataclass(frozen=True)
class VAEConfig:
    layers: int = 4
    hidden_dim: int = 32
    heads: int = 4
    ffn_factor: int = 8
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0
    positional_dropout: float = 0.1

    def validate(self):
        if min(self.layers, self.hidden_dim, self.heads, self.ffn_factor) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        for value in (self.attention_dropout, self.ffn_dropout,
                      self.positional_dropout):
            if not 0 <= value < 1:
                raise ValueError("Dropout values must be in [0, 1)")


class MultiheadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim) if heads > 1 else nn.Identity()
        self.dropout = float(dropout)

    def forward(self, x, *, causal=False):
        batch, tokens, hidden = x.shape
        def split(layer):
            return layer(x).reshape(
                batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        query, key, value = split(self.query), split(self.key), split(self.value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if causal:
            mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool,
                                         device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, -torch.inf)
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.dropout, self.training)
        result = attention @ value
        result = result.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.output(result)


class TransformerStack(nn.Module):
    def __init__(self, config, *, causal=False):
        super().__init__()
        self.causal = causal
        width = config.hidden_dim * config.ffn_factor
        self.layers = nn.ModuleList()
        for index in range(config.layers):
            self.layers.append(nn.ModuleDict({
                "attention": MultiheadSelfAttention(
                    config.hidden_dim, config.heads, config.attention_dropout),
                "attention_norm": (nn.Identity() if index == 0
                                     else nn.LayerNorm(config.hidden_dim)),
                "ffn_norm": nn.LayerNorm(config.hidden_dim),
                "ffn_in": nn.Linear(config.hidden_dim, width),
                "ffn_out": nn.Linear(width, config.hidden_dim),
            }))
        self.ffn_dropout = config.ffn_dropout

    def forward(self, x):
        for layer in self.layers:
            residual = layer["attention_norm"](x)
            x = x + layer["attention"](residual, causal=self.causal)
            residual = layer["ffn_norm"](x)
            residual = layer["ffn_out"](F.dropout(
                F.relu(layer["ffn_in"](residual)), self.ffn_dropout,
                self.training))
            x = x + residual
        return x


class TransformerVAE(nn.Module):
    def __init__(self, schema, config=VAEConfig()):
        super().__init__()
        config.validate()
        self.schema = schema
        self.config = config
        hidden = config.hidden_dim
        self.continuous_input = nn.Linear(schema.feature_width * 2, hidden)
        self.category_input = nn.Embedding(schema.max_categories, hidden)
        self.binary_input = nn.Embedding(2, hidden)
        self.kind_embedding = nn.Embedding(3, hidden)
        self.position = nn.Parameter(torch.randn(1, schema.token_count, hidden))
        self.position_dropout = nn.Dropout(config.positional_dropout)
        self.encoder_mu = TransformerStack(config)
        self.encoder_logvar = TransformerStack(config)
        self.decoder = TransformerStack(config, causal=True)
        self.continuous_head = nn.Linear(hidden, schema.feature_width)
        self.category_head = nn.Linear(hidden, schema.max_categories)
        self.binary_head = nn.Linear(hidden, 1)
        self.register_buffer("token_kinds", torch.as_tensor(schema.kinds),
                             persistent=True)
        self.register_buffer("category_cardinalities",
                             torch.as_tensor(schema.category_cardinalities),
                             persistent=True)

    def embed(self, batch):
        continuous = torch.cat(
            [batch["continuous"], batch["continuous_mask"]], dim=-1)
        result = self.continuous_input(continuous)
        category = self.category_input(batch["categorical"])
        binary = self.binary_input(batch["binary"].long())
        kinds = self.token_kinds.view(1, -1, 1)
        result = torch.where(kinds == CATEGORICAL, category, result)
        result = torch.where(kinds == BINARY, binary, result)
        result = result + self.kind_embedding(self.token_kinds).unsqueeze(0)
        return self.position_dropout(result + self.position)

    def posterior(self, batch):
        embedded = self.embed(batch)
        return (self.encoder_mu(embedded),
                self.encoder_logvar(embedded).clamp(-20.0, 10.0))

    @staticmethod
    def reparameterize(mean, logvar, *, generator=None):
        noise = (torch.randn_like(mean) if generator is None else
                 torch.randn(mean.shape, dtype=mean.dtype, device=mean.device,
                             generator=generator))
        return mean + noise * torch.exp(0.5 * logvar)

    def decode(self, latent):
        hidden = self.decoder(latent)
        return {"continuous": self.continuous_head(hidden),
                "categorical_logits": self.category_head(hidden),
                "binary_logits": self.binary_head(hidden).squeeze(-1)}

    def forward(self, batch, *, sample=True, generator=None):
        mean, logvar = self.posterior(batch)
        latent = (self.reparameterize(mean, logvar, generator=generator)
                  if sample else mean)
        return self.decode(latent), mean, logvar


@dataclass
class LossMetrics:
    loss: torch.Tensor
    reconstruction: torch.Tensor
    continuous_mse: torch.Tensor
    categorical_loss: torch.Tensor
    binary_loss: torch.Tensor
    kl: torch.Tensor
    categorical_accuracy: torch.Tensor
    binary_accuracy: torch.Tensor


def vae_loss(model, batch, outputs, mean, logvar, beta):
    zero = mean.new_zeros(())
    kinds = model.token_kinds
    continuous_active = kinds == CONTINUOUS
    if bool(continuous_active.any()):
        prediction = outputs["continuous"][:, continuous_active]
        target = batch["continuous"][:, continuous_active]
        mask = batch["continuous_mask"][:, continuous_active]
        continuous_mse = ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)
    else:
        continuous_mse = zero
    categorical_active = kinds == CATEGORICAL
    if bool(categorical_active.any()):
        logits = outputs["categorical_logits"][:, categorical_active]
        targets = batch["categorical"][:, categorical_active]
        cardinalities = model.category_cardinalities[categorical_active]
        classes = torch.arange(logits.shape[-1], device=mean.device)
        logits = logits.masked_fill(
            classes.view(1, 1, -1) >= cardinalities.view(1, -1, 1), -torch.inf)
        categorical_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        categorical_accuracy = (logits.argmax(-1) == targets).float().mean()
    else:
        categorical_loss = categorical_accuracy = zero
    binary_active = kinds == BINARY
    if bool(binary_active.any()):
        logits = outputs["binary_logits"][:, binary_active]
        targets = batch["binary"][:, binary_active]
        binary_loss = F.binary_cross_entropy_with_logits(logits, targets)
        binary_accuracy = ((logits >= 0) == targets.bool()).float().mean()
    else:
        binary_loss = binary_accuracy = zero
    reconstruction = continuous_mse + categorical_loss + binary_loss
    kl = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()
    return LossMetrics(reconstruction + float(beta) * kl, reconstruction,
                       continuous_mse, categorical_loss, binary_loss, kl,
                       categorical_accuracy, binary_accuracy)


def batch_to_device(batch, device):
    return {key: torch.as_tensor(value, device=device)
            for key, value in batch.items()}
