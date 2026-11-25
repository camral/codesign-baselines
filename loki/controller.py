"""Transformer controller features for Codesign Gym LOKI policies.

Upstream LOKI applies a Transformer to per-limb proprioceptive tokens. Codesign
Gym environments expose one flat task observation and a typed corpus embodiment
schema, so this port uses one task-state token followed by those embodiment
tokens. The released controller depth and widths are retained by default.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class FlatMaskedActorCriticPolicy(ActorCriticPolicy):
    """Categorical PPO policy whose legal-action mask is embedded in flat observations."""

    def __init__(self, *args, action_mask_slice, **kwargs):
        self.action_mask_slice = tuple(int(value) for value in action_mask_slice)
        self._action_mask = None
        super().__init__(*args, **kwargs)

    def _set_action_mask(self, observations):
        start, stop = self.action_mask_slice
        # VecNormalize maps binary mask values to opposite sides of zero.
        self._action_mask = observations[..., start:stop] > 0

    def forward(self, observations, deterministic=False):
        self._set_action_mask(observations)
        return super().forward(observations, deterministic)

    def evaluate_actions(self, observations, actions):
        self._set_action_mask(observations)
        return super().evaluate_actions(observations, actions)

    def get_distribution(self, observations):
        self._set_action_mask(observations)
        return super().get_distribution(observations)

    def _predict(self, observations, deterministic=False):
        self._set_action_mask(observations)
        return super()._predict(observations, deterministic)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        if self._action_mask is None:
            raise RuntimeError("action mask was not set before constructing the distribution")
        return self.action_dist.proba_distribution(
            action_logits=logits.masked_fill(~self._action_mask, -torch.inf))



class ControllerTransformerBlock(nn.Module):
    """Released LOKI pre-norm attention and feed-forward residual block."""

    def __init__(self, hidden_dim, heads, ffn_dim, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_in = nn.Linear(hidden_dim, ffn_dim)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_out = nn.Linear(ffn_dim, hidden_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, value):
        normalized = self.attention_norm(value)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False)
        value = value + self.attention_dropout(attended)
        normalized = self.ffn_norm(value)
        transformed = self.ffn_out(
            self.ffn_dropout(torch.relu(self.ffn_in(normalized))))
        return value + self.output_dropout(transformed)


class ControllerTransformer(nn.Module):
    def __init__(self, hidden_dim, heads, layers, ffn_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            ControllerTransformerBlock(
                hidden_dim, heads, ffn_dim, dropout)
            for _ in range(layers)])

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class MorphologyTransformerExtractor(BaseFeaturesExtractor):
    """Encode a flat state plus typed morphology tokens into one policy feature."""

    def __init__(self, observation_space, *, observation_dim: int, token_layout,
                 hidden_dim: int = 128, heads: int = 2, layers: int = 5,
                 ffn_dim: int = 1024, dropout: float = 0.0):
        super().__init__(observation_space, features_dim=hidden_dim)
        if min(observation_dim, hidden_dim, heads, layers, ffn_dim) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if hidden_dim % heads or ffn_dim % hidden_dim:
            raise ValueError("hidden_dim must divide heads and ffn_dim")
        self.observation_dim = int(observation_dim)
        self.token_layout = tuple(
            (int(kind), int(count), int(width))
            for kind, count, width in token_layout)
        if any(count < 1 or width < 1 for _, count, width in self.token_layout):
            raise ValueError("Token counts and widths must be positive")
        self.token_count = sum(count for _, count, _ in self.token_layout)
        self.design_dim = sum(count * width for _, count, width in self.token_layout)
        if int(np.prod(observation_space.shape)) != self.observation_dim + self.design_dim:
            raise ValueError("Conditioned observation width differs from token layout")
        self.max_width = max(width for _, _, width in self.token_layout)
        kinds = []
        for kind, count, _ in self.token_layout:
            kinds.extend([kind] * count)
        self.register_buffer("token_kinds", torch.tensor(kinds, dtype=torch.long))

        self.state_input = nn.Linear(self.observation_dim, hidden_dim)
        self.design_input = nn.Linear(self.max_width, hidden_dim)
        self.kind_embedding = nn.Embedding(3, hidden_dim)
        self.position = nn.Parameter(
            torch.randn(1, self.token_count + 1, hidden_dim))
        self.transformer = ControllerTransformer(
            hidden_dim, heads, layers, ffn_dim, dropout)
        self._init_upstream_embeddings()

    def _init_upstream_embeddings(self):
        # Upstream controller uses a uniform [-0.1, 0.1] token projection init.
        for layer in (self.state_input, self.design_input):
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            nn.init.zeros_(layer.bias)

    def _design_tokens(self, designs):
        batch = len(designs)
        result = designs.new_zeros((batch, self.token_count, self.max_width))
        flat_cursor = token_cursor = 0
        for _, count, width in self.token_layout:
            flat_stop = flat_cursor + count * width
            token_stop = token_cursor + count
            result[:, token_cursor:token_stop, :width] = designs[
                :, flat_cursor:flat_stop].reshape(batch, count, width)
            flat_cursor, token_cursor = flat_stop, token_stop
        return result

    def forward(self, observations):
        values = observations.reshape(len(observations), -1)
        state = values[:, :self.observation_dim]
        design = values[:, self.observation_dim:]
        scale = math.sqrt(self.features_dim)
        state_token = (self.state_input(state) * scale).unsqueeze(1)
        design_tokens = self.design_input(self._design_tokens(design)) * scale
        design_tokens = design_tokens + self.kind_embedding(
            self.token_kinds).unsqueeze(0)
        tokens = torch.cat((state_token, design_tokens), dim=1)
        tokens = tokens + self.position
        encoded = self.transformer(tokens)
        return encoded[:, 0]


def token_layout(schema):
    """Serializable `(kind, count, width)` controller layout."""
    return [(group.kind, group.count, group.width) for group in schema.groups]


def controller_policy_kwargs(schema, observation_dim):
    """Released LOKI controller dimensions with a Codesign token adapter."""
    return {
        "features_extractor_class": MorphologyTransformerExtractor,
        "features_extractor_kwargs": {
            "observation_dim": int(observation_dim),
            "token_layout": token_layout(schema),
            "hidden_dim": 128,
            "heads": 2,
            "layers": 5,
            "ffn_dim": 1024,
            "dropout": 0.0,
        },
        "net_arch": {"pi": [], "vf": []},
        "activation_fn": nn.Tanh,
        "ortho_init": False,
        "log_std_init": math.log(0.9),
        # Upstream has independent mu and value Transformer modules.
        "share_features_extractor": False,
    }
