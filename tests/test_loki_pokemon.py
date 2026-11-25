"""Pokémon categorical storage and deterministic collection contracts."""

from types import SimpleNamespace

from gymnasium import spaces
import numpy as np

from loki.collectors import CorpusReader, PokemonCollector
from loki.collectors.encoding import EmbodimentCodec
from pokemon_codesign_env.spaces import Categorical


class TeamEnv:
    def __init__(self):
        self.embodiment_space = spaces.Dict({
            f"pokemon_{i}": spaces.Dict({
                "species": Categorical(("Pikachu", "Eevee", "Mew")),
                "nature": Categorical((None, "Hardy", "Timid")),
            }) for i in range(6)
        })
        self.base_embodiment = self.embodiment_space.sample()
        self.validated = []

    def validate_embodiment(self, design):
        self.validated.append(design)
        return np.array([design["pokemon_0"]["species"] != "Mew"])


class Collector(PokemonCollector):
    def identity(self, audit_every):
        return {"seed": self.seed, "audit_every": audit_every}


def test_categorical_schema_roundtrip():
    env = TeamEnv()
    codec = EmbodimentCodec(env.embodiment_space)
    restored = EmbodimentCodec.from_schema(codec.schema, codec.constants)
    assert restored.decode(codec.encode(env.base_embodiment)) == env.base_embodiment
    assert all(field["dtype"] == "<i8" for field in codec.fields.values())


def test_pokemon_native_validation_and_resume(tmp_path):
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    for root, count, resume in ((whole, 20, False), (resumed, 7, False), (resumed, 20, True)):
        env = TeamEnv()
        result = Collector("Pokenv-Random", seed=9, env=env).collect(
            root, count=count, resume=resume, audit_every=0, shard_size=3)
        assert result.complete
        assert result.stats["invalid"] > 0
        assert env.validated
    rows = list(CorpusReader(whole).iter_embodiments())
    assert rows == list(CorpusReader(resumed).iter_embodiments())
    assert all(design["pokemon_0"]["species"] != "Mew" for _, design in rows)


def test_factory_uses_benchmark_search_config(monkeypatch):
    import codesign_gym
    import pokemon_common

    config = object()
    monkeypatch.setattr(pokemon_common, "search_config", lambda env_id: config)
    calls = []
    monkeypatch.setattr(codesign_gym, "make_vec", lambda *args, **kwargs: (calls.append((args, kwargs)), None))
    instance = SimpleNamespace(env_id="Pokenv-Random", env_kwargs={})
    PokemonCollector.make_env(instance)
    assert calls[0][0] == ("Pokenv-Random", 1)
    assert calls[0][1]["embodiment_config"] is config
    assert calls[0][1]["envs_per_worker"] == 1
