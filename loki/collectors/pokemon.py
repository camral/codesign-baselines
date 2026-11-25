"""Pokémon species/nature proposals in the benchmark's native design space."""

import hashlib
from pathlib import Path

from codesign_gym import REGISTRY

from .base import BaseCollector
from .encoding import is_pokemon_categorical


class PokemonCollector(BaseCollector):
    presets = tuple(sorted(key for key in REGISTRY if key.startswith("Pokenv")))

    def make_env(self):
        from codesign_gym import make_vec
        from pokemon_common import search_config

        # Format changes must use their registered preset so the design space
        # remains the one used by the other benchmark methods.
        if {"opponent", "generation", "difficulty", "random_league"} & self.env_kwargs.keys():
            raise ValueError("Choose a Pokémon preset instead of overriding its battle format")
        kwargs = {"embodiment_config": search_config(self.env_id),
                  "shards": 1, "envs_per_worker": 1, **self.env_kwargs}
        env, _ = make_vec(self.env_id, 1, is_eval_env=True, **kwargs)
        return env

    def configure(self):
        if not self.space.spaces or any(
            set(member.spaces) != {"species", "nature"}
            or not all(is_pokemon_categorical(leaf) for leaf in member.spaces.values())
            for member in self.space.spaces.values()
        ):
            raise ValueError("Pokémon collector requires the benchmark species/nature design space")

    def sample_embodiment(self):
        # Independent uniform categorical proposals, conditioned on validity by
        # the shared collection loop. No native child-space RNG is consumed.
        return self.sample_native_space()

    def sampler_metadata(self):
        import pokemon_codesign_env

        root = Path(pokemon_codesign_env.__file__).parent
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return {"distribution": "independent_uniform_species_and_natures",
                "search_config": self.space.search_config,
                "pokemon_python_sources_sha256": digest.hexdigest()}
