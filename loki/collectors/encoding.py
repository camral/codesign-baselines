"""Lossless numeric corpus encoding; no pickle or float32 flattening."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from gymnasium import spaces


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(json.dumps([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def is_pokemon_categorical(space):
    return (type(space).__module__ == "pokemon_codesign_env.spaces"
            and type(space).__name__ == "Categorical")


def sample_space(space, rng):
    """Sample bounded native spaces using only the collector's saved RNG."""
    if is_pokemon_categorical(space):
        return space.values[int(rng.integers(len(space.values)))]
    if isinstance(space, spaces.Dict):
        return {key: sample_space(child, rng) for key, child in space.spaces.items()}
    if isinstance(space, spaces.Box):
        if not (np.isfinite(space.low).all() and np.isfinite(space.high).all()):
            raise ValueError("Corpus sampling requires finite embodiment bounds")
        if np.issubdtype(space.dtype, np.integer):
            return rng.integers(space.low, space.high, endpoint=True, dtype=space.dtype)
        return np.asarray(rng.uniform(space.low, space.high), dtype=space.dtype)
    if isinstance(space, spaces.MultiBinary):
        return rng.integers(0, 2, size=space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Discrete):
        return int(rng.integers(space.start, space.start + space.n))
    if isinstance(space, spaces.MultiDiscrete):
        return rng.integers(space.start, space.start + space.nvec, dtype=space.dtype)
    raise TypeError(f"No native sampler for {type(space).__name__}")


class EmbodimentCodec:
    """Typed arrays plus a schema that reconstructs native embodiments exactly.

    Fixed mesh topology and mounting vertices are stored once. Mesh rows hold
    only the mutable vertices, in their original precision.
    """

    def __init__(self, space):
        self.fields = {}
        self.constants = {}
        self._mesh_type = None
        self.tree = self._build(space)

    def _field(self, shape, dtype):
        name = f"field_{len(self.fields):03d}"
        self.fields[name] = {"shape": list(shape), "dtype": np.dtype(dtype).str}
        return name

    def _build(self, space):
        if is_pokemon_categorical(space):
            return {"kind": "categorical", "field": self._field((), np.int64),
                    "values": list(space.values)}
        if hasattr(space, "base_meshes") and hasattr(space, "mounting_vertex_counts"):
            items = {}
            for key, mesh in space.base_meshes.items():
                self._mesh_type = type(mesh)
                count = int(space.mounting_vertex_counts[key])
                vertices = np.asarray(mesh.vertices)
                name = self._field(vertices[count:].shape, vertices.dtype)
                self.constants[name + "_vertices"] = vertices.copy()
                self.constants[name + "_faces"] = np.asarray(mesh.faces).copy()
                items[key] = {"field": name, "mounting_count": count}
            return {"kind": "meshes", "items": items}
        if isinstance(space, spaces.Dict):
            return {"kind": "dict", "items": {key: self._build(child) for key, child in space.spaces.items()}}
        if isinstance(space, spaces.Discrete):
            name = self._field((), space.dtype)
            self.constants[name + "_low"] = np.asarray(space.start, dtype=space.dtype)
            self.constants[name + "_high"] = np.asarray(space.start + space.n - 1, dtype=space.dtype)
            return {"kind": "int", "field": name}
        if isinstance(space, (spaces.Box, spaces.MultiBinary, spaces.MultiDiscrete)):
            name = self._field(space.shape, space.dtype)
            if isinstance(space, spaces.Box):
                low, high = space.low, space.high
            elif isinstance(space, spaces.MultiDiscrete):
                low, high = space.start, space.start + space.nvec - 1
            else:
                low, high = np.zeros(space.shape, space.dtype), np.ones(space.shape, space.dtype)
            self.constants[name + "_low"] = np.asarray(low).copy()
            self.constants[name + "_high"] = np.asarray(high).copy()
            return {"kind": "array", "field": name}
        raise TypeError(f"No corpus encoding for {type(space).__name__}")

    @property
    def schema(self):
        return {"version": 1, "tree": self.tree, "fields": self.fields,
                "constants_digest": array_digest(self.constants)}

    @classmethod
    def from_schema(cls, schema, constants):
        if schema["version"] != 1:
            raise ValueError("Unsupported embodiment schema version")
        obj = object.__new__(cls)
        obj.tree, obj.fields = schema["tree"], schema["fields"]
        obj.constants = {key: value.copy() for key, value in constants.items()}
        obj._mesh_type = None
        if array_digest(obj.constants) != schema["constants_digest"]:
            raise ValueError("Corpus constants do not match their schema")
        return obj

    def encode(self, design):
        result = {}

        def assign(name, value):
            field = self.fields[name]
            array = np.array(value, dtype=np.dtype(field["dtype"]), copy=True)
            if array.shape != tuple(field["shape"]):
                raise ValueError(f"{name}: shape {array.shape} differs from {field['shape']}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name}: non-finite embodiment value")
            # +0 and -0 describe the same numerical design.
            array[array == 0] = 0
            result[name] = array

        def visit(node, value):
            if node["kind"] == "dict":
                if value.keys() != node["items"].keys():
                    raise ValueError("Embodiment dictionary keys differ from the schema")
                for key, child in node["items"].items():
                    visit(child, value[key])
            elif node["kind"] == "categorical":
                assign(node["field"], node["values"].index(value))
            elif node["kind"] == "meshes":
                if value.keys() != node["items"].keys():
                    raise ValueError("Mesh names differ from the schema")
                for key, spec in node["items"].items():
                    name, count = spec["field"], spec["mounting_count"]
                    vertices = np.asarray(value[key].vertices)
                    base = self.constants[name + "_vertices"]
                    if vertices.shape != base.shape or not np.array_equal(vertices[:count], base[:count]):
                        raise ValueError("Mesh sampling must preserve topology and mounting vertices")
                    if not np.array_equal(value[key].faces, self.constants[name + "_faces"]):
                        raise ValueError("Mesh sampling must preserve faces")
                    assign(name, vertices[count:])
            else:
                assign(node["field"], value)

        visit(self.tree, design)
        return result

    def decode(self, row):
        def visit(node):
            if node["kind"] == "dict":
                return {key: visit(child) for key, child in node["items"].items()}
            if node["kind"] == "meshes":
                if self._mesh_type is None:
                    from envs.nerograsp.spaces import Mesh
                    self._mesh_type = Mesh
                result = {}
                for key, spec in node["items"].items():
                    name, count = spec["field"], spec["mounting_count"]
                    vertices = self.constants[name + "_vertices"].copy()
                    vertices[count:] = row[name]
                    result[key] = self._mesh_type(vertices, self.constants[name + "_faces"].copy())
                return result
            value = np.array(row[node["field"]], dtype=self.fields[node["field"]]["dtype"], copy=True)
            if node["kind"] == "categorical":
                index = int(value)
                if not 0 <= index < len(node["values"]):
                    raise ValueError("Categorical index outside stored vocabulary")
                return node["values"][index]
            return int(value) if node["kind"] == "int" else value

        return visit(self.tree)

    def stack(self, rows):
        return {name: (np.stack([row[name] for row in rows]) if rows else
                       np.empty((0, *field["shape"]), dtype=field["dtype"]))
                for name, field in self.fields.items()}
