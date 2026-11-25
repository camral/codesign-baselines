from dataclasses import dataclass

import numpy as np
import torch

from codesign_gym import REGISTRY
from common.codesign_adapter import MeshSpaceCodec
from launch_environment_suite import DEFERRED_ENV_IDS, FAMILIES


@dataclass
class _Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    def copy(self):
        return _Mesh(self.vertices.copy(), self.faces.copy())


class _MeshSpace:
    def __init__(self):
        vertices = np.arange(81, dtype=np.float64).reshape(27, 3) / 1000
        faces = np.asarray([[0, 1, 2], [24, 25, 26]], dtype=np.int64)
        self._base = {name: _Mesh(vertices.copy(), faces.copy()) for name in ("index", "thumb")}
        self.mounting_vertex_counts = {name: 25 for name in self._base}

    @property
    def base_meshes(self):
        return {name: mesh.copy() for name, mesh in self._base.items()}


def test_families_cover_every_non_pokemon_registry_id():
    selected = {env_id for family in FAMILIES.values() for env_id in family}
    expected = {env_id for env_id in REGISTRY if not env_id.startswith("Pokenv")}
    assert selected.isdisjoint(DEFERRED_ENV_IDS)
    assert selected | DEFERRED_ENV_IDS == expected


def test_mesh_codec_changes_only_mutable_vertices_and_preserves_faces():
    space = _MeshSpace()
    codec = MeshSpaceCodec(space)
    base = codec.encode(space.base_meshes)
    assert np.allclose(codec.low, base - 0.0003)
    assert np.allclose(codec.high, base + 0.0003)
    decoded = codec.decode(base + 0.0001)
    for name, mesh in decoded.items():
        original = space.base_meshes[name]
        assert np.array_equal(mesh.vertices[:25], original.vertices[:25])
        assert np.allclose(mesh.vertices[25:], original.vertices[25:] + 0.0001)
        assert np.array_equal(mesh.faces, original.faces)


def test_grouping_repeats_designs_and_reduces_world_results():
    adapter = object.__new__(__import__("common.codesign_adapter", fromlist=["FlatCodesignVecEnv"]).FlatCodesignVecEnv)
    adapter.num_designs = 2
    adapter.worlds_per_design = 3
    values = torch.tensor([[1.0], [2.0]])
    assert adapter.expand_design_values(values).reshape(-1).tolist() == [1, 1, 1, 2, 2, 2]
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert adapter.collapse_returns(returns).tolist() == [2.0, 5.0]
    rejected = torch.tensor([False, True, False, False, False, False])
    assert adapter.collapse_rejections(rejected).tolist() == [True, False]
