"""CBD-SA (+ SVD) design codec isolated to the ``fasttd3_cmaes`` baseline.

Assumes ``https://github.com/camral/structural-affinity`` is cloned at
``codesign-baselines/structural-affinity`` (or installed in the active Python environment).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

_SA_REPO = Path(__file__).resolve().parents[1] / "structural-affinity"
if _SA_REPO.is_dir() and str(_SA_REPO) not in sys.path:
    sys.path.insert(0, str(_SA_REPO))


def _make_ring_cage(verts: np.ndarray, n_rings: int = 6, n_sides: int = 8, clearance: float = 0.0012):
    """Lofted cross-section ring cage along +y, matching ``examples/nerograsp_cbdsa_reparam.py``."""
    import trimesh

    y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
    y_levels = np.linspace(y_min - clearance, y_max + clearance, n_rings)
    dy = (y_max - y_min) / max(n_rings - 1, 1)
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    chord_scale = 1.0 / np.cos(np.pi / n_sides)

    rings, centers = [], []
    for y_r in y_levels:
        slab = np.abs(verts[:, 1] - y_r) <= max(0.75 * dy, clearance * 2.0)
        sub = verts[slab] if slab.sum() >= 3 else verts
        cx, cz = float(sub[:, 0].mean()), float(sub[:, 2].mean())
        centers.append((cx, cz))
        rad = np.maximum(((sub[:, [0, 2]] - np.array([cx, cz])) @ dirs.T).max(axis=0), 1e-4) * chord_scale + clearance
        rings.append(np.column_stack([cx + rad * dirs[:, 0], np.full(n_sides, y_r), cz + rad * dirs[:, 1]]))

    poles = np.array([
        [centers[0][0], y_min - 1.5 * clearance, centers[0][1]],
        [centers[-1][0], y_max + 1.5 * clearance, centers[-1][1]],
    ])
    c_verts = np.vstack(rings + [poles])
    bot_pole, top_pole = n_rings * n_sides, n_rings * n_sides + 1

    faces = []
    for r in range(n_rings - 1):
        for s in range(n_sides):
            sn = (s + 1) % n_sides
            a, b, c, d = r * n_sides + s, r * n_sides + sn, (r + 1) * n_sides + s, (r + 1) * n_sides + sn
            faces.extend(([a, c, b], [b, c, d]))
    for s in range(n_sides):
        sn = (s + 1) % n_sides
        top_r = (n_rings - 1) * n_sides
        faces.extend(([bot_pole, s, sn], [top_pole, top_r + sn, top_r + s]))
    c_faces = np.asarray(faces, dtype=np.int64)
    if trimesh.Trimesh(vertices=c_verts, faces=c_faces, process=False).volume < 0:
        c_faces = c_faces[:, [0, 2, 1]]

    center = verts.mean(axis=0)
    for _ in range(15):
        if trimesh.Trimesh(vertices=c_verts, faces=c_faces, process=False).contains(verts).all():
            break
        c_verts = center + (c_verts - center) * 1.05
    return c_verts, c_faces


class CBDSAMeshSpaceCodec:
    """CBD-SA (+ SVD) low-dimensional codec for a NeroGrasp / NeroTools MeshSpace.

    Uses lofted ring cages (``_make_ring_cage``) matching ``examples/nerograsp_cbdsa_reparam.py``:
    - Parallel gripper (``{"left", "right"}``): ``ring_5x8`` (clr 1.5 mm) + ``ring_4x6`` (clr 3.5 mm),
      ``alpha=0.60, density=0.25, normalize=True, svd_modes=4`` (24-D total).
    - LEAP hand (``{"index", "thumb"}``): ``ring_10x12`` (clr 1.0 mm) + ``ring_6x8`` (clr 3.0 mm),
      ``alpha=0.50, density=0.35, normalize=True, svd_modes=4`` (24-D total).
    """

    def __init__(self, space, bound_m: float = 0.005, *, svd_modes: int = 4):
        from structural_affinity import affinity_matrix, mvc_3d, propagation_operator, sparsify

        self.space = space
        self.base_meshes = space.base_meshes
        self.names = tuple(sorted(self.base_meshes))
        self.mounting_counts = {name: int(space.mounting_vertex_counts[name]) for name in self.names}

        is_gripper = "left" in self.base_meshes and "right" in self.base_meshes
        if is_gripper:
            in_cfg, out_cfg = (5, 8, 0.0015), (4, 6, 0.0035)
            alpha, density, normalize = 0.60, 0.25, True
        else:
            in_cfg, out_cfg = (10, 12, 0.0010), (6, 8, 0.0030)
            alpha, density, normalize = 0.50, 0.35, True

        self.operators: dict[str, np.ndarray] = {}
        self.mode_counts: dict[str, int] = {}
        for name in self.names:
            verts = np.asarray(self.base_meshes[name].vertices, dtype=np.float64)
            mc = self.mounting_counts[name]
            inner_v, inner_f = _make_ring_cage(verts, *in_cfg)
            outer_v, outer_f = _make_ring_cage(inner_v, *out_cfg)
            W_mesh = mvc_3d(verts, inner_v, inner_f)
            W_lu = np.maximum(mvc_3d(inner_v, outer_v, outer_f), 0.0)
            W_lu /= np.maximum(W_lu.sum(axis=1, keepdims=True), 1e-12)
            C_aff = sparsify(affinity_matrix(W_lu), density=density)
            A_op = propagation_operator(C_aff, alpha, normalize=normalize)
            J = W_mesh[mc:] @ A_op.toarray()
            if svd_modes is not None and svd_modes > 0:
                r = min(int(svd_modes), J.shape[1])
                _, _, vt = np.linalg.svd(J, full_matrices=False)
                V_r = vt[:r].T
                V_r = V_r / np.maximum(np.max(np.abs(V_r), axis=0, keepdims=True), 1e-8)
                J = J @ V_r
            self.operators[name] = J
            self.mode_counts[name] = J.shape[1]

        dim = sum(k * 3 for k in self.mode_counts.values())
        self.low = np.full(dim, -float(bound_m), dtype=np.float32)
        self.high = np.full(dim, float(bound_m), dtype=np.float32)
        self.discrete = np.zeros(dim, dtype=bool)
        self.space_flat = spaces.Box(self.low, self.high, dtype=np.float32)

    def encode(self, value):
        pieces = []
        for name in self.names:
            mc = self.mounting_counts[name]
            base_v = np.asarray(self.base_meshes[name].vertices, dtype=np.float64)[mc:]
            cur_v = np.asarray(value[name].vertices, dtype=np.float64)[mc:]
            delta = cur_v - base_v
            if np.allclose(delta, 0.0):
                D = np.zeros((self.mode_counts[name], 3), dtype=np.float64)
            else:
                D, *_ = np.linalg.lstsq(self.operators[name], delta, rcond=None)
            pieces.append(D.reshape(-1).astype(np.float32))
        return np.concatenate(pieces)

    def decode(self, value):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim == 1:
            return self._decode_one(array)
        if array.ndim != 2:
            raise ValueError(f"flat embodiment must have rank one or two, got {array.shape}")
        return [self._decode_one(row) for row in array]

    def _decode_one(self, value):
        if value.shape != self.low.shape:
            raise ValueError(f"flat embodiment has shape {value.shape}, expected {self.low.shape}")
        theta = np.asarray(value, dtype=np.float64)
        result, offset = {}, 0
        for name in self.names:
            k = self.mode_counts[name]
            D = theta[offset : offset + 3 * k].reshape(k, 3)
            offset += 3 * k
            mesh = self.base_meshes[name]
            verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
            verts[self.mounting_counts[name] :] += self.operators[name] @ D
            result[name] = type(mesh)(verts, np.array(mesh.faces, copy=True))
        return result

    @staticmethod
    def project_normalized(value):
        return value.clamp(-1, 1)


def apply_cbdsa_codec(env, *, bound_m: float | None = None):
    """Replace a ``FlatCodesignVecEnv``'s ``MeshSpaceCodec`` with ``CBDSAMeshSpaceCodec`` in place."""
    raw = getattr(env, "env", None)
    if raw is None or not hasattr(raw.embodiment_space, "base_meshes"):
        return env, env.base_embodiment.copy()
    codec_kwargs = {} if bound_m is None else {"bound_m": bound_m}
    env.design_codec = CBDSAMeshSpaceCodec(raw.embodiment_space, **codec_kwargs)
    env.embodiment_space = env.design_codec.space_flat
    base_raw = raw.base_embodiment[0] if getattr(raw, "base_embodiment_is_batched", False) else raw.base_embodiment
    env.base_embodiment = env.design_codec.encode(base_raw)
    return env, env.base_embodiment.copy()
