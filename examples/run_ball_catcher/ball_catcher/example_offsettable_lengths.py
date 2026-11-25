"""Length offsets for bodies named "*_offsettable_<dir>[_<extent>][_equ<n>]".

Generalizes set_link_lengths (test_arm_link_lengths.py) from a hardcoded 3-link
chain to any capsule body discovered by name, offset along any of x/y/z. Each
offsettable body's capsule is assumed to run from its local origin to
length * axis, exactly like the arm's `fromto="0 0 0 {length} 0 0"`.
A trailing ``_equ<n>`` puts the body in a group sharing one offset entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

_NAME_RE = re.compile(r"_offsettable_([xyz])(?:_(pos|neg))?(?:_equ(\d+))?$")
_AXIS = {"x": 0, "y": 1, "z": 2}
_EXTENT = {"pos": (0.0, 1.0), "neg": (-1.0, 0.0), None: (-1.0, 1.0)}


@dataclass
class Rest:
    """Undeformed geom of one link. Everything is rebuilt from this, so a
    command is absolute and re-applying it changes nothing."""
    dir: np.ndarray  # unit vector the capsule runs along, in the body frame
    half: float
    radius: float
    density: float
    is_cyl: bool
    child_pos: dict[int, np.ndarray]


@dataclass
class OffsetSpec:
    body_id: int
    axis: int
    lo: float
    hi: float
    act: int          # index into ``offsets``; an _equ group shares one
    rest: Rest | None = None  # captured from the model on the first apply


def find_offsettable_bodies(spec: mujoco.MjSpec) -> list[OffsetSpec]:
    """Body id, offset axis (0/1/2), clip range and offset slot per body.

    A ``_pos``/``_neg`` suffix makes the body one-sided -- it may only lengthen
    or only shorten -- and an unsuffixed one goes both ways. Slots are numbered
    by first appearance; bodies tagged with the same ``_equ<n>`` share one, so
    they always stretch by the same amount, which they can only do if they also
    share a clip range.
    Requires a compiled spec -- ``body.id`` is -1 until ``spec.compile()`` runs.
    """
    found, groups, n_slot = [], {}, 0
    for body in spec.bodies:
        m = _NAME_RE.search(body.name)
        if not m:
            continue
        equ, extent = m.group(3), _EXTENT[m.group(2)]
        if equ is not None and equ in groups:
            act, name, group_extent = groups[equ]
            if extent != group_extent:
                raise RuntimeError(
                    f"{body.name}: clip range {extent} differs from {name}'s "
                    f"{group_extent}; an _equ{equ} group shares one offset, so "
                    "its members must share one pos/neg suffix")
        else:
            act, n_slot = n_slot, n_slot + 1
            if equ is not None:
                groups[equ] = (act, body.name, extent)
        found.append(OffsetSpec(body.id, _AXIS[m.group(1)], *extent, act))
    return found


def n_offset_slots(offsettable: list[OffsetSpec]) -> int:
    """Length ``offsets`` needs: one entry per body, minus grouped duplicates."""
    return max((o.act for o in offsettable), default=-1) + 1


def _mass_inertia(length, radius, density, is_cyl):
    """(mass, principal inertia) about the centre, in the geom frame (long axis
    = local z), matching MuJoCo's ``inertiafromgeom``."""
    m_cyl = density * np.pi * radius**2 * length
    transverse = m_cyl * (3.0 * radius**2 + length**2) / 12.0
    if is_cyl:
        return m_cyl, np.array([transverse, transverse, 0.5 * m_cyl * radius**2])
    m_cap = density * (2.0 / 3.0) * np.pi * radius**3
    axial = 0.5 * m_cyl * radius**2 + 2.0 * (2.0 / 5.0) * m_cap * radius**2
    transverse += 2.0 * m_cap * ((2.0 / 5.0) * radius**2 + length**2 / 4.0
                                 + (3.0 / 8.0) * radius * length)
    return m_cyl + 2.0 * m_cap, np.array([transverse, transverse, axial])


def _capture(model: mujoco.MjModel, o: OffsetSpec) -> Rest:
    """Record ``o``'s undeformed geom. Its capsule/cylinder must run from the
    body origin, along an axis whose dominant component is the named one -- a
    splayed leg qualifies, but the offset then follows the geom, not the axis."""
    gid = model.body_geomadr[o.body_id]
    radius, half = float(model.geom_size[gid][0]), float(model.geom_size[gid][1])
    direction = model.geom_pos[gid] / max(half, 1e-9)
    if abs(np.linalg.norm(direction) - 1.0) > 1e-6 or np.argmax(np.abs(direction)) != o.axis:
        raise RuntimeError(
            f"body {o.body_id}: its geom must run from the body origin along "
            f"axis {o.axis} (got pos {model.geom_pos[gid]}, half-length {half})")
    is_cyl = model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
    unit, _ = _mass_inertia(2.0 * half, radius, 1.0, is_cyl)
    kids = np.where(model.body_parentid == o.body_id)[0]
    return Rest(direction.copy(), half, radius,
                float(model.body_mass[o.body_id]) / unit, bool(is_cyl),
                {int(c): model.body_pos[c].copy() for c in kids})


def apply_offsets(model: mujoco.MjModel, offsettable: list[OffsetSpec], offsets,
                   max_offset: float, data: mujoco.MjData | None = None) -> mujoco.MjModel:
    """Stretch each offsettable body to ``clip(offsets[o.act], lo, hi) * max_offset``
    past its rest length.

    Offsets are absolute: everything is rebuilt from the rest geometry captured
    on the first call, so a zero command restores the compiled model exactly and
    repeating a command is a no-op. Direct children are shifted along the link's
    own axis to stay attached at the tip, since ``body_pos`` does not itself
    track the parent's geometry.
    """
    if data is None:
        data = mujoco.MjData(model)

    for o in offsettable:
        if o.rest is None:
            o.rest = _capture(model, o)
        r = o.rest
        delta = float(np.clip(offsets[o.act], o.lo, o.hi)) * max_offset
        half = r.half + delta / 2.0
        mass, inertia = _mass_inertia(2.0 * half, r.radius, r.density, r.is_cyl)
        # A capsule's caps stick out past its half-length; a cylinder's flat end
        # does not, so its bounds are tighter.
        cap = 0.0 if r.is_cyl else r.radius

        gid = model.body_geomadr[o.body_id]
        pos = r.dir * half
        model.geom_pos[gid] = pos
        model.geom_size[gid] = [r.radius, half, 0.0]
        model.geom_rbound[gid] = np.hypot(r.radius, half) if r.is_cyl else half + cap
        model.geom_aabb[gid] = [0.0, 0.0, 0.0, r.radius, r.radius, half + cap]
        adr = model.body_bvhadr[o.body_id]
        if adr >= 0 and model.body_bvhnum[o.body_id] == 1 and model.body_geomnum[o.body_id] == 1:
            model.bvh_aabb[adr] = model.geom_aabb[gid]

        model.body_ipos[o.body_id] = pos
        model.body_mass[o.body_id] = mass
        model.body_inertia[o.body_id] = inertia

        for cid, cpos in r.child_pos.items():
            model.body_pos[cid] = cpos + r.dir * delta

    _set_const_preserving_state(model, data)
    return model


def _set_const_preserving_state(model, data):
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(size)
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_setConst(model, data)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
