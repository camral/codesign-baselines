"""Ball-catcher as a GPU-parallel batched env, with a per-world body shape.

Each of the ``num_envs`` worlds is the same scene -- an ant carrying an arm with
a glove on its tip, and a ball pitched at it -- but every link named
``*_offsettable_<axis>[_pos|_neg]`` is stretched per world by its entry of a
normalised reset action in [-1, 1], each worth +/- ``max_offset`` metres. The
whole assembled model is searched, so an ant leg and an arm link are the same
thing here: name either one and it joins the shape. A trailing ``_equ<n>``
groups every link carrying that tag onto one shared action entry, so symmetric
limbs stay symmetric. Under
``offset_glove`` the action carries three more entries, stretching the glove's
width, height and wall height by up to ``glove_max_offset``, and under
``disable_legs`` one more that takes two of the ant's legs out of the world. That
all
happens in device memory on reset, with no recompile: only per-world model
*values* differ (geom poses/sizes, body poses/inertia), which is the case the
base class is built for.

This module is the host half -- scene assembly, spaces, obs and reward. The
per-step device work lives in ``ball_catcher_kernels``.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from . import ball_catcher_kernels as kern
from .glove import add_glove, glove_layout
from .parallel_env import ParallelMujocoEnv
# The "*_offsettable_<axis>[_pos|_neg][_equ<n>]" convention
import re

_NAME_RE = re.compile(r"_offsettable_([xyz])(?:_(pos|neg))?(?:_equ(\d+))?$")
_AXIS = {"x": 0, "y": 1, "z": 2}
_EXTENT = {"pos": (0.0, 1.0), "neg": (-1.0, 0.0), None: (-1.0, 1.0)}


# The asset tree is data, not an importable package (pyproject packages only
# ``envs*``), so it is located relative to this file -- which an editable
# install leaves pointing at the source tree.
_ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "assets"))

_SEG_DTYPE = {"geom": wp.int32, "body": wp.int32, "is_cyl": wp.int32, "act": wp.int32,
              "child_adr": wp.int32, "child_num": wp.int32, "child_id": wp.int32,
              "dir": wp.vec3, "child_pos": wp.vec3}
# Glove dimensions the reset action stretches, in action order, and the arrays
# describing them that the deform kernel reads.
_GLOVE_PARAMS = ("width", "height", "wall_h")
_GLOVE_DTYPE = {"geom": wp.int32, "base_pos": wp.vec3, "base_size": wp.vec3,
                "dpos": wp.vec3, "dsize": wp.vec3, "base": wp.float32,
                "floor": wp.float32}

# Contact margin standing in for "this geom is not here". Contacts are only
# looked for, and only written, within margin of a surface, so anything past
# every distance in the scene switches the geom off; see _set_legs.
_LEG_OFF_MARGIN = -1.0e3


def _unit_mass(half: float, radius: float, is_cyl: int) -> float:
    """Mass of the link's geom at unit density (host twin of _mass_inertia)."""
    m_cyl = np.pi * radius ** 2 * 2.0 * half
    return m_cyl if is_cyl else m_cyl + (4.0 / 3.0) * np.pi * radius ** 3


def _sq_pairs(v: np.ndarray) -> np.ndarray:
    """(y^2+z^2, z^2+x^2, x^2+y^2) -- host twin of the kernel's helper."""
    return np.stack([v[..., 1] ** 2 + v[..., 2] ** 2,
                     v[..., 2] ** 2 + v[..., 0] ** 2,
                     v[..., 0] ** 2 + v[..., 1] ** 2], axis=-1)


def _box_body(pos: np.ndarray, size: np.ndarray, density: float):
    """(mass, centre of mass, inertia diagonal about it) of axis-aligned boxes.

    Host twin of the kernel's accumulation, and the only place it is checked
    against the compiler.
    """
    m = density * 8.0 * np.prod(size, axis=1)
    com = (m[:, None] * pos).sum(0) / m.sum()
    inertia = (m[:, None] * (_sq_pairs(size) / 3.0 + _sq_pairs(pos))).sum(0)
    return m.sum(), com, inertia - m.sum() * _sq_pairs(com)


def _glove_table(cfg: dict) -> dict[str, np.ndarray]:
    """Undeformed pose/half-size of every glove box plus their derivatives with
    respect to (width, height, wall_h). The layout is affine in the three, so a
    unit step reads each derivative off exactly."""
    def rows(**kw) -> np.ndarray:
        return np.array([p + s for _, p, s in glove_layout(**{**cfg, **kw})])
    base = rows()
    jac = np.stack([rows(**{k: cfg[k] + 1.0}) - base for k in _GLOVE_PARAMS], 1)
    return {"base_pos": base[:, :3], "base_size": base[:, 3:],
            "dpos": jac[..., :3], "dsize": jac[..., 3:]}


def _quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions, broadcasting over leading dims."""
    w0, x0, y0, z0 = q.unbind(-1)
    w1, x1, y1, z1 = r.unbind(-1)
    return torch.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dim=-1)


def _find_segments(m: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Table of every offsettable link, ordered by body id.

    The whole assembled model is searched -- robot and arm alike. Each such
    body's first geom must be a capsule/cylinder running from the body origin
    along the axis in its name, so a +1 action always lengthens it; the axis
    letter only has to be the direction's dominant component, so a leg splayed
    out diagonally qualifies. Bodies hanging off the far end are recorded so the
    reset kernel can keep them attached, which means every child of an
    offsettable link has to sit at that end.

    ``act`` maps each link to its action slot. A link is its own slot unless a
    trailing ``_equ<n>`` puts it in a group, which shares one slot -- one search
    dimension, one observation entry, one offset applied to every member. Slots
    are numbered by first appearance in body-id order.
    """
    rows, kid_id, kid_pos, groups, n_slot = [], [], [], {}, 0
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        match = _NAME_RE.search(name)
        if match is None:
            continue
        g = int(m.body_geomadr[b])
        radius, half = float(m.geom_size[g][0]), float(m.geom_size[g][1])
        direction = m.geom_pos[g] / max(half, 1e-9)
        axis = _AXIS[match.group(1)]
        if (abs(np.linalg.norm(direction) - 1.0) > 1e-6
                or int(np.argmax(np.abs(direction))) != axis):
            raise RuntimeError(
                f"{name}: its geom must run from the body origin along "
                f"{match.group(1)} (got pos {m.geom_pos[g]}, half-length {half})")
        is_cyl = int(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
        kids = np.flatnonzero(m.body_parentid == b)
        lo, hi = _EXTENT[match.group(2)]
        equ = match.group(3)
        if equ is None or equ not in groups:
            act, n_slot = n_slot, n_slot + 1
            if equ is not None:
                groups[equ] = (act, lo, hi, name)
        else:
            act, glo, ghi, gname = groups[equ]
            if (lo, hi) != (glo, ghi):
                raise RuntimeError(
                    f"{name}: clip range {(lo, hi)} differs from {gname}'s "
                    f"{(glo, ghi)}; an _equ{equ} group shares one offset, so "
                    "its members must share one pos/neg suffix")
        rows.append((b, g, direction, radius, half,
                     float(m.body_mass[b]) / _unit_mass(half, radius, is_cyl),
                     is_cyl, lo, hi, act, len(kid_id), len(kids)))
        kid_id += [int(k) for k in kids]
        kid_pos += [m.body_pos[k] for k in kids]
    if not rows:
        raise RuntimeError("no '*_offsettable_*' bodies in the model")

    cols = list(zip(*rows))
    names = ("body", "geom", "dir", "radius", "half0", "density", "is_cyl",
             "lo", "hi", "act", "child_adr", "child_num")
    seg = {n: np.asarray(c) for n, c in zip(names, cols)}
    seg["child_id"] = np.asarray(kid_id, dtype=np.int64)
    seg["child_pos"] = np.asarray(kid_pos, dtype=np.float64).reshape(-1, 3)
    return seg

class BallCatcherParallelEnv(ParallelMujocoEnv):

    BASE_BODY = "base"

    def __init__(
        self,
        num_envs: int = 4,
        *,
        # arm_xml is attached to ant_xml's base body at the origin, and ant_xml
        # into scene_xml; the XMLs' own timestep is overridden by ``timestep``.
        ant_xml: str = "ant_reacher/dog.xml",
        arm_xml: str = "ant_reacher/ant_arm.xml",
        scene_xml: str = "_common/scene_ant.xml",
        leg_chain_a: str = "midbar",
        leg_chain_b: str = "back_right_leg",
        # Keyframes holding the reset pose, one per source model -- the base
        # brings its own, the arm brings its own, and they are laid over each
        # other. A name the assembled model does not have is skipped, and a joint
        # no keyframe mentions starts at qpos0.
        init_key: str | tuple[str, ...] | list[str] = ("init_pos", "init_pos2"),
        ball_size: float = 0.1,
        ball_mass: float = 1,
        max_offset: float = 0.1,
        max_return: float = 10.0,
        max_goal_return: float = 9.9,
        goal_dwell_steps: int = 0,
        goal_radius: float = 0.1,
        glove_width: float = 0.25,
        glove_height: float = 0.25,
        glove_wall_h: float = 0.2,
        glove_thickness: float = 0.04,
        glove_goal_only: bool = False,
        # When set, the reset action gains three entries -- glove width, height
        # and wall height, in that order -- scaled by glove_max_offset.
        offset_glove: bool = False,
        glove_max_offset: float = 0.05,
        # When set, the reset action gains a trailing switch: above zero the ant
        # keeps the LEG_CHAIN_* chains, at or below zero they are gone.
        disable_legs: bool = False,
        # Clear it and neither the ball nor the glove leaving the field ends the
        # episode -- it then runs to a catch, a physics blow-up or the time
        # limit. For recording video, where a run cut short after twenty steps
        # shows nothing; leave it set anywhere a return is being learned from or
        # measured, since without it the task has no failure terminal.
        terminate_out_of_field: bool = True,
        pitch_from: list[float] = [2.0, 1.0, 1.0],
        pitch_to: list[float] = [0, 0, 3.0],
        pitch_to_radius: float = 0.0,
        pitch_vel: float = 5.0,
        field_pos: list[float] = [1.0, 0.5, 1.5],
        field_size: list[float] = [1.5, 1.5, 1],
        phys_substeps: int = 8,
        max_rel_dist: float = 1.75,
        vel_scale: float = 10.0,
        force_mag: float = 0.0,
        force_every: int = 0,
        force_hold: int = 0,
        force_viz_scale: float = 0.02,
        timestep: float = 0.005,
        # parallel-runner params
        device: str = "cuda",
        viz_env_id: int = 0,
        max_episode_steps: int = 4000,
        render_mode: str | None = None,
        debug: bool = True,
        use_graph: bool = True,
        check_finite: bool = False,
        njmax: int = 128,
        nconmax: int | None = None,
        seed: int = 1234,
    ):
        self._xmls = (ant_xml, arm_xml, scene_xml)
        self.leg_chain_a = leg_chain_a
        self.leg_chain_b = leg_chain_b
        # TODO: goal only doesn't work yet. Either delete or fix.
        self._glove = dict(width=glove_width, height=glove_height,
                           wall_h=glove_wall_h, thickness=glove_thickness,
                           goal_radius=goal_radius, goal_only=glove_goal_only)
        self.offset_glove = bool(offset_glove)
        self.glove_max_offset = float(glove_max_offset)
        self.disable_legs = bool(disable_legs)
        self.terminate_out_of_field = bool(terminate_out_of_field)
        self.init_key = ((init_key,) if isinstance(init_key, str)
                         else tuple(init_key))
        self.ball_size = float(ball_size)
        self.ball_mass = float(ball_mass)
        self.max_offset = float(max_offset)
        self.goal_dwell_steps = int(goal_dwell_steps)
        self.goal_radius = float(goal_radius)
        # Without a dwell window there is no at-goal reward at all, so the whole
        # budget goes into the shaping term.
        self.at_goal_rew = (float(max_goal_return) / self.goal_dwell_steps
                            if self.goal_dwell_steps > 0 else 0.0)
        self.prog_gain = float(max_return) - self.at_goal_rew * self.goal_dwell_steps
        self.pitch_from = np.asarray(pitch_from, dtype=np.float32)
        self.pitch_to = np.asarray(pitch_to, dtype=np.float32)
        self.pitch_to_radius = float(pitch_to_radius)
        self.pitch_vel = float(pitch_vel)
        self.field_pos = np.asarray(field_pos, dtype=np.float32)
        self.field_size = np.asarray(field_size, dtype=np.float32)
        self.max_rel_dist = float(max_rel_dist)
        self.vel_scale = float(vel_scale)
        self.force_mag = float(force_mag)
        self.force_every = int(force_every)
        self.force_hold = int(force_hold)
        self.force_viz_scale = float(force_viz_scale)
        self.force_on = (self.force_every > 0
                              and self.force_hold > 0)
        self.timestep = float(timestep)
        self.check_finite = bool(check_finite)

        super().__init__(
            num_envs, device=device, viz_env_id=viz_env_id,
            max_episode_steps=max_episode_steps, debug=debug,
            seed=seed, phys_substeps=phys_substeps, render_mode=render_mode,
            use_graph=use_graph, njmax=njmax, nconmax=nconmax)

        self._cache_model_addresses()
        self._alloc_device_state()
        self._build_kernel_params()
        self._make_spaces()

        # The step program, in order. The reset kernel must launch first: it
        # rewrites model and data fields everything after it reads.
        self.register_step_launch(self._launch_reset)
        if self.force_on:
            self.register_step_launch(self._launch_force)
        self.register_step_launch(self._launch_ctrl)

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        ant_xml, arm_xml, scene_xml = self._xmls
        arm = mujoco.MjSpec.from_file(os.path.join(_ASSETS, arm_xml))
        add_glove(arm, **self._glove)
        # a child's keyframes are only namespaced correctly if it is compiled before attach
        arm.compile()                 
        ant = mujoco.MjSpec.from_file(os.path.join(_ASSETS, ant_xml))
        ant.attach(arm, prefix="", frame=ant.body(self.BASE_BODY).add_frame())
        ant.compile()

        spec = mujoco.MjSpec.from_file(os.path.join(_ASSETS, scene_xml))
        spec.attach(ant, prefix="", frame=spec.worldbody.add_frame())
        ball = spec.worldbody.add_body(name="ball", pos=self.pitch_from.tolist())
        ball.add_freejoint(name="ball_free")
        ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[self.ball_size, 0, 0], mass=self.ball_mass,
                      contype=1, conaffinity=1, rgba=[0.9, 0.9, 0, 1],
                      solref=[0.04, 1], priority=100)
        # The bounds the ball has to stay inside: drawn, never collided with.
        spec.worldbody.add_geom(
            name="field", type=mujoco.mjtGeom.mjGEOM_BOX, group=3,
            pos=self.field_pos.tolist(), size=self.field_size.tolist(),
            contype=0, conaffinity=0, rgba=[0.3, 0.6, 1.0, 0.1])
        spec.option.timestep = self.timestep
        return spec

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _cache_model_addresses(self) -> None:
        m = self.m
        self._ball_bid = m.body("ball").id
        self._ball_qadr = int(m.joint("ball_free").qposadr[0])
        self._ball_dadr = int(m.joint("ball_free").dofadr[0])
        if self._ball_qadr + 7 != m.nq:
            raise RuntimeError("the ball's freejoint must be the model's last "
                               "joint; obs slices the robot off the front")
        # Everything ahead of the ball is the robot.
        self._nq_robot, self._nv_robot = self._ball_qadr, self._ball_dadr
        self._goal_sid = m.site("goal_site").id
        self._goal_bid = int(m.site_bodyid[self._goal_sid])
        # Site frames are fixed in their body, so the site's world quat is the
        # body's composed with this constant.
        self._goal_sq = torch.as_tensor(m.site_quat[self._goal_sid],
                                        dtype=torch.float32, device=self.device)
        self._root_bid = int(m.body_rootid[self._goal_bid])
        self._base_bid = m.body(self.BASE_BODY).id
        self._segs = _find_segments(m)
        self._glove_tab = self._find_glove(m)
        self._leg_geoms = self._find_legs(m)
        self.n_links = len(self._segs["body"])
        # Action slots, not links: an _equ group of links fills one slot.
        self.n_link_slots = int(self._segs["act"].max()) + 1
        self.n_glove_params = len(_GLOVE_PARAMS) if self.offset_glove else 0
        self.n_leg_params = 1 if self.disable_legs else 0
        self.n_offsets = self.n_link_slots + self.n_glove_params + self.n_leg_params
        self._init_qpos = self._reset_pose(m)
        if self.disable_legs:
            # Reading the margin per world is the whole mechanism; without this
            # the field is one row shared by every world.
            self._make_field_per_world("geom_margin")

    def _find_legs(self, m: mujoco.MjModel) -> np.ndarray:
        """Every geom of the two switchable chains, roots included. Empty unless
        the switch exists, so a model without those bodies still builds."""
        if not self.disable_legs:
            return np.zeros(0, dtype=np.int64)
        chain = {m.body(name).id for name in (self.leg_chain_a, self.leg_chain_b)}
        for b in range(m.nbody):                 # a parent always precedes its children
            if int(m.body_parentid[b]) in chain:
                chain.add(b)
        geoms = [g for b in sorted(chain)
                 for g in range(int(m.body_geomadr[b]),
                                int(m.body_geomadr[b]) + int(m.body_geomnum[b]))]
        if not geoms:
            raise RuntimeError(f"the {self.leg_chain_a!r}/{self.leg_chain_b!r} chains carry "
                               "no geoms; disable_legs has nothing to switch off")
        return np.array(geoms, dtype=np.int64)

    def _find_glove(self, m: mujoco.MjModel) -> dict:
        """What the glove deformer needs: the boxes, the affine layout, the
        density they were compiled at, the floors their dimensions may not shrink
        past, and the rotation taking a body-frame inertia diagonal into the
        principal frame ``body_iquat`` froze at compile time."""
        cfg = {k: self._glove[k] for k in (*_GLOVE_PARAMS, "thickness")}
        tab: dict = _glove_table(cfg)
        tab["geom"] = np.array([m.geom(n).id for n, _, _ in glove_layout(**cfg)])
        bid = int(m.body("glove").id)
        tab["body"], tab["n_geom"] = bid, len(tab["geom"])
        tab["density"] = float(m.body_mass[bid]) / (
            8.0 * np.prod(tab["base_size"], axis=1).sum())
        tab["base"] = np.array([cfg[k] for k in _GLOVE_PARAMS])
        # A wall may flatten to nothing, but the tray it stands on may not: below
        # twice the thickness the bottom box inverts.
        tab["floor"] = np.array([2.0 * cfg["thickness"], 2.0 * cfg["thickness"], 0.0])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[bid])
        tab["principal"] = (rot.reshape(3, 3) ** 2).T
        if self.offset_glove:
            self._check_glove(m, tab)
        return tab

    def _check_glove(self, m: mujoco.MjModel, tab: dict) -> None:
        """The deformer keeps only the diagonal of the glove's inertia, which is
        exact just while its boxes stay symmetric about x=0 and y=0. Rebuilding
        the undeformed body and comparing with the compiler's own numbers says
        whether that -- and the layout, and the density -- still holds."""
        if (tab["base"] <= tab["floor"]).any():
            raise RuntimeError(f"glove dimensions {tab['base']} are at or below "
                               f"their floors {tab['floor']}")
        mass, com, inertia = _box_body(tab["base_pos"], tab["base_size"],
                                       tab["density"])
        got = np.concatenate([[mass], com, tab["principal"] @ inertia])
        want = np.concatenate([[m.body_mass[tab["body"]]], m.body_ipos[tab["body"]],
                               m.body_inertia[tab["body"]]])
        if not np.allclose(got, want, rtol=1e-6, atol=1e-12):
            raise RuntimeError("the glove's inertia model disagrees with the "
                               f"compiler ({got} vs {want}): is the glove still "
                               "five boxes, symmetric about x=0 and y=0?")

    def _reset_pose(self, m: mujoco.MjModel) -> np.ndarray:
        """qpos every episode starts from: qpos0 with every ``init_key``
        keyframe laid over it.

        Each source model contributes one, and attaching pads a keyframe with
        qpos0 over the joints it did not write -- so what a keyframe deviates
        from qpos0 by is exactly the part it owns, and the poses compose by
        adding those deviations. Overlapping claims mean two models are writing
        one joint, which is a modelling error rather than something to resolve
        by ordering. The ball's own entries are overwritten by the reset kernel.
        """
        keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i)
                for i in range(m.nkey)]
        qpos, claimed = np.array(m.qpos0, dtype=np.float64), np.zeros(m.nq, bool)
        for name in self.init_key:
            if name not in keys:
                continue
            delta = m.key_qpos[keys.index(name)] - m.qpos0
            mine = np.abs(delta) > 1e-9
            if (mine & claimed).any():
                raise RuntimeError(
                    f"keyframe {name!r} rewrites qpos "
                    f"{np.flatnonzero(mine & claimed).tolist()}, which an earlier "
                    f"keyframe in {list(self.init_key)} already set")
            qpos, claimed = qpos + delta, claimed | mine
        return qpos.astype(np.float32)

    def _alloc_device_state(self) -> None:
        """Per-world buffers the kernels own. None of these are ever rebound --
        the structs below hold warp views straight into their storage."""
        n, dev = self.num_envs, self.device
        self._alloc_action_buffers(self.act_dim(), self.n_offsets)
        self._offsets_t = torch.zeros((n, self.n_offsets), device=dev)

        # Force perturbation variables
        self._force_phase = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._force_cycle = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._force_dir = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        self._tick = torch.zeros(1, dtype=torch.int32, device=dev)
        self._tunables = torch.zeros(kern.N_TUNABLES, device=dev)
        self._wp_tunables = wp.from_torch(self._tunables)
        self._qpos_init = torch.as_tensor(self._init_qpos, device=dev)

        # Host-side episode state: no kernel holds a view, so these may be
        # rebound rather than written in place.
        self._best_dist = torch.zeros(n, device=dev)   # closest so far, per episode
        self._d0 = torch.ones(n, device=dev)
        self._goal_steps = torch.zeros(n, dtype=torch.long, device=dev)
        # Worlds whose physics went non-finite on the last _compute_obs, and the
        # running total. Both device-side: reading either costs a host sync, and
        # the step path already pays for exactly one.
        self._diverged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._diverged_total = torch.zeros((), dtype=torch.long, device=dev)

        # Per-step obs constants, hoisted out of the hot path.
        self._fpos = torch.as_tensor(self.field_pos, device=dev)
        self._fsize = torch.as_tensor(self.field_size, device=dev)
        off, scl = self._qpos_norm()
        self._qpos_off = torch.as_tensor(off, device=dev)
        self._qpos_scl = torch.as_tensor(scl, device=dev)


    def _qpos_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """(offset, scale) mapping the robot's qpos into ~[-1, 1]: free-joint
        position by the field box, quaternions as they are, joints by range."""
        m = self.m
        off = np.zeros(self._nq_robot, dtype=np.float32)
        scl = np.ones(self._nq_robot, dtype=np.float32)
        for j in range(m.njnt):
            adr = int(m.jnt_qposadr[j])
            if adr >= self._nq_robot:
                continue
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                off[adr:adr + 3], scl[adr:adr + 3] = self.field_pos, 1.0 / self.field_size
            elif m.jnt_limited[j]:
                scl[adr] = 1.0 / max(np.abs(m.jnt_range[j]).max(), 1e-6)
        return off, scl

    def _build_kernel_params(self) -> None:
        """Pack the kernel arguments once: array views, then frozen constants."""
        sim = kern.SimArrays()
        for attr in ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl", "xfrc_applied"):
            setattr(sim, attr, getattr(self.gpu_data, attr))
        for attr in ("geom_pos", "geom_size", "geom_rbound", "geom_aabb",
                     "geom_margin", "geom_rgba",
                     "body_pos", "body_ipos", "body_mass", "body_inertia"):
            setattr(sim, attr, getattr(self.gpu_model, attr))
        self._sim = sim

        env = kern.EnvArrays()
        env.mask = wp.from_torch(self.reset_mask)
        env.qpos_init = wp.from_torch(self._qpos_init)
        env.deform = wp.from_torch(self._embodiment_buf)
        env.ctrl_act = wp.from_torch(self._act_buf)
        env.ctrl_scale = wp.from_torch(self.ctrl_scale)
        env.ctrl_mid = wp.from_torch(self.ctrl_mid)
        env.offsets = wp.from_torch(self._offsets_t)
        env.force_phase = wp.from_torch(self._force_phase)
        env.force_cycle = wp.from_torch(self._force_cycle)
        env.force_dir = wp.from_torch(self._force_dir)
        env.tick = wp.from_torch(self._tick)
        self._env = env

        seg = kern.SegArrays()
        for name, val in self._segs.items():
            dtype = _SEG_DTYPE.get(name, wp.float32)
            as_np = np.float32 if dtype in (wp.float32, wp.vec3) else np.int32
            setattr(seg, name, wp.array(val.astype(as_np), dtype=dtype,
                                        device=self.device))
        self._seg = seg

        # Built whether or not the glove deforms, so the kernel never holds an
        # unbound array; c.n_gparam is what switches it on.
        gl = kern.GloveArrays()
        for name, dtype in _GLOVE_DTYPE.items():
            as_np = np.int32 if dtype is wp.int32 else np.float32
            setattr(gl, name, wp.array(self._glove_tab[name].astype(as_np),
                                       dtype=dtype, device=self.device))
        gl.principal = wp.mat33(*self._glove_tab["principal"].ravel())
        for name in ("density", "body", "n_geom"):
            setattr(gl, name, self._glove_tab[name])
        self._glove_arr = gl

        # Likewise always built. gpu_model.geom_margin is only widened to one row
        # per world when disable_legs is set, so nothing may index it by world
        # unless c.n_legparam says the switch exists.
        legs = self._leg_geoms
        leg = kern.LegArrays()
        leg.geom = wp.array(legs.astype(np.int32), dtype=wp.int32, device=self.device)
        leg.margin0 = wp.array(self.m.geom_margin[legs].astype(np.float32),
                               dtype=wp.float32, device=self.device)
        leg.rgba0 = wp.array(self.m.geom_rgba[legs].astype(np.float32),
                             dtype=wp.vec4, device=self.device)
        leg.off_margin, leg.n_geom = _LEG_OFF_MARGIN, len(legs)
        self._leg_arr = leg

        c = kern.CatcherConst()
        c.seed = int(self.seed)
        c.nq, c.nv, c.nu = int(self.m.nq), int(self.m.nv), int(self.m.nu)
        c.n_seg = int(self.n_links)
        c.n_slot = int(self.n_link_slots)
        c.n_gparam = int(self.n_glove_params)
        c.n_legparam = int(self.n_leg_params)
        c.ball_qadr, c.ball_dadr = self._ball_qadr, self._ball_dadr
        c.force_every = max(int(self.force_every), 1)
        c.force_hold = int(self.force_hold)
        c.base_bid = self._base_bid

        self._consts = c

        self._refresh_tunables()

    def _refresh_tunables(self) -> None:
        """Push the host-side mirrors of the live tunables to the device. One
        small H2D copy, valid at any time -- the kernels read these from memory,
        so nothing has to be recaptured."""
        self._tunables.copy_(torch.tensor(
            [self.max_offset, self.pitch_vel, self.pitch_to_radius,
             *self.pitch_from, *self.pitch_to, self.glove_max_offset, self.force_mag],
            dtype=torch.float32))

    def act_dim(self) -> int:
        # ctrl action only; the offset action is a separate argument to step().
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # robot qpos + qvel, ball pos/vel, goal pos/quat/vel, ball - goal, offsets
        return self._nq_robot + self._nv_robot + 19 + self.n_offsets

    def _make_spaces(self) -> None:
        # Both spaces are normalised: obs components are divided by their own
        # scale in _compute_obs, and actions are remapped onto each actuator's
        # ctrl range by ctrl_kernel. The trailing offset dims declare the reset
        # kernel's own clip range instead of [-1, 1], since that is the interval
        # they are actually observed in -- a one-sided link never reports a
        # negative offset, and a search drawing contours from this box should
        # not propose one either.
        low = np.full(self.obs_dim(), -1.0, dtype=np.float32)
        high = np.full(self.obs_dim(), 1.0, dtype=np.float32)
        low[-self.n_offsets:], high[-self.n_offsets:] = self.offset_bounds
        self.single_observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32, seed=self.seed)
        self.single_action_space = spaces.Box(
            low=np.full(self.act_dim(), -1.0, dtype=np.float32),
            high=np.full(self.act_dim(), 1.0, dtype=np.float32),
            dtype=np.float32, seed=self.seed)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    # ------------------------------------------------------------------ #
    # Kernel launches in step (before the physics substeps)
    # ------------------------------------------------------------------ #
    def _launch_reset(self) -> None:
        wp.launch(kern.tick_kernel, dim=1, inputs=[self._env])
        wp.launch(kern.reset_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._seg, self._glove_arr,
                          self._leg_arr, self._consts, self._wp_tunables])

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    def _launch_force(self) -> None:
        wp.launch(kern.force_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts, self._wp_tunables])

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _goal_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and linear velocity of ``goal_site``. The velocity
        comes from ``cvel``, which the step program's closing kinematics pass
        does not refresh, so it lags the position by one substep."""
        site = wp.to_torch(self.gpu_data.site_xpos)[:, self._goal_sid]
        cvel = wp.to_torch(self.gpu_data.cvel)[:, self._goal_bid]
        com = wp.to_torch(self.gpu_data.subtree_com)[:, self._root_bid]
        return site, cvel[:, 3:] + torch.cross(cvel[:, :3], site - com, dim=1)

    def _goal_quat(self) -> torch.Tensor:
        """World orientation of ``goal_site`` as a wxyz quaternion, taken from
        the body quat the kinematics pass already wrote rather than reconstructed
        from ``site_xmat``. Components are in [-1, 1], so it enters the obs
        unscaled; the sign is pinned to w >= 0 so the double cover does not hand
        the policy two encodings of one pose."""
        q = _quat_mul(wp.to_torch(self.gpu_data.xquat)[:, self._goal_bid],
                      self._goal_sq)
        return torch.where(q[:, :1] < 0, -q, q)

    def _ball_pos(self) -> torch.Tensor:
        return self.qpos_torch()[:, self._ball_qadr:self._ball_qadr + 3]

    def _compute_obs(self) -> torch.Tensor:
        qpos, qvel = self.qpos_torch(), self.qvel_torch()
        ball_p = self._ball_pos()
        ball_v = qvel[:, self._ball_dadr:self._ball_dadr + 3]
        goal_p, goal_v = self._goal_state()
        obs = torch.cat([
            (qpos[:, :self._nq_robot] - self._qpos_off) * self._qpos_scl,
            qvel[:, :self._nv_robot] / self.vel_scale,
            (ball_p - self._fpos) / self._fsize,
            ball_v / self.vel_scale,
            (goal_p - self._fpos) / self._fsize,
            self._goal_quat(),
            goal_v / self.vel_scale,
            (ball_p - goal_p) / self.max_rel_dist,
            self._offsets_t,
        ], dim=1)
        if self.check_finite:
            self._assert_finite(obs, "obs")
        # A world whose physics has diverged has no usable state left, and it
        # cannot end its own episode: every terminal test below is a comparison,
        # and a comparison against NaN is False. So flag it here, terminate it in
        # _reward_terminated_truncated, and hand back a finite placeholder row --
        # not a repair, just "nothing to report" in a normalised obs space. The
        # respawn on the next step rebuilds the world from scratch.
        self._diverged = ~torch.isfinite(obs).all(dim=1)
        self._diverged_total += self._diverged.sum()
        return obs.masked_fill_(self._diverged[:, None], 0.0)

    def sample_observations(self, n: int = 1) -> torch.Tensor:
        """``n`` observations drawn from the distribution of episode starts.

        ``observation_space`` is a bounding hull, not the reachable set, so
        sampling it per dimension yields rows the physics could never hand back.
        Respawning and reading the result keeps every coupling exact.

        This *resets the batch*: use a throwaway env, not one mid-rollout.
        ``reset`` also zeroes the staged offset action, so the arm is the
        undeformed one.
        """
        rows, collected = [], 0
        while collected < int(n):
            rows.append(self.reset().clone())
            collected += rows[-1].shape[0]
        return torch.cat(rows)[:int(n)]

    def _reward_terminated_truncated(self):
        ball = self._ball_pos()
        goal, _ = self._goal_state()
        dist = torch.linalg.norm(ball - goal, dim=1)

        # Worlds the reset kernel respawned at the head of this step: reset_mask
        # still holds the previous step's dones, since step() overwrites it only
        # after this returns. They have made no progress yet, and their episode's
        # reward scale is fixed here from the distance they start at.
        fresh = self.reset_mask
        self._best_dist = torch.where(fresh, dist, self._best_dist)
        self._d0 = torch.where(fresh, dist.clamp(min=1e-3), self._d0)

        at_goal = dist < self.goal_radius
        # Total time at the goal this episode, not the current unbroken run
        self._goal_steps = self._goal_steps.masked_fill(fresh, 0) + at_goal.long()
        held_at_goal = (self._goal_steps >= self.goal_dwell_steps
                        if self.goal_dwell_steps > 0
                        else torch.zeros_like(at_goal))

        reward = (self.prog_gain / self._d0) * (self._best_dist - dist).clamp(min=0)
        reward = reward + at_goal.float() * self.at_goal_rew
        self._best_dist = torch.minimum(self._best_dist, dist)

        left_field = ((ball - self._fpos).abs() > self._fsize).any(dim=1)
        goal_left_field = ((goal - self._fpos).abs() > self._fsize).any(dim=1)

        # See https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
        # for the difference between termination and truncation. A caught ball
        # and a lost one both genuinely end the episode, so a value function must
        # not bootstrap past them (unlike the time limit); a diverged world has
        # no state to bootstrap from either.
        terminated = held_at_goal | self._diverged
        if self.terminate_out_of_field:
            terminated = terminated | left_field | goal_left_field
        truncated = self.world_steps >= self.max_episode_steps
        if self.check_finite:
            self._assert_finite(reward, "reward")
        reward = reward.masked_fill(self._diverged, 0.0)

        self.last_at_goal = at_goal
        self.last_held_at_goal = held_at_goal
        self.last_left_field = left_field
        self.last_time_limit = truncated
        self.last_diverged = self._diverged
        return reward, terminated, truncated

    @property
    def diverged_worlds(self) -> int:
        """Worlds ended so far because their physics went non-finite.

        Should stay at 0; a count that climbs is a physics problem, not a
        nuisance to be filtered out. Reading this syncs with the device, so read
        it at an evaluation boundary, not in the inner rollout loop.
        """
        return int(self._diverged_total)

    # ------------------------------------------------------------------ #
    # Live tunables + host-side views
    # ------------------------------------------------------------------ #
    def change_max_offset(self, new_offset: float | None = None,
                          glove_offset: float | None = None) -> None:
        """Set the length change [m] a +/-1 offset action buys, for the
        offsettable links and/or the glove's dimensions. None keeps a value."""
        if new_offset is not None:
            self.max_offset = float(new_offset)
        if glove_offset is not None:
            self.glove_max_offset = float(glove_offset)
        self._refresh_tunables()

    def set_pitch(self, vel: float | None = None, to_radius: float | None = None,
                  pitch_from=None, pitch_to=None) -> None:
        """Retarget the pitch mid-run (a curriculum knob); None keeps a value."""
        if vel is not None:
            self.pitch_vel = float(vel)
        if to_radius is not None:
            self.pitch_to_radius = float(to_radius)
        if pitch_from is not None:
            self.pitch_from = np.asarray(pitch_from, dtype=np.float32)
        if pitch_to is not None:
            self.pitch_to = np.asarray(pitch_to, dtype=np.float32)
        self._refresh_tunables()

    def set_force_mag(self, newtons: float) -> None:
        """Set the perturbation force magnitude (0 to disable)."""
        self.ball_force_mag = float(newtons)
        self._refresh_tunables()

    @property
    def env_offsets(self) -> np.ndarray:
        """Host copy of the per-world offsets, in **metres**. The device side
        holds the normalised form, so this scales on the way out -- by
        ``max_offset`` for the links, ``glove_max_offset`` for the glove. The
        leg switch is a flag, not a length, and passes through as 0 or 1."""
        scale = np.full(self.n_offsets, self.max_offset, dtype=np.float32)
        scale[self.n_link_slots:self.n_link_slots + self.n_glove_params] = self.glove_max_offset
        scale[self.n_link_slots + self.n_glove_params:] = 1.0
        return self._offsets_t.detach().cpu().numpy() * scale

    @property
    def env_offsets_norm(self) -> np.ndarray:
        """Host copy of the offsets, normalised to [-1, 1]. Matches the trailing
        dims of the observation and of an offset action."""
        return self._offsets_t.detach().cpu().numpy()

    @property
    def offset_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) the reset kernel clips each offset entry to, and hence the
        interval the observation space declares for them.

        A ``*_pos`` link only lengthens, so anything below zero is the
        undeformed arm. Were this [-1, 1] throughout, a search drawing contours
        from the observation space would spend half its pool on candidates that
        are physically identical to the zero one -- and score them with a critic
        that has never seen an offset outside this range in any transition.
        """
        lo = np.full(self.n_offsets, -1.0, dtype=np.float32)
        hi = np.ones(self.n_offsets, dtype=np.float32)
        lo[self._segs["act"]] = self._segs["lo"]  # group members agree, by build check
        hi[self._segs["act"]] = self._segs["hi"]
        if self.n_glove_params:                  # a dimension may not shrink past its floor
            g, end = self._glove_tab, self.n_link_slots + self.n_glove_params
            lo[self.n_link_slots:end] = np.maximum(
                (g["floor"] - g["base"]) / max(self.glove_max_offset, 1e-9), -1.0)
        return lo, hi

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _mirror_render_model(self, env_idx: int | None = None) -> None:
        """Copy one world's deformed links back into its CPU ``MjModel``.

        ``render()`` calls this for the world it is about to draw; ``step()``
        calls it with no argument for the interactive viewer's world. Without it
        the CPU model still holds the compile-time shape, since the reset kernel
        deforms only the device-side model.
        """
        w = self.viz_env_id if env_idx is None else int(env_idx)
        v, gm = self.variants[w], self.gpu_model
        g, b, kid = self._segs["geom"], self._segs["body"], self._segs["child_id"]
        if self.offset_glove:                    # the glove's boxes deform too
            g = np.concatenate([g, self._glove_tab["geom"]])
            b = np.concatenate([b, [self._glove_tab["body"]]])
        v.geom_pos[g] = wp.to_torch(gm.geom_pos)[w, g].cpu().numpy()
        # body_ipos is not just inertia here: a geom whose frame the compiler
        # folded into the body's inertial frame (geom_sameframe == 2, which is
        # what a rotated single-geom link gets) is *drawn* from body_ipos, and
        # geom_pos is ignored. Leaving it stale grows that geom about its old
        # centre.
        v.body_ipos[b] = wp.to_torch(gm.body_ipos)[w, b].cpu().numpy()
        v.geom_size[g] = wp.to_torch(gm.geom_size)[w, g].cpu().numpy()
        v.geom_rbound[g] = wp.to_torch(gm.geom_rbound)[w, g].cpu().numpy()
        v.geom_aabb[g] = wp.to_torch(gm.geom_aabb)[w, g].reshape(len(g), 6).cpu().numpy()
        v.body_pos[kid] = wp.to_torch(gm.body_pos)[w, kid].cpu().numpy()
        if self.disable_legs:            # switched-off chains are drawn at alpha 0
            lg = self._leg_geoms
            v.geom_rgba[lg] = wp.to_torch(gm.geom_rgba)[w, lg].cpu().numpy()
        # The visualizer culls against bvh_aabb, which is baked at compile time;
        # a one-geom body's box is just its geom's aabb.
        for bid, gid in zip(self._segs["body"], self._segs["geom"]):
            adr = int(v.body_bvhadr[bid])
            if adr >= 0 and v.body_bvhnum[bid] == 1 and v.body_geomnum[bid] == 1:
                v.bvh_aabb[adr] = v.geom_aabb[gid]
        if self.offset_glove:
            self._mirror_glove_bvh(v)

    def _mirror_glove_bvh(self, v: mujoco.MjModel) -> None:
        """The glove's own culling boxes. Its hierarchy has interior nodes, whose
        children must stay inside them, so every node gets the same box bounding
        the whole glove -- an oversized box costs culling, never correctness.
        Bounding volumes are stored in the body's *inertial* frame."""
        gid, bid = self._glove_tab["geom"], self._glove_tab["body"]
        lo = (v.geom_pos[gid] - v.geom_size[gid]).min(0)
        hi = (v.geom_pos[gid] + v.geom_size[gid]).max(0)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, v.body_iquat[bid])
        rot = rot.reshape(3, 3).T
        adr, num = int(v.body_bvhadr[bid]), int(v.body_bvhnum[bid])
        v.bvh_aabb[adr:adr + num] = np.concatenate([
            rot @ ((lo + hi) / 2 - v.body_ipos[bid]), np.abs(rot) @ ((hi - lo) / 2)])

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Draw the ball -> goal_site vector the reward is computed from."""
        if scene.ngeom >= scene.maxgeom:
            return

        # ball > goal_site vector logic
        ball = self._ball_pos()[env_idx].detach().cpu().numpy().astype(np.float64)
        goal = wp.to_torch(self.gpu_data.site_xpos)[env_idx, self._goal_sid]
        goal = goal.detach().cpu().numpy().astype(np.float64)
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array([0.2, 0.9, 0.3, 0.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.01, ball, goal)
        scene.ngeom += 1


        # body force vector logic
        if not self.force_on:
            return
        f = wp.to_torch(self.gpu_data.xfrc_applied)[env_idx, self._base_bid, :2]
        f = f.detach().cpu().numpy().astype(np.float64)
        mag = float(np.linalg.norm(f))
        if mag < 1e-9:
            return

        base = self.xpos_torch()[env_idx, self._base_bid].detach().cpu().numpy()
        base = base.astype(np.float64)
        tip = base + np.array([f[0], f[1], 0.0]) / mag * (
            mag * self.force_viz_scale)

        # might need to make the force width parametric instead of 0.1
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array([1.0, 0.25, 0.1, 0.9], dtype=np.float32))
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_ARROW, 0.1, base, tip)
        scene.ngeom += 1


    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0, 0,
                            self.field_pos[2]]
        camera.elevation = -15.0
        camera.azimuth = 135.0
        camera.distance = 3.0 * float(self.field_size.max())
