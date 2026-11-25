"""Device code for the ball-catcher task: everything that runs inside the graph.

Launched in this order at the head of every control step:

1. ``tick_kernel``  -- advance the shared RNG stream index.
2. ``reset_kernel`` -- respawn + re-deform the worlds whose ``mask`` bit is set.
3. ``ctrl_kernel``  -- map the policy action onto actuator ctrl, after the reset.

Per-step state arrives through the arrays in ``EnvArrays``/``SimArrays`` or the
live-tunables array; anything fixed for the run rides in ``CatcherConst`` or the
host-built ``SegArrays``/``GloveArrays``, since launch arguments freeze at
capture time.
"""

from __future__ import annotations

import math

import warp as wp

_PI = math.pi

# --- live tunables ----------------------------------------------------- #
# Read from device memory, so a curriculum can change them mid-run with a single
# device write: no graph recapture, no host sync.
T_MAX_OFFSET = 0     # metres a full-deflection (+/-1) offset action stretches by
T_PITCH_VEL = 1      # ball launch speed
T_TGT_RADIUS = 2     # radius of the ball the pitch target is sampled in
T_FROM = 3           # pitch origin, 3 floats
T_TO = 6             # pitch target centre, 3 floats
T_GLOVE_OFFSET = 9   # metres a full-deflection glove offset action stretches by
T_FORCE_MAG = 10       # disturbance force magnitude, newtons
N_TUNABLES = 11


@wp.struct
class SimArrays:
    """Batched mujoco_warp model/data fields (zero-copy views)."""
    qpos: wp.array2d(dtype=wp.float32)
    qvel: wp.array2d(dtype=wp.float32)
    # Solver state, carried from step to step rather than recomputed.
    qacc: wp.array2d(dtype=wp.float32)
    qacc_warmstart: wp.array2d(dtype=wp.float32)
    time: wp.array(dtype=wp.float32)
    ctrl: wp.array2d(dtype=wp.float32)
    geom_pos: wp.array2d(dtype=wp.vec3)
    geom_size: wp.array2d(dtype=wp.vec3)
    geom_rbound: wp.array2d(dtype=wp.float32)
    geom_aabb: wp.array3d(dtype=wp.vec3)
    geom_margin: wp.array2d(dtype=wp.float32)
    geom_rgba: wp.array2d(dtype=wp.vec4)
    body_pos: wp.array2d(dtype=wp.vec3)
    body_ipos: wp.array2d(dtype=wp.vec3)
    body_mass: wp.array2d(dtype=wp.float32)
    body_inertia: wp.array2d(dtype=wp.vec3)
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector)


@wp.struct
class EnvArrays:
    """Env-owned per-world state. Allocated once, never rebound."""
    mask: wp.array(dtype=wp.bool)             # worlds to respawn this step
    qpos_init: wp.array(dtype=wp.float32)     # reset pose (keyframe or qpos0)
    deform: wp.array2d(dtype=wp.float32)      # staged reset action, [-1, 1]
    ctrl_act: wp.array2d(dtype=wp.float32)    # staged policy action, [-1, 1]
    ctrl_scale: wp.array(dtype=wp.float32)    # [-1,1] -> actuator ctrlrange
    ctrl_mid: wp.array(dtype=wp.float32)
    offsets: wp.array2d(dtype=wp.float32)     # offsets in force, normalised
    force_phase: wp.array(dtype=wp.int32)
    force_cycle: wp.array(dtype=wp.int32)
    force_dir: wp.array2d(dtype=wp.float32)
    tick: wp.array(dtype=wp.int32)            # step counter, the RNG stream


@wp.struct
class SegArrays:
    """One entry per offsettable link, in action order. Host-built, constant."""
    geom: wp.array(dtype=wp.int32)
    body: wp.array(dtype=wp.int32)
    dir: wp.array(dtype=wp.vec3)          # unit stretch direction, body frame
    radius: wp.array(dtype=wp.float32)
    half0: wp.array(dtype=wp.float32)     # undeformed half-length
    density: wp.array(dtype=wp.float32)
    is_cyl: wp.array(dtype=wp.int32)      # else capsule
    lo: wp.array(dtype=wp.float32)        # action clip, from the body name
    hi: wp.array(dtype=wp.float32)
    act: wp.array(dtype=wp.int32)         # action slot; an _equ group shares one
    child_adr: wp.array(dtype=wp.int32)   # ragged block of bodies to keep
    child_num: wp.array(dtype=wp.int32)   # attached at the link's moving end
    child_id: wp.array(dtype=wp.int32)
    child_pos: wp.array(dtype=wp.vec3)    # their undeformed body_pos


@wp.struct
class GloveArrays:
    """The glove's boxes. Host-built, constant. The last three entries of the
    reset action stretch its (width, height, wall_h) when ``n_gparam`` is 3."""
    geom: wp.array(dtype=wp.int32)
    base_pos: wp.array(dtype=wp.vec3)     # undeformed pose/half-size per box,
    base_size: wp.array(dtype=wp.vec3)    # and their derivatives wrt each param
    dpos: wp.array2d(dtype=wp.vec3)       # [box, param]
    dsize: wp.array2d(dtype=wp.vec3)
    base: wp.array(dtype=wp.float32)      # undeformed dimension, per param
    floor: wp.array(dtype=wp.float32)     # smallest it may shrink to
    principal: wp.mat33                   # body-frame inertia -> body_iquat frame
    density: float
    body: int
    n_geom: int


@wp.struct
class LegArrays:
    """Geoms of the leg chains the reset action can switch off, and the values
    to put back when it switches them on. Host-built, constant."""
    geom: wp.array(dtype=wp.int32)
    margin0: wp.array(dtype=wp.float32)   # compile-time contact margin
    rgba0: wp.array(dtype=wp.vec4)
    off_margin: float                     # margin that suppresses every contact
    n_geom: int


@wp.struct
class CatcherConst:
    """Addresses and counts - nothing here changes."""
    seed: int
    nq: int
    nv: int
    nu: int
    n_seg: int
    n_slot: int          # link action slots; less than n_seg under _equ groups
    n_gparam: int        # 3 if the action also deforms the glove, else 0
    n_legparam: int      # 1 if the action can switch the leg chains off, else 0
    ball_qadr: int
    ball_dadr: int
    force_every: int
    force_hold: int
    base_bid: int

@wp.kernel
def tick_kernel(env: EnvArrays):
    """Advance the RNG stream index. One thread, once per step: the graph bakes
    launch arguments in, so a host-side seed would replay one spawn forever."""
    env.tick[0] = env.tick[0] + 1


@wp.func
def _sq_pairs(v: wp.vec3) -> wp.vec3:
    """(y^2+z^2, z^2+x^2, x^2+y^2) -- the diagonal of a box's inertia and of the
    parallel-axis shift both take this form."""
    return wp.vec3(v[1] * v[1] + v[2] * v[2], v[2] * v[2] + v[0] * v[0],
                   v[0] * v[0] + v[1] * v[1])


@wp.func
def _mass_inertia(half: float, r: float, rho: float, is_cyl: int) -> wp.vec4:
    """(mass, Ixx, Iyy, Izz) of a capsule/cylinder about its centre, in the geom
    frame (long axis = local z). Matches MuJoCo's ``inertiafromgeom``."""
    length = 2.0 * half
    m_cyl = rho * _PI * r * r * length
    transverse = m_cyl * (3.0 * r * r + length * length) / 12.0
    if is_cyl == 1:
        return wp.vec4(m_cyl, transverse, transverse, 0.5 * m_cyl * r * r)
    m_cap = rho * (2.0 / 3.0) * _PI * r * r * r
    transverse += 2.0 * m_cap * (0.4 * r * r + length * length / 4.0
                                 + 0.375 * r * length)
    axial = 0.5 * m_cyl * r * r + 0.8 * m_cap * r * r
    return wp.vec4(m_cyl + 2.0 * m_cap, transverse, transverse, axial)


@wp.func
def _deform(sim: SimArrays, seg: SegArrays, w: int, i: int, delta: float):
    """Stretch link ``i`` of world ``w`` by ``delta`` metres along its own axis.

    Everything is rebuilt from the undeformed values rather than accumulated, so
    re-running this with a new action is exact. Bodies hanging off the link's far
    end travel with it; ``body_iquat`` needs no update because the geom's frame
    (its principal axes) does not rotate.
    """
    g = seg.geom[i]
    r = seg.radius[i]
    half = seg.half0[i] + 0.5 * delta
    pos = seg.dir[i] * half
    sim.geom_pos[w, g] = pos
    sim.geom_size[w, g] = wp.vec3(r, half, 0.0)
    # Broadphase bounds must follow the new size or a stretched link is culled.
    if seg.is_cyl[i] == 1:
        sim.geom_rbound[w, g] = wp.sqrt(r * r + half * half)
        sim.geom_aabb[w, g, 1] = wp.vec3(r, r, half)
    else:
        sim.geom_rbound[w, g] = half + r
        sim.geom_aabb[w, g, 1] = wp.vec3(r, r, half + r)
    sim.geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)

    b = seg.body[i]
    mi = _mass_inertia(half, r, seg.density[i], seg.is_cyl[i])
    sim.body_mass[w, b] = mi[0]
    sim.body_inertia[w, b] = wp.vec3(mi[1], mi[2], mi[3])
    sim.body_ipos[w, b] = pos

    for k in range(seg.child_num[i]):
        j = seg.child_adr[i] + k
        sim.body_pos[w, seg.child_id[j]] = seg.child_pos[j] + seg.dir[i] * delta


@wp.func
def _glove_delta(env: EnvArrays, gl: GloveArrays, w: int, i: int, p: int,
                 gmax: float) -> float:
    """Clip glove parameter ``p``'s action so the dimension cannot shrink past
    its floor, record what was realised, and return the change in metres."""
    lo = (gl.floor[p] - gl.base[p]) / wp.max(gmax, 1.0e-9)
    o = wp.clamp(env.deform[w, i], wp.max(lo, -1.0), 1.0)
    env.offsets[w, i] = o
    return o * gmax


@wp.func
def _deform_glove(sim: SimArrays, gl: GloveArrays, w: int, d: wp.vec3):
    """Stretch the glove of world ``w`` by ``d`` metres of (width, height,
    wall_h).

    The layout is affine in the three, so every box's pose and size come off its
    base value and stored derivatives. Mass and inertia are summed over the boxes
    about the body origin and shifted to the new centre of mass; the sum stays
    diagonal in the body frame because the walls are symmetric about x=0 and y=0,
    and ``principal`` rotates that diagonal into the frame ``body_iquat`` froze at
    compile time.
    """
    mass = float(0.0)
    mom = wp.vec3(0.0, 0.0, 0.0)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(gl.n_geom):
        g = gl.geom[k]
        p = gl.base_pos[k] + gl.dpos[k, 0] * d[0] + gl.dpos[k, 1] * d[1] + gl.dpos[k, 2] * d[2]
        s = gl.base_size[k] + gl.dsize[k, 0] * d[0] + gl.dsize[k, 1] * d[1] + gl.dsize[k, 2] * d[2]
        sim.geom_pos[w, g] = p
        sim.geom_size[w, g] = s
        sim.geom_rbound[w, g] = wp.length(s)
        sim.geom_aabb[w, g, 0] = wp.vec3(0.0, 0.0, 0.0)
        sim.geom_aabb[w, g, 1] = s
        mi = gl.density * 8.0 * s[0] * s[1] * s[2]
        mass += mi
        mom += mi * p
        acc += mi * (_sq_pairs(s) / 3.0 + _sq_pairs(p))
    com = mom / mass
    sim.body_mass[w, gl.body] = mass
    sim.body_ipos[w, gl.body] = com
    sim.body_inertia[w, gl.body] = gl.principal * (acc - mass * _sq_pairs(com))


@wp.func
def _set_legs(sim: SimArrays, leg: LegArrays, w: int, on: bool):
    """Put the switchable leg chains of world ``w`` in or out of the world.

    Contact filtering by contype/conaffinity is baked into a geom-pair list when
    the model is uploaded, so it cannot vary per world. The contact margin can:
    it is read per world both in the broadphase and where a contact would be
    written, and a margin this negative fails every one of those tests, so the
    chain generates no contact with anything. Zero alpha then takes it out of the
    picture as well.
    """
    for k in range(leg.n_geom):
        g = leg.geom[k]
        c = leg.rgba0[k]
        sim.geom_margin[w, g] = wp.where(on, leg.margin0[k], leg.off_margin)
        sim.geom_rgba[w, g] = wp.where(on, c, wp.vec4(c[0], c[1], c[2], 0.0))


@wp.kernel
def reset_kernel(env: EnvArrays, sim: SimArrays, seg: SegArrays,
                 gl: GloveArrays, leg: LegArrays, c: CatcherConst,
                 tun: wp.array(dtype=wp.float32)):
    """Respawn + re-deform one world per thread, wholly on device.

    Threads whose ``mask`` bit is clear return immediately, so this can sit
    unconditionally at the head of the captured step graph instead of being gated
    by a host round-trip.
    """
    w = wp.tid()
    if not env.mask[w]:
        return

    for j in range(c.nq):
        sim.qpos[w, j] = env.qpos_init[j]
    # A respawn must leave no solver state behind. qacc_warmstart seeds the next
    # solve, and it is largest exactly at a terminal state; carried into a fresh
    # episode it can diverge, and its output is then the next warm start -- a
    # ladder that ends in a NaN no later reset can clear.
    for j in range(c.nv):
        sim.qvel[w, j] = 0.0
        sim.qacc[w, j] = 0.0
        sim.qacc_warmstart[w, j] = 0.0
    sim.time[w] = 0.0

    env.force_phase[w] = 0

    # Ball: launched from pitch_from straight at a target drawn uniformly from a
    # ball about pitch_to. Gravity does the rest, so the flight is an arc.
    rs = wp.rand_init(c.seed + env.tick[0], w)
    src = wp.vec3(tun[T_FROM], tun[T_FROM + 1], tun[T_FROM + 2])
    tgt = wp.vec3(tun[T_TO], tun[T_TO + 1], tun[T_TO + 2])
    tgt += wp.sample_unit_sphere(rs) * tun[T_TGT_RADIUS]
    aim = tgt - src
    dist = wp.length(aim)
    vel = wp.vec3(0.0, 0.0, 0.0)
    if dist > 1.0e-9:
        vel = aim / dist * tun[T_PITCH_VEL]

    for j in range(3):
        sim.qpos[w, c.ball_qadr + j] = src[j]
        sim.qvel[w, c.ball_dadr + j] = vel[j]
        sim.qvel[w, c.ball_dadr + 3 + j] = 0.0
    sim.qpos[w, c.ball_qadr + 3] = 1.0
    sim.qpos[w, c.ball_qadr + 4] = 0.0
    sim.qpos[w, c.ball_qadr + 5] = 0.0
    sim.qpos[w, c.ball_qadr + 6] = 0.0

    # Stored normalised, in the same [-1, 1] units the action arrives in and the
    # observation declares; max_offset is applied here and nowhere else. Links of
    # an _equ group read one slot, so they stretch by the same amount.
    for i in range(c.n_seg):
        a = seg.act[i]
        o = wp.clamp(env.deform[w, a], seg.lo[i], seg.hi[i])
        env.offsets[w, a] = o
        _deform(sim, seg, w, i, o * tun[T_MAX_OFFSET])

    # The glove's three dimensions ride on the end of the same action, under
    # their own tunable: they stretch a box, not a link, so a shared scale would
    # be a coincidence rather than a shape.
    if c.n_gparam > 0:
        i = c.n_slot
        gmax = tun[T_GLOVE_OFFSET]
        _deform_glove(sim, gl, w, wp.vec3(
            _glove_delta(env, gl, w, i, 0, gmax),
            _glove_delta(env, gl, w, i + 1, 1, gmax),
            _glove_delta(env, gl, w, i + 2, 2, gmax)))

    # A switch, not a magnitude: the leg chains are on above zero and gone at or
    # below it, so the all-zeros action is the legless world. What is recorded is
    # the state that resulted, which is what the observation should show.
    if c.n_legparam > 0:
        i = c.n_slot + c.n_gparam
        on = env.deform[w, i] > 0.0
        env.offsets[w, i] = wp.where(on, 1.0, 0.0)
        _set_legs(sim, leg, w, on)


@wp.kernel
def force_kernel(env: EnvArrays, sim: SimArrays, c: CatcherConst,
                      tun: wp.array(dtype=wp.float32)):
    """Random-heading disturbance force on the ball, one thread per world.

    Every ``force_every`` control steps a fresh heading is drawn and held for
    ``force_hold`` of them; the force is released for the rest of the cycle.
    Both the phase and the RNG stream index live in device arrays, never in
    kernel parameters, because this kernel is recorded into the graph.

    ``xfrc_applied`` is a spatial vector whose top half is the linear force
    (mjwarp's ``transform_force`` reads it as ``spatial_top``), applied at the
    body's centre of mass in world coordinates -- a pure force, no torque.
    """
    w = wp.tid()

    # random perturbation logic:
    p = env.force_phase[w]

    if p == 0:
        cyc = env.force_cycle[w]
        rs = wp.rand_init(c.seed + cyc, w)
        ang = wp.randf(rs) * _PI * 2.0
        env.force_dir[w, 0] = wp.cos(ang)
        env.force_dir[w, 1] = wp.sin(ang)
        env.force_cycle[w] = cyc + 1

    mag = tun[T_FORCE_MAG]
    if p >= c.force_every - c.force_hold:
        sim.xfrc_applied[w, c.base_bid] = wp.spatial_vector(
            env.force_dir[w, 0] * mag, env.force_dir[w, 1] * mag,
            0.0, 0.0, 0.0, 0.0)
    else:
        sim.xfrc_applied[w, c.base_bid] = wp.spatial_vector(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    env.force_phase[w] = (p + 1) % c.force_every


@wp.kernel
def ctrl_kernel(env: EnvArrays, sim: SimArrays, c: CatcherConst):
    """Map the staged action onto actuator ctrl, one thread per world.

    Runs after the reset kernel, so a world respawned this step is commanded from
    its new pose rather than from the terminal pose of the episode that ended.
    """
    w = wp.tid()
    for a in range(c.nu):
        sim.ctrl[w, a] = (wp.clamp(env.ctrl_act[w, a], -1.0, 1.0)
                          * env.ctrl_scale[a] + env.ctrl_mid[a])
