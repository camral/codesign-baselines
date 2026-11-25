""" A "glove" for ball_catcher_env.py. Should be attached to the 'arm_tip' site of the arm. """

from __future__ import annotations

import mujoco

_RGBA = (0.55, 0.38, 0.22, 1)


def glove_layout(
    width: float = 0.2,
    height: float = 0.2,
    wall_h: float = 0.1,
    thickness: float = 0.02,
) -> list[tuple[str, list[float], list[float]]]:
    """(name, pos, half-size) of every glove box, in the glove's body frame.

    The one description of the shape: the builder below lays the glove out from
    it, and the env differentiates it to deform the glove per world. Every entry
    is affine in the three dimensions, which is what makes that possible.
    """
    # comment back and left for baseball glove
    hw, hh, t = width / 2.0, height / 2.0, thickness
    return [
        ("glove_bottom", [0, 0, 0], [hh, hw - t/2, t / 2]),
        ("glove_front", [-hh - t/2, 0, wall_h / 2], [t / 2, hw + t/2, wall_h / 2 + t/2]),
        ("glove_back", [hh + t/2, 0, wall_h / 2], [t / 2, hw + t/2, wall_h / 2 + t/2]),
        ("glove_left", [0, -hw, wall_h / 2], [hh, t / 2, wall_h / 2 + t/2]),
        ("glove_right", [0, hw, wall_h / 2], [hh, t / 2, wall_h / 2 + t/2]),
    ]


def add_glove(
    spec: mujoco.MjSpec,
    *,
    width: float = 0.2,
    height: float = 0.2,
    wall_h: float = 0.1,
    thickness: float = 0.02,
    goal_radius = 0.1,
    rgba=_RGBA,
) -> "mujoco.MjsBody":

    site = spec.site("arm_tip")
    body = site.parent.add_body(name="glove", pos=site.pos, quat=site.quat)
    for name, pos, size in glove_layout(width, height, wall_h, thickness):
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos,
                      size=size, rgba=list(rgba))
    body.add_site(name="goal_site", pos=[0, 0, goal_radius], group=2, size=goal_radius, rgba=[0, 1, 0, 0.2])

    return body
