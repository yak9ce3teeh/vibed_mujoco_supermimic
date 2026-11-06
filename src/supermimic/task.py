"""Flat-ground motion tracking: coordinate features, reward, and fall rule."""

import mujoco
import numpy as np
import torch

from .motion import JOINT_NAMES, ROOT_POS, ROOT_QUAT, JOINT_POS, ROOT_LINVEL, ROOT_ANGVEL, JOINT_VEL


def rotate(q, v):
    u = q[..., 1:]
    return (
        v
        + 2 * q[..., :1] * torch.cross(u, v, dim=-1)
        + 2 * torch.cross(u, torch.cross(u, v, dim=-1), dim=-1)
    )


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def product(a, b):
    return torch.cat(
        (
            a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
            a[..., :1] * b[..., 1:]
            + b[..., :1] * a[..., 1:]
            + torch.cross(a[..., 1:], b[..., 1:], dim=-1),
        ),
        -1,
    )


def heading(q):
    direction = rotate(q, torch.zeros_like(q[..., 1:]) + q.new_tensor([1.0, 0.0, 0.0]))
    half = -0.5 * torch.atan2(direction[..., 1], direction[..., 0])
    z = torch.zeros_like(half)
    return torch.stack((half.cos(), z, z, half.sin()), -1)


def observation(state, previous, default):
    q = state[..., ROOT_QUAT]
    gravity = rotate(
        conjugate(q), torch.zeros_like(state[..., :3]) + state.new_tensor([0.0, 0.0, -1.0])
    )
    return torch.cat(
        (
            state[..., JOINT_POS] - default,
            state[..., JOINT_VEL],
            state[..., ROOT_ANGVEL],
            gravity,
            previous,
        ),
        -1,
    )


def goal_features(state, reference):
    """Heading-relative future target, no future simulated states."""
    q = state[..., ROOT_QUAT]
    h = heading(q)
    while h.ndim < reference.ndim:
        h = h.unsqueeze(-2)
    h = h.expand(reference.shape[:-1] + (4,))
    pos = state[..., :3].unsqueeze(-2)
    relative_q = product(h, reference[..., ROOT_QUAT])
    x = rotate(
        relative_q, torch.zeros_like(reference[..., :3]) + reference.new_tensor([1.0, 0.0, 0.0])
    )
    z = rotate(
        relative_q, torch.zeros_like(reference[..., :3]) + reference.new_tensor([0.0, 0.0, 1.0])
    )
    return torch.cat(
        (
            rotate(h, reference[..., :3] - pos),
            x,
            z,
            reference[..., JOINT_POS],
            reference[..., JOINT_VEL],
            rotate(h, reference[..., ROOT_LINVEL]),
            reference[..., ROOT_ANGVEL],
        ),
        -1,
    )


def squared_angle(a, b):
    relative = product(conjugate(a), b)
    angle = 2 * torch.atan2(
        torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs()
    )
    return angle.square()


def termination_errors(state, reference):
    up = torch.zeros_like(state[..., :3]) + state.new_tensor([0.0, 0.0, -1.0])
    tilt = rotate(conjugate(state[..., 3:7]), up)[..., 2]
    target_tilt = rotate(conjugate(reference[..., 3:7]), up)[..., 2]
    return (state[..., 2] - reference[..., 2]).abs(), (tilt - target_tilt).abs()


def tracking_terms(state, reference, bodies, target_bodies, action, previous, limits, dt):
    pos, quat, vel, omega = bodies
    rp, rq, rv, rw = target_bodies
    kernels = [
        ((pos - rp).square().sum(-1).mean(-1), 0.3),
        (squared_angle(quat, rq).mean(-1), 0.4),
        ((vel - rv).square().sum(-1).mean(-1), 1.0),
        ((omega - rw).square().sum(-1).mean(-1), 3.14),
    ]
    terms = {
        name: torch.exp(-error / scale**2) * dt
        for name, (error, scale) in zip(
            ("body_position", "body_orientation", "body_linear_velocity", "body_angular_velocity"),
            kernels,
        )
    }
    terms["root_position"] = (
        0.5 * torch.exp(-(state[..., :3] - reference[..., :3]).square().sum(-1) / 0.3**2) * dt
    )
    terms["root_orientation"] = (
        0.5 * torch.exp(-squared_angle(state[..., 3:7], reference[..., 3:7]) / 0.4**2) * dt
    )
    terms["action_rate"] = -0.1 * (action - previous).square().sum(-1) * dt
    joint = state[..., 7:36]
    terms["joint_limits"] = (
        -2
        * ((limits[:, 0] - joint).clamp_min(0) + (joint - limits[:, 1]).clamp_min(0)).sum(-1)
        * dt
    )
    return terms


class TrackingTask(torch.nn.Module):
    def __init__(self, model, dt=0.02):
        super().__init__()
        from .kinematics import Kinematics

        self.fk = Kinematics(model)
        self.dt = dt
        limits = model.jnt_range[[model.joint(n).id for n in JOINT_NAMES]].copy()
        self.register_buffer("limits", torch.as_tensor(limits, dtype=torch.float64))

    def forward(self, state, reference, action, previous):
        terms = tracking_terms(
            state,
            reference,
            self.fk(state),
            self.fk(reference),
            action,
            previous,
            self.limits.to(state),
            self.dt,
        )
        height, tilt = termination_errors(state, reference)
        alive = torch.sigmoid((0.25 - height) / 0.02) * torch.sigmoid((0.8 - tilt) / 0.04)
        return sum(terms.values()), alive, terms


def reward(env, motion, frame, action, previous):
    """The same task kernels, evaluated against actual native rigid-body fields."""
    d, m = env.data, env.model
    velocity = np.empty((m.nbody - 1, 6))
    for b in range(1, m.nbody):
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, velocity[b - 1], 0)
    state = torch.as_tensor(np.r_[d.qpos.copy(), d.qvel.copy()], dtype=torch.float64)
    reference = torch.as_tensor(motion.states[frame], dtype=torch.float64)
    bodies = tuple(
        torch.as_tensor(x.copy(), dtype=torch.float64)
        for x in (d.xpos[1:], d.xquat[1:], velocity[:, 3:], velocity[:, :3])
    )
    targets = tuple(
        torch.as_tensor(x[frame], dtype=torch.float64)
        for x in (motion.body_pos, motion.body_quat, motion.body_vel, motion.body_omega)
    )
    limits = torch.as_tensor(m.jnt_range[[m.joint(n).id for n in JOINT_NAMES]], dtype=torch.float64)
    terms = tracking_terms(
        state,
        reference,
        bodies,
        targets,
        torch.as_tensor(action),
        torch.as_tensor(previous),
        limits,
        env.control_dt,
    )
    height, tilt = termination_errors(state, reference)
    pos = d.xpos[1:]
    relative = pos - pos[:1]
    target_relative = motion.body_pos[frame] - motion.body_pos[frame, :1]
    metrics = dict(
        global_mpjpe=float(np.linalg.norm(pos - motion.body_pos[frame], axis=-1).mean()),
        root_relative_mpjpe=float(np.linalg.norm(relative - target_relative, axis=-1).mean()),
        joint_rmse=float(
            np.sqrt(np.mean((d.qpos[env.qpos_indices] - motion.states[frame, 7:36]) ** 2))
        ),
        root_height=float(d.qpos[2]),
    )
    metrics["reward_terms"] = {name: float(value) for name, value in terms.items()}
    return float(sum(terms.values())), bool((height > 0.25) | (tilt > 0.8)), metrics
