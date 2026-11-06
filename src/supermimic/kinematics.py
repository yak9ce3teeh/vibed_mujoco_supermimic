"""Differentiable rigid-body FK from the compiled MuJoCo joint contract."""

import mujoco
import numpy as np
import torch
from torch import nn

from .dynamics import rotation_exp
from .task import product, rotate


class Kinematics(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.parents = tuple(int(i) for i in model.body_parentid)
        self.joints = []
        for b in range(model.nbody):
            ids = list(
                range(int(model.body_jntadr[b]), int(model.body_jntadr[b] + model.body_jntnum[b]))
            )
            self.joints.append(
                [
                    (int(model.jnt_type[j]), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), j)
                    for j in ids
                ]
            )
        for name, values in [
            ("offset", model.body_pos),
            ("orientation", model.body_quat),
            ("anchor", model.jnt_pos),
            ("axis", model.jnt_axis),
            ("qpos0", model.qpos0),
        ]:
            self.register_buffer(
                name, torch.as_tensor(np.asarray(values).copy(), dtype=torch.float64)
            )
        roots = [
            b
            for b in range(1, model.nbody)
            if any(j[0] == mujoco.mjtJoint.mjJNT_FREE for j in self.joints[b])
        ]
        if roots != [1] or self.parents[1] != 0:
            raise ValueError("FK supports one root freejoint followed by hinge/fixed bodies")
        if any(
            j[0] not in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE)
            for row in self.joints
            for j in row
        ):
            raise ValueError("FK supports free/hinge/fixed bodies only")

    def forward(self, state):
        shape = state.shape[:-1]
        zero = torch.zeros(shape + (3,), device=state.device, dtype=state.dtype)
        identity = torch.zeros(shape + (4,), device=state.device, dtype=state.dtype)
        identity[..., 0] = 1
        positions, quaternions, velocities, omegas = [zero], [identity], [zero], [zero]
        for b in range(1, len(self.parents)):
            # Fixed body transforms precede hinge rotations in the body frame.
            parent = self.parents[b]
            pq, pw, pv, pp = (
                quaternions[parent],
                omegas[parent],
                velocities[parent],
                positions[parent],
            )
            offset = rotate(pq, self.offset[b].to(state).expand_as(zero))
            pos = pp + offset
            quat = product(pq, self.orientation[b].to(state).expand_as(identity))
            velocity = pv + torch.cross(pw, offset, dim=-1)
            omega = pw
            for kind, qp, qv, j in self.joints[b]:
                if kind == mujoco.mjtJoint.mjJNT_FREE:
                    pos = state[..., qp : qp + 3]
                    quat = state[..., qp + 3 : qp + 7]
                    velocity = state[..., 36 + qv : 36 + qv + 3]
                    omega = rotate(quat, state[..., 36 + qv + 3 : 36 + qv + 6])
                else:
                    anchor = rotate(quat, self.anchor[j].to(state).expand_as(zero))
                    # Preserve the moving joint pivot when rotating about a nonzero anchor.
                    pivot = pos + anchor
                    pivot_velocity = velocity + torch.cross(omega, anchor, dim=-1)
                    axis = self.axis[j].to(state).expand_as(zero)
                    world_axis = rotate(quat, axis)
                    quat = product(
                        quat,
                        rotation_exp(axis * (state[..., qp : qp + 1] - self.qpos0[qp].to(state))),
                    )
                    omega = omega + world_axis * state[..., 36 + qv : 36 + qv + 1]
                    new_anchor = rotate(quat, self.anchor[j].to(state).expand_as(zero))
                    pos = pivot - new_anchor
                    velocity = pivot_velocity - torch.cross(omega, new_anchor, dim=-1)
            positions.append(pos)
            quaternions.append(quat)
            velocities.append(velocity)
            omegas.append(omega)
        return tuple(
            torch.stack(x[1:], dim=-2) for x in (positions, quaternions, velocities, omegas)
        )
