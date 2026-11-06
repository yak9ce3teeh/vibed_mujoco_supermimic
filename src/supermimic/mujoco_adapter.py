"""Native CPU MuJoCo transfer boundary. All angles are radians, quaternions WXYZ."""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np


class MujocoAdapter:
    """Hold one joint target for four physics steps; recompute PD at every step.

    Controller vectors follow ``joint_names``. Full qpos/qvel follow MuJoCo's
    compiled model order. Root linear velocity is world-frame; the freejoint
    angular qvel is body-frame. No free-base actuator is ever commanded.
    """

    def __init__(
        self,
        model_path,
        joint_names,
        kp,
        kd,
        q_default,
        action_scale,
        *,
        physics_dt=0.005,
        decimation=4,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 29 or len(set(self.joint_names)) != 29:
            raise ValueError("joint_names must contain 29 unique joint names")
        if not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        if not isinstance(decimation, int) or decimation < 1:
            raise ValueError("decimation must be a positive integer")
        self.model.opt.timestep = physics_dt
        self.decimation = decimation
        self.control_dt = physics_dt * decimation
        self.kp = self._vector(kp, "kp")
        self.kd = self._vector(kd, "kd")
        self.q_default = self._vector(q_default, "q_default")
        self.action_scale = self._vector(action_scale, "action_scale")
        if (self.kp < 0).any() or (self.kd < 0).any():
            raise ValueError("PD gains must be nonnegative")
        joints = [self.model.joint(name).id for name in self.joint_names]
        if any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("controlled joints must be hinge joints")
        self.qpos_indices = self.model.jnt_qposadr[joints].copy()
        self.qvel_indices = self.model.jnt_dofadr[joints].copy()
        actuators = []
        for joint in joints:
            ids = np.flatnonzero(
                (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (self.model.actuator_trnid[:, 0] == joint)
            )
            if len(ids) != 1:
                raise ValueError("each controlled joint requires exactly one motor")
            a = int(ids[0])
            if (
                not np.array_equal(self.model.actuator_gear[a], [1, 0, 0, 0, 0, 0])
                or self.model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED
                or self.model.actuator_gainprm[a, 0] != 1
                or self.model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_NONE
                or not self.model.actuator_ctrllimited[a]
            ):
                raise ValueError("requires direct unit-gear torque motors with ctrlrange")
            actuators.append(a)
        self.actuator_indices = np.asarray(actuators)
        self.torque_range = self.model.actuator_ctrlrange[self.actuator_indices].copy()
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("requires one floating base")
        self.root_qpos = int(self.model.jnt_qposadr[free[0]])
        self.root_qvel = int(self.model.jnt_dofadr[free[0]])
        self.root_body = int(self.model.jnt_bodyid[free[0]])
        self.reset()

    @staticmethod
    def _vector(value, name):
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), (29,)).copy()
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite")
        return result

    def reset(self, qpos=None, qvel=None, context=None):
        q = self.model.qpos0.copy() if qpos is None else np.asarray(qpos, dtype=float).copy()
        v = np.zeros(self.model.nv) if qvel is None else np.asarray(qvel, dtype=float).copy()
        if q.shape != (self.model.nq,) or v.shape != (self.model.nv,):
            raise ValueError("reset requires full model qpos/qvel arrays")
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("reset state must be finite")
        quat = q[self.root_qpos + 3 : self.root_qpos + 7]
        if not np.isclose(np.linalg.norm(quat), 1, atol=1e-6):
            raise ValueError("root quaternion must be unit WXYZ")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        self.context = copy.deepcopy(context)
        self.steps = 0
        self.last_action = np.zeros(29)
        self.last_torque = np.zeros(29)
        self.torque_trace = np.zeros((self.decimation, 29))
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        spatial = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.root_body, spatial, 0
        )
        p, v = self.root_qpos, self.root_qvel
        return dict(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            joint_pos=self.data.qpos[self.qpos_indices].copy(),
            joint_vel=self.data.qvel[self.qvel_indices].copy(),
            root_pos_world=self.data.qpos[p : p + 3].copy(),
            root_quat_wxyz=self.data.qpos[p + 3 : p + 7].copy(),
            root_linvel_world=spatial[3:].copy(),
            root_angvel_world=spatial[:3].copy(),
            root_angvel_body=self.data.qvel[v + 3 : v + 6].copy(),
            time=float(self.data.time),
            step=self.steps,
        )

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("action must be finite shape (29,)")
        target = self.q_default + self.action_scale * action
        self.last_action = action.copy()
        for i in range(self.decimation):
            torque = self.kp * (target - self.data.qpos[self.qpos_indices])
            torque -= self.kd * self.data.qvel[self.qvel_indices]
            torque = np.clip(torque, self.torque_range[:, 0], self.torque_range[:, 1])
            self.data.ctrl[:] = 0
            self.data.ctrl[self.actuator_indices] = torque
            self.torque_trace[i] = torque
            mujoco.mj_step(self.model, self.data)
        self.last_torque = torque.copy()
        self.steps += 1
        # mj_step leaves derived positions at the previous integration point.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo produced nonfinite state")
        return self.state()
