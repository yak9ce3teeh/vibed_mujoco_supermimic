"""Load named G1 LAFAN clips. Source rotations XYZW; runtime rotations WXYZ."""

from dataclasses import dataclass

import joblib
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# The single supported floating-base G1 state contract.
JOINT_DIM, QPOS_DIM, QVEL_DIM, STATE_DIM = 29, 36, 35, 71
OBS_DIM, GOAL_DIM = 93, 73
HISTORY, FUTURE, HORIZON, WORLD_WINDOW = 10, 10, 32, 8
ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
JOINT_POS = slice(7, 36)
ROOT_LINVEL = slice(36, 39)
ROOT_ANGVEL = slice(39, 42)
JOINT_VEL = slice(42, 71)

# Dataset contract: humanoidverse/config/robot/g1/g1_29dof.yaml dof_names.
JOINT_NAMES = tuple(
    n + "_joint"
    for n in (
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",
        "left_wrist_pitch",
        "left_wrist_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    )
)


@dataclass
class Motion:
    name: str
    states: np.ndarray  # [T,71]: full MuJoCo qpos36, qvel35
    body_pos: np.ndarray
    body_quat: np.ndarray
    body_vel: np.ndarray
    body_omega: np.ndarray
    dt: float


def load_motions(path, names, env):
    source = joblib.load(path)  # Only load trusted local motion libraries.
    result = {}
    if source and "states" in next(iter(source.values())):
        manifest=json.loads((Path(path).parent/'manifest.json').read_text())
        buffer=np.empty(mujoco.mj_sizeModel(env.model),dtype=np.uint8)
        mujoco.mj_saveModel(env.model,buffer=buffer)
        if manifest.get('compiled_model_sha256')!=hashlib.sha256(buffer.tobytes()).hexdigest():
            raise ValueError('Canonical FK cache compiled-model identity mismatch')
    for name in names:
        clip = source[name]
        if "states" in clip:
            arrays=[np.asarray(clip[key]) for key in ("states","body_pos","body_quat","body_vel","body_omega")]
            if arrays[0].ndim!=2 or arrays[0].shape[1]!=71 or not all(np.isfinite(a).all() for a in arrays):
                raise ValueError("Invalid canonical full-recording cache")
            if float(clip["dt"])!=env.control_dt:raise ValueError("Canonical cache/control clock mismatch")
            result[name]=Motion(name,*arrays,float(clip["dt"]))
            continue
        dof = np.asarray(clip["dof"], dtype=float)
        xyz = np.asarray(clip["root_trans_offset"], dtype=float)
        xyzw = np.asarray(clip["root_rot"], dtype=float)
        if dof.ndim != 2 or dof.shape[1] != 29 or xyz.shape != (len(dof), 3):
            raise ValueError("invalid G1 motion shapes")
        if not all(np.isfinite(a).all() for a in (dof, xyz, xyzw)):
            raise ValueError("motion contains nonfinite data")
        if not np.allclose(np.linalg.norm(xyzw, axis=1), 1, atol=1e-4):
            raise ValueError("source quaternions must be unit XYZW")
        old_t = np.arange(len(dof)) / float(clip["fps"])
        times = np.minimum(np.arange(0, old_t[-1] + 1e-9, env.control_dt), old_t[-1])
        q = np.broadcast_to(env.model.qpos0, (len(times), env.model.nq)).copy()
        q[:, :3] = np.stack([np.interp(times, old_t, xyz[:, j]) for j in range(3)], -1)
        rotation = Slerp(old_t, Rotation.from_quat(xyzw))(times).as_quat()
        q[:, 3:7] = rotation[:, [3, 0, 1, 2]]
        for i in range(1, len(q)):
            if q[i, 3:7] @ q[i - 1, 3:7] < 0:
                q[i, 3:7] *= -1
        q[:, env.qpos_indices] = np.stack(
            [np.interp(times, old_t, dof[:, j]) for j in range(29)], -1
        )
        # mj_differentiatePos respects freejoint local angular velocity semantics.
        v = np.empty((len(q), env.model.nv))
        for i in range(len(q) - 1):
            mujoco.mj_differentiatePos(env.model, v[i], env.control_dt, q[i], q[i + 1])
        v[-1] = v[-2]
        data = mujoco.MjData(env.model)
        positions, quats, velocities, omegas = [], [], [], []
        for qi, vi in zip(q, v):
            data.qpos[:] = qi
            data.qvel[:] = vi
            mujoco.mj_forward(env.model, data)
            spatial = np.empty((env.model.nbody - 1, 6))
            for b in range(1, env.model.nbody):
                mujoco.mj_objectVelocity(
                    env.model, data, mujoco.mjtObj.mjOBJ_XBODY, b, spatial[b - 1], 0
                )
            positions.append(data.xpos[1:].copy())
            quats.append(data.xquat[1:].copy())
            velocities.append(spatial[:, 3:].copy())
            omegas.append(spatial[:, :3].copy())
        result[name] = Motion(
            name,
            np.concatenate((q, v), -1),
            np.array(positions),
            np.array(quats),
            np.array(velocities),
            np.array(omegas),
            env.control_dt,
        )
    return result
