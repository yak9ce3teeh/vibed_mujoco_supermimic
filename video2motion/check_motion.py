"""Read-only native G1 kinematic QC and optional reference video; no simulation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic.motion import JOINT_NAMES, load_motions
from supermimic.mujoco_adapter import MujocoAdapter


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def support_z(model, data, gid):
    """Exact lower support of the primitive, in world Z (not geom centre)."""
    normal = data.geom_xmat[gid].reshape(3, 3)[2]
    size = model.geom_size[gid]
    kind = model.geom_type[gid]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        radius = abs(normal) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = size[0] + size[1] * abs(normal[2])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = size[1] * abs(normal[2]) + size[0] * np.linalg.norm(normal[:2])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius = np.linalg.norm(size * normal)
    else:
        raise ValueError(f'Unsupported physical foot geometry {kind}, cannot claim surface QC')
    return float(data.geom_xpos[gid, 2] - radius)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--key', default='hybrik_dance_0000_0400')
    p.add_argument('--model', type=Path, default=Path('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml'))
    p.add_argument('--controller', type=Path, default=ROOT / 'configs/g1_controller.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--render', action='store_true')
    a = p.parse_args()
    c = json.loads(a.controller.read_text())
    env = MujocoAdapter(a.model, JOINT_NAMES, c['kp'], c['kd'], c['q_default'], c['action_scale'])
    m = load_motions(a.motion, [a.key], env)[a.key]
    states = m.states
    q = states[:, 7:36]
    ids = [env.model.joint(n).id for n in JOINT_NAMES]
    limits = env.model.jnt_range[ids]
    excess = np.maximum(limits[:, 0] - q, q - limits[:, 1]).clip(0)
    feet = []
    for gid in range(env.model.ngeom):
        if not (env.model.geom_contype[gid] or env.model.geom_conaffinity[gid]):
            continue
        body = int(env.model.geom_bodyid[gid])
        while body:
            name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
            if name in ('left_ankle_roll_link', 'right_ankle_roll_link'):
                feet.append(gid)
                break
            body = int(env.model.body_parentid[body])
    assert feet
    heights = []
    for s in states:
        env.data.qpos[:] = s[:36]
        env.data.qvel[:] = s[36:]
        mujoco.mj_forward(env.model, env.data)
        heights.append([support_z(env.model, env.data, gid) for gid in feet])
    heights = np.asarray(heights)
    assert all(np.isfinite(x).all() for x in (states, m.body_pos, m.body_vel, heights))
    report = dict(motion_sha256=sha(a.motion), model_xml_sha256=sha(a.model), controller_sha256=sha(a.controller),
        key=a.key, runtime_frames=len(states), runtime_fps=1/m.dt, runtime_span_seconds=(len(states)-1)*m.dt,
        finite=True, root_z_min_max=states[:, 2].take([np.argmin(states[:, 2]), np.argmax(states[:, 2])]).tolist(),
        quaternion_max_unit_error=float(abs(np.linalg.norm(states[:, 3:7], axis=1)-1).max()),
        joint_names=list(JOINT_NAMES), joint_outside_fraction=(excess > 1e-8).mean(axis=0).tolist(),
        joint_max_excess_rad=excess.max(axis=0).tolist(), joint_velocity_peak_rad_s=abs(states[:, 42:]).max(axis=0).tolist(),
        joint_velocity_rms_rad_s=np.sqrt((states[:, 42:]**2).mean(axis=0)).tolist(),
        joint_step_max_rad=abs(np.diff(q, axis=0)).max(axis=0).tolist(),
        foot_physical_geom_ids=feet, foot_surface_z_min_m=float(heights.min()),
        foot_surface_below_zero_fraction=float((heights < 0).mean()),
        ground_assumption='World Z=0, target model flat plane; exact primitive support of collision geoms',
        corrections='None: no clipping, lifting, root alignment or wrist replacement',
        interpretation='Kinematic format/FK checks only; not physical feasibility or policy tracking')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'qc.json').write_text(json.dumps(report, indent=2)+'\n')
    if a.render:
        env.model.vis.global_.offwidth = 960
        env.model.vis.global_.offheight = 640
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 3.5
        camera.azimuth = 130
        camera.elevation = -15
        env.model.geom_rgba[env.model.geom_bodyid > 0] = [0.65, 0.25, 0.85, 1]
        indices = list(range(0, len(states), 2))
        if indices[-1] != len(states)-1:
            indices.append(len(states)-1)
        with mujoco.Renderer(env.model, 640, 960) as renderer, imageio.get_writer(a.output/'reference.mp4', fps=25, codec='libx264') as writer:
            for i in indices:
                env.data.qpos[:] = states[i, :36]
                mujoco.mj_forward(env.model, env.data)
                camera.lookat[:] = states[i, :3]
                camera.lookat[2] = max(camera.lookat[2], .7)
                renderer.update_scene(env.data, camera=camera)
                writer.append_data(renderer.render())
        report['render'] = dict(frames=len(indices), fps=25, physics_steps=0, camera='display-only follow; poses unchanged')
        (a.output/'qc.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
