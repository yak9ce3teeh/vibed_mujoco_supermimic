"""Headless official GMR API runner; preserves every input frame and raw qpos."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prediction', type=Path, required=True)
    p.add_argument('--body-models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key', default='hybrik_dance_0000_0400')
    p.add_argument('--target-model', type=Path, default=Path('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml'))
    a = p.parse_args()
    neutral = a.body_models / 'smplx/SMPLX_NEUTRAL.npz'
    if not neutral.is_file():
        raise SystemExit(f'Missing licensed official model: {neutral}; obtain under its license, no automatic download.')
    from general_motion_retargeting import GeneralMotionRetargeting, ROBOT_XML_DICT
    from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'src'))
    from supermimic.motion import JOINT_NAMES
    model_path = Path(ROBOT_XML_DICT['unitree_g1']).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    ids = [model.joint(n).id for n in JOINT_NAMES]
    assert len(set(ids)) == 29 and all(model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE for i in ids)
    target = mujoco.MjModel.from_xml_path(str(a.target_model))
    offsets = {}
    for name, source_id in zip(JOINT_NAMES, ids):
        target_id = target.joint(name).id
        if not np.allclose(model.jnt_axis[source_id], target.jnt_axis[target_id], atol=1e-10):
            raise ValueError(f'Named joint axis mismatch: {name}')
        delta = target.body_pos[target.jnt_bodyid[target_id]] - model.body_pos[model.jnt_bodyid[source_id]]
        if abs(delta).max() > 1e-8:
            offsets[name] = delta.tolist()
    data, body, body_output, height = load_gvhmr_pred_file(str(a.prediction), a.body_models)
    frames, fps = get_gvhmr_data_offline_fast(data, body, body_output, tgt_fps=30)
    retarget = GeneralMotionRetargeting(actual_human_height=height, src_human='smplx', tgt_robot='unitree_g1')
    qpos = np.array([retarget.retarget(frame).copy() for frame in frames])
    assert qpos.shape == (len(frames), model.nq) and np.isfinite(qpos).all()
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert len(free) == 1
    addr = int(model.jnt_qposadr[free[0]])
    quat = qpos[:, addr+3:addr+7]
    assert np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-4)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output/'gmr_original_qpos.npz', qpos=qpos, fps=fps)
    joblib.dump({a.key: dict(dof=qpos[:, model.jnt_qposadr[ids]],
        root_trans_offset=qpos[:, addr:addr+3], root_rot=quat[:, [1, 2, 3, 0]], fps=float(fps))}, a.output/'motion_source.pkl')
    metadata = dict(source_prediction_sha256=sha(a.prediction), body_model_sha256=sha(neutral),
        source_robot_xml=str(model_path), source_robot_xml_sha256=sha(model_path),
        target_model_xml_sha256=sha(a.target_model), target_minus_source_joint_body_offsets_m=offsets,
        frames=len(frames), fps=float(fps), sample_span_seconds=(len(frames)-1)/fps,
        joint_mapping=[dict(name=n, qpos_address=int(model.jnt_qposadr[i])) for n,i in zip(JOINT_NAMES,ids)],
        transforms=['Official GMR GVHMR helper rotates human frame by Rx(+90deg): xyz -> x,-z,y; left-multiplies orientation.',
                    'Official GMR human scaling, configured local target offsets and IK unchanged; offset_to_ground=False.',
                    'Exporter only reorders named hinge qpos and WXYZ -> XYZW; no new root lift/alignment/clipping.'],
        adapter_note='Headless API equivalent; includes frame0 unlike upstream demo loop which increments before first retarget.',
        motion_sha256=sha(a.output/'motion_source.pkl'))
    (a.output/'conversion.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
