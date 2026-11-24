"""Export official BRT G1 NMR outputs by its executable XML scatter mapping."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--vendor', type=Path, default=Path(__file__).parent / 'vendor/BeyondRetarget')
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target-model', type=Path, default=Path('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml'))
    args = parser.parse_args()
    vendor = args.vendor.resolve()
    sys.path.insert(0, str(vendor))
    from lib.robot.robot_spec import load_robot_spec
    spec = load_robot_spec(str(vendor / 'config/robot/g1.yaml'), str(vendor))
    model = mujoco.MjModel.from_xml_path(spec.xml_path)
    hinge_ids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    native_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge_ids]
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / 'src'))
    from supermimic.motion import JOINT_NAMES
    assert len(hinge_ids) == 29 and set(native_names) == set(JOINT_NAMES)
    mapping = list(spec.model_to_xml_dof)
    assert sorted(mapping) == list(range(29))
    data = np.load(args.prediction, allow_pickle=False)
    source_dof = data['dof'].astype(float)
    xml_dof = np.empty_like(source_dof)
    xml_dof[:, mapping] = source_dof
    q = data['root_rot_quat'].astype(float)
    root = data['root_trans'].astype(float)
    fps = float(data['fps'].reshape(-1)[0])
    assert source_dof.shape == (len(root), 29)
    assert all(np.isfinite(v).all() for v in (source_dof, q, root))
    assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4)
    # Source free body and target both use a pelvis-origin free pose. No root
    # recentering, lifting, joint clipping, or wrist replacement is performed.
    canonical = xml_dof[:, [native_names.index(name) for name in JOINT_NAMES]]
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'hybrik_dance_0000_0400': dict(dof=canonical, root_trans_offset=root,
                root_rot=q[:, [1, 2, 3, 0]], fps=fps)}, args.output / 'motion_source.pkl')
    qpos = np.broadcast_to(model.qpos0, (len(root), model.nq)).copy()
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    assert len(free) == 1
    addr = int(model.jnt_qposadr[free[0]])
    qpos[:, addr:addr + 3] = root
    qpos[:, addr + 3:addr + 7] = q
    qpos[:, model.jnt_qposadr[hinge_ids]] = xml_dof
    np.savez_compressed(args.output / 'source_qpos.npz', qpos=qpos, fps=fps,
                        joint_names=np.array(native_names), network_dof=source_dof)
    target = mujoco.MjModel.from_xml_path(str(args.target_model))
    differences = {}
    for j in hinge_ids:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        tj = target.joint(name).id
        if not np.allclose(model.jnt_axis[j], target.jnt_axis[tj], atol=1e-10):
            raise ValueError(f'Joint axis mismatch: {name}')
        source_body = int(model.jnt_bodyid[j])
        target_body = int(target.jnt_bodyid[tj])
        delta = target.body_pos[target_body] - model.body_pos[source_body]
        if np.max(abs(delta)) > 1e-8:
            differences[name] = delta.tolist()
    metadata = dict(source=str(args.prediction.resolve()), source_sha256=sha(args.prediction),
        source_xml=str(Path(spec.xml_path).resolve()), source_xml_sha256=sha(spec.xml_path),
        mapping=[dict(network_index=i, xml_hinge_index=mapping[i], joint_name=native_names[mapping[i]]) for i in range(29)],
        declared_yaml_order_conflicts_with_executable_mapping=spec.model_dof_order != [native_names[i] for i in mapping],
        mapping_basis='Official model_to_xml_dof executable scatter, not the contradictory YAML model_dof_order labels',
        root_transform='identity xyz; WXYZ to XYZW storage only; no height/pose correction in exporter',
        target_xml_sha256=sha(args.target_model), target_minus_source_joint_body_offset_m=differences,
        model_difference_note='Named hinge axes equal; static geometry offsets differ, no compensation applied',
        source_frames=len(root), fps=fps, sample_span_seconds=(len(root)-1)/fps,
        canonical_joint_names=list(JOINT_NAMES), motion_sha256=sha(args.output/'motion_source.pkl'),
        warning='Official phase-all postprocess itself filters/root-grounds/contact-IK; raw output is separately retained. Kinematic compatibility is not dynamic feasibility.')
    (args.output/'conversion.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), frames=len(root), fps=fps, yaml_label_conflict=metadata['declared_yaml_order_conflicts_with_executable_mapping'])))


if __name__ == '__main__':
    main()
