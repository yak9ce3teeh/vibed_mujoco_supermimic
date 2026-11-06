"""Convert only the verified 64-file BONES smoke subset to local Motion source.

Follows first-party ProtoMotions/SONIC conventions: cm to m, extrinsic xyz
Euler degrees to XYZW, named joint degrees to radians, all rows at 120Hz.
No root alignment, coordinate swaps, clipping, height offset or fake SMPL data.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic.motion import JOINT_NAMES, load_motions
from supermimic.mujoco_adapter import MujocoAdapter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extrinsic_matrix(angles):
    x,y,z=np.deg2rad(angles)
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    return rz @ ry @ rx


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.project.resolve();data=root/'dataset'
    output=data/'processed/bones_seed_smoke_g32/motion_source'
    output.mkdir(parents=True,exist_ok=False)
    selected_path=data/'manifests/bones_seed/local_package_balanced_seed7_g32.jsonl'
    selected=[json.loads(line) for line in selected_path.read_text().splitlines()]
    audit_path=data/'manifests/bones_seed/archive-validation.json';audit=json.loads(audit_path.read_text())
    parsed={row['path']:row for row in audit['selected_csv']}
    assert audit['passed'] and len(selected)==64 and {r['move_g1_path'] for r in selected}==set(parsed)
    motions={};manifest=[];rootz=[];allq=[];max_rotation_error=0.
    expected=['Frame','root_translateX','root_translateY','root_translateZ','root_rotateX','root_rotateY','root_rotateZ']+[name+'_dof' for name in JOINT_NAMES]
    for item in selected:
        key=item['move_g1_path'];path=data/'processed/bones_seed_smoke_g32/csv'/key
        if sha(path)!=parsed[key]['sha256']:raise ValueError(f'CSV hash mismatch {key}')
        with path.open() as stream:
            reader=csv.reader(stream);header=next(reader)
            if header!=expected:raise ValueError(f'Explicit named order mismatch {key}')
            values=np.array(list(reader),dtype=np.float64)
        if values.shape!=(parsed[key]['rows'],36) or not np.isfinite(values).all():raise ValueError(key)
        frames=values[:,0]
        if not np.array_equal(frames,frames[0]+np.arange(len(frames))):raise ValueError('Non-contiguous Frame array')
        angles=values[:,4:7]
        rotation=Rotation.from_euler('xyz',angles,degrees=True)
        q=rotation.as_quat()
        for i in range(1,len(q)):
            if q[i]@q[i-1]<0:q[i]*=-1
        if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-12):raise ValueError('Quaternion norm')
        # Independently detect intrinsic/extrinsic confusion on actual nonzero rotations.
        candidates=np.flatnonzero(np.sum(abs(angles)>1e-3,axis=1)>=2)
        for i in candidates[::max(1,len(candidates)//3)][:4]:
            error=float(abs(rotation[i].as_matrix()-extrinsic_matrix(angles[i])).max())
            max_rotation_error=max(max_rotation_error,error)
            if error>1e-12:raise ValueError('Extrinsic xyz composition mismatch')
        dof=np.deg2rad(values[:,7:]);xyz=values[:,1:4]/100
        motions[key]={'fps':120.,'dof':dof,'root_trans_offset':xyz,'root_rot':q}
        rootz.append(xyz[:,2]);allq.append(dof)
        manifest.append(dict(clip_id=key,recording_id=item['recording_id'],is_mirror=item['is_mirror'],
            parent_csv=str(path.relative_to(root)),parent_csv_sha256=sha(path),
            archive_artifact_id='bones_seed_g1_archive',archive_sha256=audit['archive_sha256'],
            source_revision='2f59b2077b9da34dd4e43618e705c7cb962c9a66',
            source_frame_start=int(frames[0]),source_frame_end_exclusive=int(frames[-1]+1),source_frame_step=1,
            num_frames=len(frames),fps=120.,dt_seconds=1/120,
            conversion='XYZ /100; Rotation.from_euler(lowercase xyz,degrees=True).as_quat XYZW; named joints deg2rad',
            coordinate_changes='none: no axis swap, first-frame subtraction, root height offset or joint clipping',
            split=item['split'],purpose='conversion smoke only; not used by active training'))
    artifact=output/'bones_seed_g32_motion_source.pkl';joblib.dump(motions,artifact,compress=3)
    config=json.loads((root/'configs/g1_controller.json').read_text())
    # Controller may include provenance; only six numerical fields enter adapter.
    env=MujocoAdapter('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml',JOINT_NAMES,
        config['kp'],config['kd'],config['q_default'],config['action_scale'])
    samplekeys=[next(x['clip_id'] for x in manifest if not x['is_mirror']),next(x['clip_id'] for x in manifest if x['is_mirror'])]
    adapted=load_motions(artifact,samplekeys,env)
    checks={name:dict(frames50hz=len(m.states),finite=bool(all(np.isfinite(a).all() for a in [m.states,m.body_pos,m.body_quat,m.body_vel,m.body_omega])),dt=m.dt) for name,m in adapted.items()}
    if not all(x['finite'] and x['dt']==.02 for x in checks.values()):raise ValueError('Runtime resampling/FK nonfinite')
    joint_limits=env.model.jnt_range[env.joint_ids] if hasattr(env,'joint_ids') else np.array([env.model.jnt_range[__import__('mujoco').mj_name2id(env.model,__import__('mujoco').mjtObj.mjOBJ_JOINT,name)] for name in JOINT_NAMES])
    joint=np.concatenate(allq);outside=(joint<joint_limits[:,0])|(joint>joint_limits[:,1])
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),converted=64,
        artifact=str(artifact.relative_to(root)),artifact_sha256=sha(artifact),
        selected_manifest_sha256=sha(selected_path),archive_validation_sha256=sha(audit_path),
        converter_sha256=sha(Path(__file__)),source_frames_preserved=True,fps=120,dt_seconds=1/120,
        max_extrinsic_RzRyRx_matrix_error=max_rotation_error,
        root_z_quantiles_m=dict(zip(['min','p01','p50','p99','max'],map(float,np.quantile(np.concatenate(rootz),[0,.01,.5,.99,1])))),
        xml_joint_outside_fraction=float(outside.mean()),xml_joint_outside_fraction_per_joint=dict(zip(JOINT_NAMES,map(float,outside.mean(axis=0)))),
        diagnostics_note='No clipping, correction or first-frame normalization; no active training input changed.',
        runtime50hz_FK_checks=checks,
        root_coordinate_axes='preserved source XYZ; global axis specification not independently established',
        evidence='dataset/manifests/bones_seed/schema-evidence-v3.json')
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (output/'conversion-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
