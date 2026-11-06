"""Prepare all forty public G1 recordings, without temporal cropping or root offsets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import mujoco
import numpy as np
from supermimic.motion import JOINT_NAMES, load_motions
from supermimic.mujoco_adapter import MujocoAdapter


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('dataset/retargeted/lafan1_public_g1'))
    p.add_argument('--output',type=Path,default=Path('dataset/processed/lafan1_full_g1'))
    p.add_argument('--model',default='/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml')
    p.add_argument('--controller',type=Path,default=Path('configs/g1_controller.json'))
    p.add_argument('--legacy',type=Path,default=Path('dataset/raw/lafan_29dof.pkl'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists():raise FileExistsError('Canonical manifest already exists; do not overwrite provenance')
    files=sorted(args.source.glob('*.csv'))
    if len(files)!=40:raise ValueError(f'Expected forty complete recordings, found {len(files)}')
    legacy=joblib.load(args.legacy);source={};rows=[]
    for path in files:
        a=np.loadtxt(path,delimiter=',',dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=36 or not np.isfinite(a).all():raise ValueError(f'Invalid public G1 CSV: {path}')
        if not np.allclose(np.linalg.norm(a[:,3:7],axis=-1),1,atol=1e-5):raise ValueError('Nonunit XYZW source quaternion')
        source[path.stem]=dict(fps=30,dof=a[:,7:].astype(np.float32),root_trans_offset=a[:,:3].astype(np.float32),root_rot=a[:,3:7].astype(np.float32))
        old=legacy[path.stem]
        errors={key:float(np.max(np.abs(np.asarray(old[key])-values))) for key,values in [('dof',a[:,7:]),('root_trans_offset',a[:,:3]),('root_rot',a[:,3:7])]}
        if max(errors.values())>1e-5:raise ValueError(f'Public/legacy consistency failed {path.stem}: {errors}')
        rows.append(dict(recording_id=path.stem,source=str(path.resolve()),source_sha256=digest(path),source_rows=len(a),source_fps=30,legacy_max_absolute_difference=errors))
    if sum(r['source_rows'] for r in rows)!=264705:raise ValueError('Unexpected complete source row count')
    raw=args.output/'source-30hz.joblib';joblib.dump(source,raw,compress=0)
    controller=json.loads(args.controller.read_text())
    env=MujocoAdapter(args.model,JOINT_NAMES,controller['kp'],controller['kd'],controller['q_default'],controller['action_scale'])
    motions=load_motions(raw,list(source),env)
    canonical={}
    for row in rows:
        m=motions[row['recording_id']]
        fields={key:np.asarray(getattr(m,key),dtype=np.float32) for key in ['states','body_pos','body_quat','body_vel','body_omega']}
        if not all(np.isfinite(a).all() for a in fields.values()):raise ValueError('Nonfinite canonical FK or velocities')
        if not np.allclose(np.linalg.norm(fields['states'][:,3:7],axis=-1),1,atol=1e-6):raise ValueError('Nonunit runtime quaternion')
        canonical[m.name]=dict(**fields,dt=.02)
        row.update(runtime_frames=len(m.states),runtime_transitions=len(m.states)-1,runtime_fps=50,first_time_seconds=0,last_time_seconds=(len(m.states)-1)*.02)
    if sum(r['runtime_frames'] for r in rows)!=441140:raise ValueError('Unexpected 50Hz frame count')
    output=args.output/'motions-50hz.joblib';joblib.dump(canonical,output,compress=0)
    buffer=np.empty(mujoco.mj_sizeModel(env.model),dtype=np.uint8);mujoco.mj_saveModel(env.model,buffer=buffer)
    manifest=dict(compiled_model_sha256=hashlib.sha256(buffer.tobytes()).hexdigest(),source_url='https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/07ff8845a94d59d4195a8ac04fe918d4e6338239',license_evidence='dataset/manifests/lafan1/acquisition.json; third-party G1 derivative; do not infer redistribution rights from original LAFAN1 license',schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),dataset='LAFAN1 public G1 all40 full recordings',source_revision='07ff8845a94d59d4195a8ac04fe918d4e6338239',source_rows=264705,runtime_frames=441140,runtime_transitions=441100,recordings=rows,joint_names=list(JOINT_NAMES),source_quaternion='XYZW',runtime_quaternion='WXYZ',state_layout='world xyz/WXYZ/q29/world linear velocity/body angular velocity/qd29',units='metres/radians/seconds local simulator interpretation, consistent with legacy retargeted library; not an assertion of undisclosed author retargeting',resampling='linear xyz/joints, quaternion SLERP/sign continuity, native mj_differentiatePos; last velocity repeats preceding derivative',body_features='native MuJoCo FK all40 nonworld bodies',model_xml_sha256=digest(args.model),controller_sha256=digest(args.controller),legacy_sha256=digest(args.legacy),canonical_path=str(output.resolve()),canonical_sha256=digest(output),source_container_sha256=digest(raw),no_temporal_cropping=True,no_root_translation_or_rotation_offsets=True,split='all40 training and in-distribution evaluation; no heldout recordings')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:manifest[k] for k in ['source_rows','runtime_frames','runtime_transitions','canonical_path','canonical_sha256']}),flush=True)


if __name__=='__main__':main()
