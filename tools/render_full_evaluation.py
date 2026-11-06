"""Complete saved evaluations, with explicitly reanchored H32 world predictions.

No simulation rollout: actual states/actions come only from the saved bundle.
Only visual camera/material settings change. Existing short diagnostic videos stay intact.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
from supermimic.models import WorldModel, architecture
from supermimic.motion import JOINT_NAMES, load_motions
from supermimic.mujoco_adapter import MujocoAdapter
from supermimic.task import termination_errors
from supermimic.render import append_robot_ghost, digest, recorded_states, prediction_errors
from supermimic.run_layout import checkpoint_path, content_digest


HORIZON=32


def render_indices(transitions):
    if transitions<2:raise ValueError('Complete25fps replay needs at least two recorded transitions')
    indices=np.arange(0,transitions+1,2,dtype=np.int64)
    if indices[-1]!=transitions:indices[-1]=transitions
    return indices


@torch.inference_mode()
def rolling_prediction(world,actual,actions):
    n=len(actions);prediction=np.full(actual.shape,np.nan,dtype=np.float64)
    anchor_steps=np.minimum(np.arange(n+1)//HORIZON,(n-1)//HORIZON)*HORIZON
    endpoints=[];end_steps=[];invalid=[]
    for anchor in range(0,n,HORIZON):
        state=torch.as_tensor(actual[anchor],dtype=torch.float32)[None]
        prediction[anchor]=actual[anchor]
        end=min(anchor+HORIZON,n)
        for step in range(anchor,end):
            state=world(state,torch.as_tensor(actions[step],dtype=torch.float32)[None])
            if not bool(torch.isfinite(state).all()):
                invalid.append(dict(anchor=anchor,first_nonfinite_step=step+1))
                break
            prediction[step+1]=state[0].numpy()
        endpoints.append(prediction[end].copy());end_steps.append(end)
    return prediction,anchor_steps,np.asarray(end_steps),np.asarray(endpoints),invalid


def termination(record,actual,reference,row):
    h,g=termination_errors(torch.as_tensor(actual[-1],dtype=torch.float64),torch.as_tensor(reference[-1],dtype=torch.float64))
    h,g=float(h),float(g);hard=h>.25 or g>.8
    if hard!=bool(record['dones'][-1]):raise ValueError('Saved termination and original height/gravity-z rule disagree')
    if bool(row['failure'])!=hard:raise ValueError('Evaluation report failure differs from saved transition')
    if not hard and (not bool(record['truncated'][-1]) or len(record['actions'])!=int(record['remaining_transitions'])):
        raise ValueError('Nonfailure must reach the genuine reference end')
    return dict(cause='tracking_termination' if hard else 'reference_end',source_cause=row['termination_cause'],height_error_m=h,height_threshold_m=.25,height_trigger=h>.25,gravity_z_error=g,gravity_z_threshold=.8,gravity_z_trigger=g>.8)


def append_full_ghost(model,data,scene,color):
    """Keep legacy checks; large-coordinate rounding must match native visuals exactly."""
    first=scene.ngeom
    try:
        return append_robot_ghost(model,data,scene,color)
    except ValueError as error:
        if not str(error).startswith('ghost changed native geometry:'):raise
        # mjvGeom is float32 even when mjData is float64. At63m a correct
        # visual transform can differ by >1e-6m. Compare independent native
        # visual fields bit-for-bit; preserve the physical rounding error below.
        options=mujoco.MjvOption();options.sitegroup[:]=0
        native=mujoco.MjvScene(model,maxgeom=1000)
        mujoco.mjv_updateScene(model,data,options,mujoco.MjvPerturb(),mujoco.MjvCamera(),mujoco.mjtCatBit.mjCAT_DYNAMIC,native)
        expected={g.objid:g for g in native.geoms[:native.ngeom] if g.objtype==mujoco.mjtObj.mjOBJ_GEOM}
        ghosts=scene.geoms[first:scene.ngeom]
        if {g.objid for g in ghosts}!=set(expected):raise ValueError('Ghost/native visual geometry set differs')
        errors=dict(position_m=0.,rotation_matrix=0.,geometry_size=0.)
        for geom in ghosts:
            reference=expected[geom.objid]
            if (geom.type,geom.dataid)!=(reference.type,reference.dataid):raise ValueError('Ghost/native visual type differs')
            for field in ('pos','mat','size'):
                a,b=np.asarray(getattr(geom,field)),np.asarray(getattr(reference,field))
                if not np.isfinite(b).all() or not np.array_equal(a,b):raise ValueError(f'Ghost/native visual {field} differs')
            errors['position_m']=max(errors['position_m'],float(np.max(abs(geom.pos-data.geom_xpos[geom.objid]))))
            errors['rotation_matrix']=max(errors['rotation_matrix'],float(np.max(abs(geom.mat.reshape(9)-data.geom_xmat[geom.objid]))))
        return dict(geometry_ids=[int(g.objid) for g in ghosts],max_native_transform_error=errors,float32_rounding_verified_exact_native_visual=True)


def render_movie(model,actual,predicted,reference,indices,anchors,term,row,path):
    data,pred_data,ref_data=(mujoco.MjData(model) for _ in range(3))
    camera=mujoco.MjvCamera();camera.azimuth=135.;camera.elevation=-18.
    model.vis.global_.offwidth=960;model.vis.global_.offheight=512
    renderer=mujoco.Renderer(model,height=512,width=960)
    options=mujoco.MjvOption();options.sitegroup[:]=0
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title=ImageFont.truetype(font,19);small=ImageFont.truetype(font,15)
    colors=dict(actual=[.94,.96,1.,1.],world_model=[1.,.38,.05,.45],reference=[.65,.25,1.,.30])
    errors=prediction_errors(predicted,actual)
    audit=dict(frames_checked=0,ghost_frames_checked=0,maximum_errors=dict(position_m=0.,rotation_matrix=0.,geometry_size=0.),single_floor_all_frames=True,float32_rounding_verified_frames=0)
    camera_range=[float('inf'),0.];valid=np.isfinite(predicted).all(-1)
    temporary=path.with_name(path.stem+'.partial.mp4')
    if temporary.exists():temporary.unlink()
    began=time.monotonic()
    try:
        with imageio.get_writer(str(temporary),fps=25,codec='libx264',quality=8,pixelformat='yuv420p',macro_block_size=16) as writer:
            for step in indices:
                for state,target in ((actual[step],data),(reference[step],ref_data)):
                    target.qpos[:]=state[:36];target.qvel[:]=state[36:];mujoco.mj_forward(model,target)
                if valid[step]:
                    pred_data.qpos[:]=predicted[step,:36];pred_data.qvel[:]=predicted[step,36:];mujoco.mj_forward(model,pred_data)
                center=(actual[step,:3]+reference[step,:3])/2;center[2]=max(.4,center[2])
                bodies=[data.xpos[1:],ref_data.xpos[1:]]+([pred_data.xpos[1:]] if valid[step] else [])
                radius=float(np.linalg.norm(np.concatenate(bodies)-center,axis=-1).max())+.25
                camera.lookat[:]=center;camera.distance=max(3.2,radius/math.sin(math.radians(model.vis.global_.fovy/2))+.3)
                camera_range[0]=min(camera_range[0],camera.distance);camera_range[1]=max(camera_range[1],camera.distance)
                renderer.update_scene(data,camera=camera,scene_option=options)
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid]!=0:
                        geom.rgba[:]=colors['actual'];geom.matid=-1;geom.texid=-1
                audits=[]
                if valid[step]:audits.append(append_full_ghost(model,pred_data,renderer.scene,colors['world_model']))
                audits.append(append_full_ghost(model,ref_data,renderer.scene,colors['reference']))
                floor_count=sum(g.objtype==mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[g.objid]==0 for g in renderer.scene.geoms[:renderer.scene.ngeom])
                if floor_count!=1:raise ValueError('Repeated/missing floor')
                for item in audits:
                    audit['float32_rounding_verified_frames']+=int(item.get('float32_rounding_verified_exact_native_visual',False))
                    for key,value in item['max_native_transform_error'].items():audit['maximum_errors'][key]=max(audit['maximum_errors'][key],value)
                audit['frames_checked']+=1;audit['ghost_frames_checked']+=len(audits)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle((0,0,960,82),fill=(19,25,34));draw.rectangle((0,439,960,512),fill=(19,25,34))
                draw.text((12,5),f"{row['clip']} | final{row['joint_block']} | complete saved evaluation | 1x",font=title,fill='white')
                for x,label,color in [(12,'Actual: white','white'),(245,'WM prediction: orange',(255,155,80)),(570,'Reference goal: purple',(195,140,255))]:draw.text((x,32),label,font=small,fill=color)
                offset=int(step-anchors[step]);anchor_label='REANCHOR' if offset==0 else f'h={offset}/32'
                draw.text((12,58),f'H32 rolling / reanchor every .64s | anchor={anchors[step]} ({anchors[step]*.02:.2f}s) | {anchor_label}',font=small,fill=(210,218,229))
                wm=f"WM-actual root {errors['root_position_cm'][step]:.1f}cm / q {errors['joint_rmse_rad'][step]:.3f}rad" if valid[step] else 'WM NONFINITE: ghost unavailable until next true anchor'
                draw.text((12,444),f"t={step*.02:.2f}s | step {step}/{len(actual)-1} | reference end {int(row['remaining_transitions'])*.02:.2f}s | {wm}",font=small,fill='white')
                draw.text((12,466),f"Full-reference return {row['normalized_return']:.4f} | report MPJPE {row['global_mpjpe']:.3f}m | shared world, no pose offsets",font=small,fill=(202,210,222))
                endpoint=f"{term['cause'].upper()} | height {term['height_error_m']:.3f}m / .25m | gravity-z {term['gravity_z_error']:.3f} / .8" if step==len(actual)-1 else 'Recorded executed actions; no new physics. Camera follows actual/reference midpoint.'
                draw.text((12,489),endpoint,font=small,fill=(255,160,135) if step==len(actual)-1 and term['cause']=='tracking_termination' else (185,218,192))
                writer.append_data(np.asarray(image))
    finally:renderer.close()
    frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if frames!=len(indices):raise ValueError('Complete video decode frame count differs')
    temporary.replace(path)
    return dict(render_seconds=time.monotonic()-began,rendered_frames=frames,video_fps=25,video_duration_seconds=seconds,physical_duration_seconds=(len(actual)-1)*.02,quantization_error_seconds=seconds-(len(actual)-1)*.02,final_hold_seconds=0,playback_speed=1.,camera=dict(policy='actual/reference midpoint; radius includes all three unmodified native body positions',azimuth=135,elevation=-18,distance_range=camera_range),geometry_verification=audit,colors_rgba=colors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--clip',action='append');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads);run=args.run.resolve();snapshot=run/'evals/final'
    report=json.loads((snapshot/'report.json').read_text());rows=[r for r in report['evaluation'] if r.get('evaluation_role')=='all40_primary_start0']
    if len(rows)!=40 or len({r['clip'] for r in rows})!=40:raise ValueError('Requires forty distinct primary evaluations')
    if args.clip:
        unknown=set(args.clip)-{r['clip'] for r in rows}
        if unknown:raise ValueError(f'Unknown recordings: {unknown}')
        rows=[r for r in rows if r['clip'] in args.clip]
    cp=checkpoint_path(snapshot);cp_hash=digest(cp)
    if cp_hash!=report['checkpoint_sha256']:raise ValueError('Final checkpoint differs from final evaluation provenance')
    source_ids=dict(checkpoint_sha256=cp_hash,evaluation_report_sha256=digest(snapshot/'report.json'),trajectory_bundle_sha256=digest(snapshot/'trajectories.npz'))
    pending=[]
    for row in rows:
        folder=run/'visuals/final'/row['clip']/'start0000';manifest=folder/'manifest.json'
        if manifest.exists():
            old=json.loads(manifest.read_text())
            if old.get('status')=='complete' and all(old.get(key)==value for key,value in source_ids.items()) and digest(folder/'actual-wm-reference.mp4')==old['video_sha256'] and digest(folder/'prediction.npz')==old['prediction_npz_sha256']:
                print(json.dumps(dict(status='cached_complete',clip=row['clip'])),flush=True);continue
        pending.append((row,folder))
    if not pending:return
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if available<cp.stat().st_size*3:raise MemoryError('Insufficient host headroom to load trusted full checkpoint once and release replay')
    began=time.monotonic();saved=torch.load(cp,map_location='cpu',weights_only=False)
    if architecture(saved['config'])!=architecture(report['config']):raise ValueError('Checkpoint/report architecture mismatch')
    world=WorldModel(hidden=architecture(saved['config'])['world']).eval();world.load_state_dict(saved['world']);del saved;gc.collect()
    load_seconds=time.monotonic()-began
    c=report['controller'];env=MujocoAdapter(report['config']['model'],JOINT_NAMES,c['kp'],c['kd'],c['q_default'],c['action_scale'])
    manifest_path=Path(report['config']['motions']).with_name('manifest.json');data_manifest=json.loads(manifest_path.read_text())
    if digest(manifest_path)!=report['data_manifest_sha256'] or digest(Path(report['config']['motions']))!=data_manifest['canonical_sha256']:raise ValueError('Canonical reference identity differs')
    motions=load_motions(report['config']['motions'],[r['clip'] for r,_ in pending],env)
    references={name:m.states for name,m in motions.items()};del motions;gc.collect()
    for row,folder in pending:
        row=dict(row,joint_block=report['joint_block'])
        folder.mkdir(parents=True,exist_ok=True);source=snapshot/(row['trajectory_key']+'.npz');actual,record=recorded_states(source)
        if content_digest(record)!=row['trajectory_source_sha256']:raise ValueError('Packed trajectory content differs from report')
        n=len(record['actions']);frames=record['frames']
        if n!=row['steps'] or not np.array_equal(frames,np.arange(n)) or not np.isfinite(actual).all():raise ValueError('Incomplete/discontinuous actual start0 rollout')
        reference=references[row['clip']][np.arange(n+1)].copy();indices=render_indices(n);term=termination(record,actual,reference,row)
        start=time.monotonic();predicted,anchors,end_steps,end_predictions,invalid=rolling_prediction(world,actual,record['actions']);inference=time.monotonic()-start
        prediction_path=folder/'prediction.npz';temporary=prediction_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:np.savez_compressed(f,actual_states=actual,reference_states=reference,predicted_states=predicted,executed_actions=record['actions'],state_reference_frames=np.arange(n+1),anchor_steps=anchors,model_offsets=np.arange(n+1)-anchors,render_indices=indices,window_end_steps=end_steps,window_end_predictions=end_predictions,prediction_valid=np.isfinite(predicted).all(-1))
        temporary.replace(prediction_path)
        metadata=dict(status='rendering',clip=row['clip'],stage='final',joint_block=report['joint_block'],**source_ids,checkpoint=str(cp),trajectory_bundle=str(snapshot/'trajectories.npz'),trajectory_key=row['trajectory_key'],trajectory_source_sha256=row['trajectory_source_sha256'],canonical_reference_sha256=data_manifest['canonical_sha256'],compiled_model_sha256=data_manifest['compiled_model_sha256'],model_xml_sha256=digest(Path(report['config']['model'])),source_physical_fps=50,source_physical_dt=.02,source_transitions=n,source_states=n+1,reference_remaining_transitions=row['remaining_transitions'],inference_seconds=inference,checkpoint_load_seconds=load_seconds,wm=dict(mode='H32 rolling',horizon=32,reanchor_period_seconds=.64,anchor_input='actual saved state at every32 controls; only saved actions within each window',anchor_display='true initialization displayed at anchors; pre-reanchor H32 endpoints retained separately',nonfinite_windows=invalid),sampling='Every second50Hz source frame; for oddN replace last regular sample by actual endpoint to keep25fps duration error<=one frame',termination=term,world_alignment='No recentering, translation/rotation alignment or physical stepping',prediction_npz_sha256=digest(prediction_path))
        (folder/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        video=folder/'actual-wm-reference.mp4';result=render_movie(env.model,actual,predicted,reference,indices,anchors,term,row,video)
        metadata.update(result,status='complete',video_sha256=digest(video));(folder/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({key:metadata[key] for key in ('status','clip','source_transitions','rendered_frames','physical_duration_seconds','inference_seconds','render_seconds','video_duration_seconds','quantization_error_seconds')}),flush=True)


if __name__=='__main__':main()
