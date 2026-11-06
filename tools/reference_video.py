"""Explicit reference-only 1x kinematic preview; no training or physics steps."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from supermimic.motion import JOINT_NAMES,load_motions
from supermimic.mujoco_adapter import MujocoAdapter
from supermimic.render import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--clip',default='walk1_subject1_clip0')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run=args.run.resolve();report=json.loads((run/'report.json').read_text())
    config=report.get('config',report.get('manifest',{}).get('config'))
    c=json.loads(Path(config['controller']).read_text());model_path=Path(config['model'])
    env=MujocoAdapter(model_path,JOINT_NAMES,c['kp'],c['kd'],c['q_default'],c['action_scale'])
    motion=load_motions(config['motions'],[args.clip],env)[args.clip]
    output=args.output or run/'visuals/reference/start0000';output.mkdir(parents=True,exist_ok=False)
    model=env.model;model.vis.global_.offwidth=960;model.vis.global_.offheight=512
    renderer=mujoco.Renderer(model,height=512,width=960);data=mujoco.MjData(model)
    camera=mujoco.MjvCamera();camera.distance=2.8;camera.azimuth=135;camera.elevation=-15
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    video=output/'reference.mp4'
    try:
        with imageio.get_writer(str(video),fps=50,codec='libx264',pixelformat='yuv420p',macro_block_size=16) as writer:
            for frame,state in enumerate(motion.states):
                data.qpos[:]=state[:36];data.qvel[:]=state[36:];mujoco.mj_forward(model,data)
                camera.lookat[:]=[state[0],state[1],.6]
                renderer.update_scene(data,camera=camera)
                for geom in renderer.scene.geoms[:renderer.scene.ngeom]:
                    if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[geom.objid]:
                        geom.rgba[:]=[.65,.25,1,1];geom.matid=-1;geom.texid=-1
                image=Image.new('RGB',(960,576),(19,25,34));image.paste(Image.fromarray(renderer.render()),(0,64))
                draw=ImageDraw.Draw(image);draw.text((15,8),f'Reference only | {args.clip} | frame {frame} | t={frame*.02:.2f}s | 1x',font=font,fill='white')
                draw.text((15,36),'Camera follows root XY; literal world pose, no physics or final hold',font=font,fill='white')
                writer.append_data(np.asarray(image))
    finally:renderer.close()
    frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(video))
    metadata=dict(io_schema_version=2,reference_only=True,clip=args.clip,video=str(video),sha256=digest(video),
        motion_sha256=digest(Path(config['motions'])),model_xml_sha256=digest(model_path),physical_duration_seconds=(len(motion.states)-1)*.02,
        encoded_frames=frames,encoded_seconds=seconds,output_fps=50,playback_speed=1,final_hold_seconds=0)
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
