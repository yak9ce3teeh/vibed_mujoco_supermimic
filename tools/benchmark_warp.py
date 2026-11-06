"""Cached stock-Warp physics-only throughput; not end-to-end training speed."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import warp as wp
from supermimic.mujoco_adapter import MujocoAdapter
from supermimic.motion import JOINT_NAMES,load_motions
from supermimic.warp_adapter import MujocoWarpAdapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--model',type=Path,default=Path('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml'))
    parser.add_argument('--motions',type=Path,default=Path('dataset/processed/lafan_29dof_10s-clipped.pkl'))
    parser.add_argument('--controller',type=Path,default=Path('configs/g1_controller.json'))
    args=parser.parse_args();torch.set_num_threads(1)
    c=json.loads(args.controller.read_text());native=MujocoAdapter(args.model,JOINT_NAMES,c['kp'],c['kd'],c['q_default'],c['action_scale'])
    motion=load_motions(args.motions,['walk1_subject1_clip0'],native)['walk1_subject1_clip0']
    free,total=torch.cuda.mem_get_info();baseline_used=total-free;torch.cuda.reset_peak_memory_stats()
    adapter=MujocoWarpAdapter(native,args.num_envs,check_every_step=False)
    adapter.reset(motion.states[200,:36],motion.states[200,36:])
    action=torch.as_tensor((motion.states[200,7:36]-native.q_default)/native.action_scale,device='cuda',dtype=torch.float32)[None].expand(args.num_envs,-1).contiguous()
    for _ in range(args.warmup):adapter.step(action)
    adapter.synchronize();adapter.check()
    start=time.perf_counter()
    for _ in range(args.steps):adapter.step(action)
    adapter.synchronize();seconds=time.perf_counter()-start;adapter.check()
    free,total=torch.cuda.mem_get_info()
    result=dict(backend=adapter.metadata,num_envs=args.num_envs,warmup_control_ticks=args.warmup,
        measured_control_ticks=args.steps,physics_substeps_per_tick=4,measured_seconds=seconds,
        aggregate_world_control_steps_per_second=args.num_envs*args.steps/seconds,
        aggregate_world_physics_steps_per_second=4*args.num_envs*args.steps/seconds,
        control_calls_per_second=args.steps/seconds,overflow=adapter.overflow.cpu().tolist(),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        warp_pool_used_bytes=wp.get_mempool_used_mem_current('cuda:0'),
        warp_pool_highwater_bytes=wp.get_mempool_used_mem_high('cuda:0'),
        cuda_global_used_before_bytes=baseline_used,cuda_global_used_after_bytes=total-free,
        global_memory_note='Device-global snapshots include desktop/other processes; Warp pool highwater and Torch allocator peak are separate allocators, do not add as exact process peak.',
        protocol='Fixed reference-frame200 joint target after20warm ticks, full physics continues after falls; all lanes counted, no episode masking. Not collector/training throughput.',
        finite_state=bool(torch.isfinite(adapter.state()).all()),
        root_motor_max_abs=float(adapter.ctrl[:,:6].abs().max()),
        time_range_seconds=[float(adapter.time.min()),float(adapter.time.max())])
    print('BENCHMARK '+json.dumps(result),flush=True)


if __name__=='__main__':main()
