"""Controlled native-vs-stock-Warp timing, plus useful collection throughput.

All samples use one environment's dependency versions, XML/controller, clock and
same contact initial state. GPU batch throughput is distinct from single-lane
latency; episode lengths can differ between float64 native and float32 Warp.
"""
import argparse,gc,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import warp as wp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supermimic.mujoco_adapter import MujocoAdapter
from supermimic.warp_adapter import MujocoWarpAdapter
from supermimic.warp_collect import collect_warp
from supermimic.motion import JOINT_NAMES,load_motions
from supermimic.models import Actor, architecture
from supermimic.task import TrackingTask
from supermimic.run_layout import load_weights
from supermimic import training


def process_vram_mib():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    for line in result.stdout.splitlines():
        fields=line.split(',')
        if len(fields)==2 and fields[0].strip()==str(os.getpid()):return float(fields[1])
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physics-batches',type=int,nargs='+',default=[1,64,128,256,512,1024])
    p.add_argument('--collector-batches',type=int,nargs='+',default=[64,128,256])
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--steps',type=int,default=60)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    torch.set_num_threads(1);torch.manual_seed(7);wp.config.quiet=True
    c=json.loads((root/'configs/g1_controller.json').read_text())
    native=MujocoAdapter('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml',JOINT_NAMES,c['kp'],c['kd'],c['q_default'],c['action_scale'])
    motion=load_motions(root/'dataset/processed/lafan_29dof_10s-clipped.pkl',['walk1_subject1_clip0'],native)['walk1_subject1_clip0']
    commands=((motion.states[201:201+a.warmup+a.steps,7:36]-native.q_default)/native.action_scale).astype(np.float32)
    physics=[];times=[]
    for rep in range(a.repeats):
        native.reset(motion.states[200,:36],motion.states[200,36:])
        for action in commands[:a.warmup]:native.step(action)
        start=time.perf_counter()
        for action in commands[a.warmup:]:native.step(action)
        times.append(time.perf_counter()-start)
    physics.append(dict(backend='native',batch=1,seconds=times,median_seconds=float(np.median(times)),aggregate_rate=float(a.steps/np.median(times)),batch_ticks_per_second=float(a.steps/np.median(times)),dtype='float64',checks='finite action/state every step; native derived fields updated after fourth substep'))
    print('PHYSICS '+json.dumps(physics[-1]),flush=True)
    for batch in a.physics_batches:
        torch.cuda.reset_peak_memory_stats()
        adapter=MujocoWarpAdapter(native,batch,check_every_step=True)
        gpu_commands=torch.as_tensor(commands,device='cuda')[:,None].expand(-1,batch,-1).contiguous()
        times=[];memory=[]
        for rep in range(a.repeats):
            adapter.reset(motion.states[200,:36],motion.states[200,36:])
            for action in gpu_commands[:a.warmup]:adapter.step(action)
            adapter.synchronize();start=time.perf_counter()
            for action in gpu_commands[a.warmup:]:adapter.step(action)
            adapter.synchronize();times.append(time.perf_counter()-start);adapter.check();memory.append(process_vram_mib())
        physics.append(dict(backend='warp',batch=batch,seconds=times,median_seconds=float(np.median(times)),aggregate_rate=float(batch*a.steps/np.median(times)),batch_ticks_per_second=float(a.steps/np.median(times)),dtype='float32',overflow_max=int(adapter.overflow.max()),torch_peak_bytes=torch.cuda.max_memory_allocated(),warp_pool_highwater_bytes=wp.get_mempool_used_mem_high('cuda:0'),process_gpu_memory_mib_snapshots=memory,checks='finite action/state and overflow every step; forward after fourth substep'))
        print('PHYSICS '+json.dumps(physics[-1]),flush=True)
        del adapter,gpu_commands;gc.collect();torch.cuda.empty_cache()
    checkpoint=root/'logs/2026-11-05_10-56-46/ckpt/final.pt';saved=load_weights(checkpoint)
    actor=Actor(c['q_default'],action_limit=saved['config'].get('action_limit',20),hidden=architecture(saved['config'])).cuda().eval();actor.load_state_dict(saved['actor'])
    training.DEVICE=torch.device('cuda');task=TrackingTask(native.model).cuda()
    starts=np.resize(np.array([0,200]),64)
    # Equal actor/clip/start mixture, no action or observation noise. Each CPU
    # timing repeats64 actual episodes; GPU timings use their stated batch size.
    collector=[]
    training.rollout(native,motion,0,actor,'actor',np.random.default_rng(7),noise=0)
    torch.cuda.synchronize()
    for rep in range(a.repeats):
        torch.cuda.synchronize();start=time.perf_counter()
        episodes=[training.rollout(native,motion,int(s),actor,'actor',np.random.default_rng(7),noise=0) for s in starts]
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        total=sum(len(e['actions']) for e in episodes)
        collector.append(dict(backend='native',batch=1,episodes=64,repeat=rep,seconds=seconds,valid_transitions=total,valid_rate=total/seconds,aggregate_lane_ticks=total,active_utilization=1.,lengths=[len(e['actions']) for e in episodes],completed=sum(e['completed'] for e in episodes)))
        print('COLLECTOR '+json.dumps({k:v for k,v in collector[-1].items() if k!='lengths'}),flush=True)
    for batch in a.collector_batches:
        torch.cuda.reset_peak_memory_stats();adapter=MujocoWarpAdapter(native,batch)
        fixed=np.resize(np.array([0,200]),batch)
        collect_warp(adapter,motion,actor,np.random.default_rng(7),batch,task,training.noisy_observation,noise=0,starts_override=fixed)
        for rep in range(a.repeats):
            adapter.synchronize();start=time.perf_counter()
            episodes,stats=collect_warp(adapter,motion,actor,np.random.default_rng(7),batch,task,training.noisy_observation,noise=0,starts_override=fixed)
            adapter.synchronize();seconds=time.perf_counter()-start
            collector.append(dict(backend='warp',batch=batch,episodes=len(episodes),repeat=rep,seconds=seconds,
                valid_transitions=stats['active_transitions'],valid_rate=stats['active_transitions']/seconds,
                aggregate_lane_ticks=stats['aggregate_world_control_ticks'],active_utilization=stats['active_utilization'],
                lengths=[len(e['actions']) for e in episodes],completed=sum(e['completed'] for e in episodes),
                torch_peak_bytes=torch.cuda.max_memory_allocated(),warp_pool_highwater_bytes=wp.get_mempool_used_mem_high('cuda:0'),
                process_gpu_memory_mib=process_vram_mib()))
            print('COLLECTOR '+json.dumps({k:v for k,v in collector[-1].items() if k!='lengths'}),flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()
    result=dict(physics=physics,collector=collector,warmup=a.warmup,measured_ticks=a.steps,repeats=a.repeats,
        actor_checkpoint=str(checkpoint),actor_width=128,physics_dt=.005,substeps=4,control_dt=.02,torch_threads=1,
        physics_action_protocol='Same ref201 onward PD target commands, same contact ref200 reset, same time window and full per-step finite checks; single native trajectory rate compared with aggregate independent GPU worlds',
        collector_protocol='Same final CUDA actor, clip, alternating reset0/200 and noise0; native production serial rollout vs GPU production collect_warp. Floating-point/contact/policy batch differences can change episode lengths, so not pure backend causality.',
        memory_note='Torch peak and Warp pool highwater are different allocators; process NVML values are snapshots, not an exact transient peak; Warp highwater is cumulative within this process.',
        compile_note='Initial cold CCD compilation about60seconds, excluded from cached timing; stock MULTICCD cylinder pairs at most1contact warning applies, no XML edits.')
    if a.evidence:a.evidence.write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),layout='constrained')
    cpu=physics[0]['aggregate_rate'];xs=[x['batch'] for x in physics[1:]];ys=[x['aggregate_rate'] for x in physics[1:]]
    axes[0].plot(xs,ys,'o-',color='#286da5',label='Warp CUDA: aggregate worlds')
    axes[0].axhline(cpu,color='#b45b32',ls='--',label='Native CPU: serial1world')
    axes[0].set(xscale='log',yscale='log',xlabel='GPU parallel environments',ylabel='Aggregate control transitions / second',title='Physics only: identical command window')
    axes[0].set_xticks(xs,[str(x) for x in xs]);axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    native_rows=[r for r in collector if r['backend']=='native'];native_rate=float(np.median([r['valid_rate'] for r in native_rows]))
    labels=['Native serial\n64episodes'];rates=[native_rate]
    for batch in a.collector_batches:
        rows=[r for r in collector if r['backend']=='warp' and r['batch']==batch]
        labels.append(f'Warp {batch}\n{batch}episodes');rates.append(float(np.median([r['valid_rate'] for r in rows])))
    axes[1].bar(labels,rates,color=['#b45b32']+['#286da5']*len(a.collector_batches))
    for i,rate in enumerate(rates):axes[1].text(i,rate,f'{rate:.0f}/s\n{rate/native_rate:.2f}×',ha='center',va='bottom',fontsize=9)
    axes[1].set(ylabel='Actual saved episode transitions / second',title='Actor + physics + reward + replay collection')
    axes[1].set_ylim(0,max(rates)*1.22);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('CPU native vs GPU stock MuJoCo Warp | RTX4060Ti16GB',fontsize=13)
    fig.supxlabel('Median of3 warm timings; 50Hz control =4×5ms physics. Right: same CUDA actor, reset mixture, noise0; differing lengths retained, idle lane ticks excluded.',fontsize=9)
    fig.savefig(a.output,dpi=220);plt.close(fig)
    print('FIGURE '+str(a.output),flush=True)


if __name__=='__main__':main()
