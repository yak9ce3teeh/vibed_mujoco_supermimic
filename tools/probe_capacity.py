"""Isolated real-pipeline capacity probe; writes only its explicit experiment directory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', required=True)
    p.add_argument('--num-envs', type=int, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--world-updates', type=int, default=2)
    p.add_argument('--actor-updates', type=int, default=1)
    p.add_argument('--blocks', type=int, default=1)
    p.add_argument('--replay-capacity', type=int, default=32768)
    p.add_argument('--reserve-mib', type=int, default=2048)
    p.add_argument('--stress-full-trace', action='store_true')
    p.add_argument('--stress-normalization', action='store_true')
    p.add_argument('--timeout-seconds', type=float, default=120)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-c', '''
import json,runpy,sys,torch
import warp as wp
stress=int(sys.argv.pop(1))
normalization_rows=int(sys.argv.pop(1))
held=torch.empty(stress,device='cuda',dtype=torch.uint8) if stress else None
if held is not None: held.zero_()
torch.cuda.reset_peak_memory_stats()
if normalization_rows:
 import supermimic.training as training
 original=training.finite_modules
 def checked(*modules):
  original(*modules)
  world=modules[1]
  saved={key:value.clone() for key,value in world.named_buffers()}
  x=torch.zeros((normalization_rows,71),device='cuda');x[:,3]=1
  y=x.clone()
  world.fit_scaling(x,y)
  for key,value in world.named_buffers():value.copy_(saved[key])
  del x,y
 training.finite_modules=checked
try: runpy.run_module('supermimic',run_name='__main__')
finally:
 torch.cuda.synchronize()
 print('CAPACITY_MEMORY '+json.dumps(dict(torch_peak_allocated=torch.cuda.max_memory_allocated(),torch_peak_reserved=torch.cuda.max_memory_reserved(),warp_pool_highwater=wp.get_mempool_used_mem_high('cuda:0'),stress_bytes=stress,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),amp=False)),flush=True)
''', str(2*498*args.num_envs*232*4 if args.stress_full_trace else 0),str(args.replay_capacity if args.stress_normalization else 0),
        '--model', '/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml',
        '--motions', 'dataset/processed/lafan_29dof_10s-clipped.pkl',
        '--controller', 'configs/g1_controller.json', '--output', str(args.output/'run'),
        '--device', 'cuda', '--physics-backend', 'warp', '--num-envs', str(args.num_envs),
        '--episodes', str(args.num_envs), '--architecture-profile', args.profile,
        '--batch', str(args.batch), '--world-updates', str(args.world_updates),
        '--actor-updates', str(args.actor_updates), '--blocks', str(args.blocks),
        '--warmup-blocks', '1', '--warmup-max', '1', '--eval-every', str(args.blocks),
        '--checkpoint-every', str(args.blocks), '--replay-capacity',str(args.replay_capacity),'--threads', '2']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='src', OMP_NUM_THREADS='2')
    samples = []
    started = time.monotonic()
    with (args.output/'console.log').open('w') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        stopped = False
        while child.poll() is None:
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
            used, free = map(int, raw.strip().split(','))
            rss = 0
            try:
                rss = int(next(line.split()[1] for line in Path(f'/proc/{child.pid}/status').read_text().splitlines() if line.startswith('VmRSS:')))
            except (FileNotFoundError, StopIteration):
                pass
            samples.append(dict(seconds=time.monotonic()-started, used_mib=used, free_mib=free,rss_kib=rss))
            if free < args.reserve_mib or time.monotonic()-started > args.timeout_seconds:
                child.kill()  # Only this isolated probe, never another process.
                stopped = True
                break
            time.sleep(.05)
        code = child.wait()
    result = dict(config=vars(args)|{'output':str(args.output)}, exit_code=code,
        reserve_stop=stopped, wall_seconds=time.monotonic()-started,
        peak_host_rss_kib=max(x['rss_kib'] for x in samples),sampling_seconds=.05, peak_device_used_mib=max(x['used_mib'] for x in samples),
        min_device_free_mib=min(x['free_mib'] for x in samples), command=command,
        normalization_stress_rows=args.replay_capacity if args.stress_normalization else 0,stress_note='Conservative allocation for two full498-step B-lane float32 trace copies; not a completed physical rollout' if args.stress_full_trace else None)
    for line in (args.output/'console.log').read_text().splitlines():
        if line.startswith('CAPACITY_MEMORY '):result['allocators']=json.loads(line.split(' ',1)[1])
    if (args.output/'run/report.json').exists():
        report=json.loads((args.output/'run/report.json').read_text())
        result.update(reason=report['reason'], total_collected=report.get('total_collected'),
            optimization=report.get('optimization'), gradient=report.get('final_gradient_check'))
    (args.output/'capacity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
