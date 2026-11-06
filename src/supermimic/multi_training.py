"""Full-recording, fixed-step multi-motion SuperMimic training (fixed physics).

Physical resume intentionally resets live lanes; it is not bitexact continuation.
The learner/optimizer/RNG/replay/curriculum state is preserved at block boundaries.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import tarfile
import time
import numpy as np
import torch
from tqdm import tqdm
from . import training as core
from .models import architecture, build_models
from .motion import JOINT_NAMES, load_motions
from .mujoco_adapter import MujocoAdapter
from .warp_adapter import MujocoWarpAdapter
from .task import TrackingTask
from .curriculum import StartBins
from .multi_collect import StreamCollector
from .multi_replay import sample_windows, policy_batch, trim_segments
from .run_layout import case_id, content_digest, write_packed, stage_key, archive_terminal_before_resume


RESUME_MODE='reset_live_lanes_on_resume'


def source_hash():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()


def fixed_validation(env,motions,actor):
    episodes=[];starts=[]
    for name,motion in motions.items():
        fraction=.2+.6*(int(hashlib.sha256(name.encode()).hexdigest()[:8],16)/0xffffffff)
        candidates=[int(fraction*(len(motion.states)-65)),int(.35*(len(motion.states)-65)),int(.65*(len(motion.states)-65))]
        best=None
        for start in candidates:
            ep=core.rollout(env,motion,start,actor,'actor',np.random.default_rng(7007),noise=0,max_steps=64)
            if best is None or len(ep['actions'])>len(best['actions']):best=ep
            if len(ep['actions'])>=8:break
        episodes.append(best);starts.append(dict(clip=name,start=best['start'],steps=len(best['actions']),eligible=len(best['actions'])>=8))
    if sum(x['eligible'] for x in starts)<32:raise RuntimeError('Fewer than32/40 fixed validation recordings have a legalK8 window')
    return episodes,starts


@torch.no_grad()
def validation_error(world,episodes,motions,reward_mean):
    windows=[]
    for ep in episodes:
        for i in range(0,min(len(ep['actions'])-7,32),8):windows.append((ep,i))
    state=core.tensor(np.stack([e['states'][i] for e,i in windows]));initial=state.clone()
    errors=[];persistence=[]
    for k in range(8):
        state=world(state,core.tensor(np.stack([e['actions'][i+k] for e,i in windows])))
        target=core.tensor(np.stack([e['next_states'][i+k] for e,i in windows]))
        errors.append((state-target).square().cpu().numpy());persistence.append((initial-target).square().cpu().numpy())
    groups={name:dict(rmse=float(np.sqrt(np.mean(np.asarray(errors)[...,sl]))),persistence_rmse=float(np.sqrt(np.mean(np.asarray(persistence)[...,sl])))) for name,sl in [('root_position_m',slice(0,3)),('joint_position_rad',slice(7,36)),('root_angular_velocity_rad_s',slice(39,42)),('root_linear_velocity_m_s',slice(36,39)),('joint_velocity_rad_s',slice(42,71))]}
    samples=[(e,j) for e,i in windows for j in range(i,i+8)]
    previous=core.tensor(np.stack([e['actions'][j-1] if j else np.zeros(29) for e,j in samples]))
    actual=core.tensor(np.stack([e['states'][j] for e,j in samples]));action=core.tensor(np.stack([e['actions'][j] for e,j in samples]));ref=core.tensor(np.stack([motions[e['clip']].states[e['frames'][j]+1] for e,j in samples]))
    reward,alive,_=core.TASK(world(actual,action),ref,action,previous)
    target=core.tensor(np.asarray([e['rewards'][j] for e,j in samples]));labels=core.tensor(np.asarray([e['dones'][j] for e,j in samples],dtype=np.float32))
    a=reward.cpu().numpy();b=target.cpu().numpy();prob=(1-alive).cpu().numpy();done=labels.cpu().numpy()
    outcomes=dict(reward_rmse=float(np.sqrt(np.mean((a-b)**2))),reward_constant_mean_rmse=float(np.sqrt(np.mean((b-reward_mean)**2))),reward_correlation=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else 0.,terminal_brier=float(np.mean((prob-done)**2)),count=len(samples),positive_count=int(done.sum()))
    return dict(groups=groups,count=len(windows)*8,windows=len(windows),definition='Fixed initial-actor episodes from the same40 training recordings; K1..8 aggregate, not unseen-motion generalization'),outcomes


def evaluate_benchmark(env,motions,actor,rng,monitor_clip):
    rows=[];cases={};started=time.monotonic()
    for name,motion in motions.items():
        ep=core.rollout(env,motion,0,actor,'actor',rng,noise=0)
        row=core.summarize(ep,.02);row.update(policy='trained',evaluation_role='all40_primary_start0',failure=bool(ep['dones'][-1]))
        key=case_id(name,0,'trained');row.update(trajectory_key=key,trajectory_bundle='trajectories.npz')
        record={field:ep[field] for field in ('states','next_states','actions','raw_actions','rewards','dones','truncated','frames','reward_terms','reward_term_names','start','clip','remaining_transitions','action_limit')}
        row['trajectory_source_sha256']=content_digest(record);cases[key]=record;rows.append(row)
    primary_seconds=time.monotonic()-started
    ep=core.rollout(env,motions[monitor_clip],200,actor,'actor',rng,noise=0)
    row=core.summarize(ep,.02);row.update(policy='trained',evaluation_role='representative_walk_reset200_auxiliary',failure=bool(ep['dones'][-1]))
    key=case_id(monitor_clip,200,'trained');record={field:ep[field] for field in ('states','next_states','actions','raw_actions','rewards','dones','truncated','frames','reward_terms','reward_term_names','start','clip','remaining_transitions','action_limit')}
    row.update(trajectory_key=key,trajectory_bundle='trajectories.npz',trajectory_source_sha256=content_digest(record));cases[key]=record
    benchmark=dict(clip_count=len(rows),failure_count=sum(r['failure'] for r in rows),completed_count=sum(r['completed'] for r in rows),macro_normalized_return=float(np.mean([r['normalized_return'] for r in rows])),macro_global_mpjpe=float(np.mean([r['global_mpjpe'] for r in rows])),macro_joint_rmse=float(np.mean([r['joint_rmse'] for r in rows])),total_executed_steps=sum(r['steps'] for r in rows),total_reference_steps=sum(r['remaining_transitions'] for r in rows),seconds=primary_seconds,steps_per_second=sum(r['steps'] for r in rows)/primary_seconds,scope='All40 training recordings, in-distribution; deterministic first-frame full-reference, fixed physics, no observation/action noise')
    return rows+[row],cases,benchmark


def setup(args):
    core.DEVICE=torch.device(args.device);torch.set_num_threads(args.threads)
    controller=json.loads(Path(args.controller).read_text())
    env=MujocoAdapter(args.model,JOINT_NAMES,controller['kp'],controller['kd'],controller['q_default'],controller['action_scale'])
    core.TASK=TrackingTask(env.model).to(core.DEVICE)
    manifest=json.loads(Path(args.motions).with_name('manifest.json').read_text());names=[r['recording_id'] for r in manifest['recordings']]
    if core.file_hash(args.motions)!=manifest['canonical_sha256']:raise ValueError('Canonical full-recording file hash differs from manifest')
    motions=load_motions(args.motions,names,env)
    return controller,env,motions,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml')
    p.add_argument('--motions',default='dataset/processed/lafan1_full_g1/motions-50hz.joblib')
    p.add_argument('--controller',default='configs/g1_controller.json');p.add_argument('--output',required=True)
    p.add_argument('--num-envs',type=int,default=2048);p.add_argument('--chunk-steps',type=int,default=64)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--world-updates',type=int,default=100);p.add_argument('--actor-updates',type=int,default=10)
    p.add_argument('--warmup-blocks',type=int,default=5);p.add_argument('--warmup-max',type=int,default=20);p.add_argument('--warmup-gate',action='store_true')
    p.add_argument('--blocks',type=int,default=5000);p.add_argument('--max-hours',type=float,default=10)
    p.add_argument('--eval-every',type=int,default=25);p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--replay-capacity',type=int,default=2097152);p.add_argument('--seed',type=int,default=7);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--architecture-profile',default='paper-1',choices=['legacy','paper-0.125','paper-0.25','paper-0.5','paper-1']);p.add_argument('--width',type=int,default=128)
    p.add_argument('--action-limit',type=float,default=20.);p.add_argument('--device',default='cuda',choices=['cuda']);p.add_argument('--resume');p.add_argument('--eval-checkpoint')
    p.add_argument('--monitor-clip',default='walk1_subject1')
    args=p.parse_args();args.dataset_mode='full40_stream';args.physics_backend='warp';args.architecture=architecture(args)
    args.resume_mode=RESUME_MODE;args.training_gates=False
    if min(args.num_envs,args.chunk_steps,args.batch,args.world_updates,args.actor_updates,args.eval_every,args.checkpoint_every)<=0 or args.warmup_max<args.warmup_blocks:p.error('Invalid positive counts/warmup bounds')
    if args.eval_checkpoint:
        evaluate_checkpoint(args.eval_checkpoint,args.output);return
    out=Path(args.output).resolve()
    if (out/'report.json').exists() and not args.resume:raise FileExistsError('Use a fresh datetime run or explicit resume')
    for folder in ('ckpt','logs','evals','visuals'):(out/folder).mkdir(parents=True,exist_ok=True)
    for name in ('episodes','eval','consol'):(out/f'logs/{name}.jsonl').touch()
    torch.manual_seed(args.seed);np.random.seed(args.seed);rng=np.random.default_rng(args.seed)
    controller,env,motions,manifest=setup(args)
    actor,world,critic=(m.to(core.DEVICE) for m in build_models(env.q_default,args));initial=copy.deepcopy(actor)
    optimizers=[torch.optim.Adam(actor.parameters(),lr=2e-4,betas=(.9,.95)),torch.optim.Adam(world.parameters(),lr=2e-4),torch.optim.Adam(critic.parameters(),lr=2e-4)]
    progress=core.OptimizationProgress(args.world_updates,args.actor_updates);curriculum=StartBins(motions)
    identity=dict(configuration={k:v for k,v in vars(args).items() if k not in ('output','resume','eval_checkpoint','blocks','max_hours')},assets=core.asset_identity(args,env),recordings=list(motions),resume_mode=RESUME_MODE)
    frozen_source=source_hash();replay=[];validation=None
    state=dict(warm_blocks=0,joint_blocks=0,warm_complete=args.warmup_blocks==0,total_collected=0,reason='running',training=[],eval_history=[],best_initial=None,best_trained=None,elapsed_before=0.,reserve_seconds=900.,evaluation_seconds=0.,next_episode_id=0)
    reset_receipt=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('multi_train_state_version')!=1 or saved['identity']!=identity or saved['source_sha256']!=frozen_source:raise ValueError('Multi resume identity/source mismatch; no single/Warp-exact migration')
        archive_terminal_before_resume(out,saved)
        for module,key in [(actor,'actor'),(initial,'initial_actor'),(world,'world'),(critic,'critic')]:module.load_state_dict(saved[key])
        for optimizer,key in zip(optimizers,('actor_optimizer','world_optimizer','critic_optimizer')):optimizer.load_state_dict(saved[key])
        core.load_progress(progress,saved['optimization']);state=saved['run_state'];replay=saved['replay'];validation=saved['validation'];curriculum.load_state_dict(saved['curriculum']);core.restore_rng(saved['rng'],rng)
        live=saved['live_lanes'];reset_receipt=dict(resume_mode=RESUME_MODE,reset_live_lanes=len(live['episode_ids']),discarded_partial_episode_count=int((live['lengths']>0).sum()),discarded_partial_steps=int(live['lengths'].sum()),not_hard_failure=True,learner_replay_curriculum_restored=True)
        state['reason']='running'
    adapter=MujocoWarpAdapter(env,args.num_envs)
    collector=StreamCollector(adapter,motions,actor,core.TASK,curriculum,core.noisy_observation,state['next_episode_id'])
    started=time.monotonic();elapsed_before=state['elapsed_before'];flag={'stop':False}
    def stop_handler(*_):flag['stop']=True
    signal.signal(signal.SIGINT,stop_handler);signal.signal(signal.SIGTERM,stop_handler)
    def elapsed():return elapsed_before+time.monotonic()-started
    previous_report=json.loads((out/'report.json').read_text()) if args.resume and (out/'report.json').exists() else {}
    root_report=dict(io_schema_version=2,dataset_mode=args.dataset_mode,config=vars(args),manifest=dict(config=vars(args),source_sha256=frozen_source,assets=identity['assets'],data_manifest=str(Path(args.motions).with_name('manifest.json').resolve()),data_manifest_sha256=core.file_hash(Path(args.motions).with_name('manifest.json')),recordings=list(motions)),resume_parameters=[] if not reset_receipt else [reset_receipt],physics_sampler=adapter.metadata,canonical_eval_backend='native',resume_mode=RESUME_MODE,reason='running')
    if previous_report:
        root_report={**previous_report,**root_report}
        root_report['manifest']=previous_report.get('manifest',root_report['manifest'])
        root_report['resume_parameters']=previous_report.get('resume_parameters',[])+([reset_receipt] if reset_receipt else [])
        for name in ('fixed_validation','warmup_gradient_check','preflight_gradient_check'):
            if name in previous_report:root_report[name]=previous_report[name]
    if not (out/'ckpt/source.tar.gz').exists():
        with tarfile.open(out/'ckpt/source.tar.gz','w:gz') as archive:
            for path in Path(__file__).parent.glob('*.py'):archive.add(path,arcname='src/supermimic/'+path.name)
    def report():
        root_report.update(reason=state['reason'],joint_blocks=state['joint_blocks'],warm_blocks=state['warm_blocks'],total_collected=state['total_collected'],elapsed_seconds=elapsed(),optimization=progress.audit(),checkpoint='ckpt/final.pt',best_initial=state['best_initial'],best_trained=state['best_trained'],final_evaluation_reserve_seconds=state['reserve_seconds'],evaluation_seconds=state['evaluation_seconds'])
        core.atomic_save(root_report,out/'report.json',True)
    def payload():return core.weights_payload(actor,initial,world,critic,args,controller,progress)
    def save_full():
        state['elapsed_before']=elapsed();state['next_episode_id']=collector.next_episode_id
        value=payload();value.update(multi_train_state_version=1,train_state_version=1,source_sha256=frozen_source,identity=identity,run_state=state,replay=replay,validation=validation,curriculum=curriculum.state_dict(),live_lanes=collector.live_state(),rng=core.rng_state(rng),resume_mode=RESUME_MODE,actor_optimizer=optimizers[0].state_dict(),world_optimizer=optimizers[1].state_dict(),critic_optimizer=optimizers[2].state_dict())
        core.atomic_save(value,out/'ckpt/final.pt')
        core.atomic_save(dict(joint_block=state['joint_blocks'],warm_blocks=state['warm_blocks'],total_collected=state['total_collected'],checkpoint_kind='full_learner_resume_reset_live_lanes',resume_mode=RESUME_MODE),out/'ckpt/final.json',True)
    def evaluate(stage):
        folder=out/'evals'/stage;folder.mkdir(exist_ok=True);cp=out/'ckpt'/f'{stage}.pt'
        if stage=='final':save_full()
        else:core.atomic_save(payload(),cp)
        cache=next((r for r in reversed(state['eval_history']) if r['joint_block']==state['joint_blocks'] and r['warm_blocks']==state['warm_blocks']),None)
        beginning=time.monotonic()
        if cache:
            old_folder=Path(cache['path']);data=json.loads((old_folder/'report.json').read_text());data=copy.deepcopy(data)
            import shutil
            shutil.copyfile(old_folder/'trajectories.npz',folder/'trajectories.npz')
            data['reused_same_frozen_model_stage']=cache['stage']
        else:
            with core.preserve_rng(rng):rows,cases,benchmark=evaluate_benchmark(env,motions,actor,rng,args.monitor_clip)
            write_packed(folder/'trajectories.npz',cases)
            data=dict(evaluation=rows,benchmark=benchmark,controller=controller,physics=core.foot_friction_one(env))
        benchmark=data['benchmark'];state['evaluation_seconds']+=time.monotonic()-beginning
        rate=benchmark['steps_per_second'];state['reserve_seconds']=max(state['reserve_seconds'],900.,1.25*manifest['runtime_transitions']/rate+120)
        data.update(io_schema_version=2,dataset_mode=args.dataset_mode,config=vars(args),stage=stage,joint_block=state['joint_blocks'],warm_blocks=state['warm_blocks'],total_collected=state['total_collected'],checkpoint=str(cp),checkpoint_sha256=core.file_hash(cp),checkpoint_kind='full_learner_resume_reset_live_lanes' if stage=='final' else 'weights_only',optimization=progress.audit(),source_sha256=frozen_source,paper_main=dict(normalized_return=benchmark['macro_normalized_return'],failure_count=benchmark['failure_count'],completed_count=benchmark['completed_count'],steps=benchmark['total_executed_steps'],completed=benchmark['completed_count']==40,definition='Macro over40 full recordings, not a single-clip score'),train_auxiliary_macro=benchmark,canonical_eval_backend='native',data_manifest_sha256=root_report['manifest']['data_manifest_sha256'])
        core.atomic_save(data,folder/'report.json',True)
        row=dict(stage=stage,joint_block=state['joint_blocks'],warm_blocks=state['warm_blocks'],total_collected=state['total_collected'],path=str(folder),checkpoint=str(cp),checkpoint_sha256=data['checkpoint_sha256'],checkpoint_kind=data['checkpoint_kind'],paper_main=data['paper_main'],score=benchmark)
        state['eval_history'].append(row);core.append_log(out/'logs/eval.jsonl',row);core.atomic_save(row,cp.with_suffix('.json'),True)
        key=(benchmark['completed_count'],benchmark['macro_normalized_return'],-benchmark['macro_global_mpjpe'])
        if stage=='initial':state['best_initial']=row
        if state['joint_blocks']>0 and (state['best_trained'] is None or key>tuple(state['best_trained']['ranking'])):
            row['ranking']=list(key);state['best_trained']=copy.deepcopy(row)
            core.atomic_save(payload(),out/'best.pt');core.atomic_save(row,out/'best.json',True)
        elif not (out/'best.pt').exists():core.atomic_save(payload(),out/'best.pt');core.atomic_save(dict(row,best_is_initial=True),out/'best.json',True)
        best_path=out/'best.json'
        if best_path.exists():
            best=json.loads(best_path.read_text());best.update(checkpoint=str(out/'best.pt'),checkpoint_sha256=core.file_hash(out/'best.pt'));core.atomic_save(best,best_path,True)
        report();tqdm.write(json.dumps(dict(stage=stage,benchmark=benchmark)))
        return data
    report()
    if validation is None:
        with core.preserve_rng(rng):validation,validation_plan=fixed_validation(env,motions,initial)
        root_report['fixed_validation']=validation_plan
        evaluate('initial')
        save_full()
    last_save=time.monotonic();bar=tqdm(total=args.blocks+args.warmup_max,initial=state['joint_blocks']+state['warm_blocks'],disable=not __import__('sys').stderr.isatty(),desc='full40 blocks')
    try:
        while state['joint_blocks']<args.blocks:
            if flag['stop']:state['reason']='signal';break
            if elapsed()>=args.max_hours*3600-state['reserve_seconds']:state['reason']='wall_time_budget';break
            warm=not state['warm_complete'];begin=time.monotonic()
            segments,episodes,collection=collector.collect(args.chunk_steps);replay.extend(segments)
            resident=trim_segments(replay,args.replay_capacity);state['total_collected']+=collection['active_transitions']
            state.setdefault('reward_baseline_mean',sum(float(e['rewards'][e['prefix']:].sum()) for e in segments)/collection['active_transitions'])
            collected=time.monotonic()
            if warm or not progress.steps['world']:
                world.fit_scaling(core.tensor(np.concatenate([e['states'][e['prefix']:] for e in replay])),core.tensor(np.concatenate([e['next_states'][e['prefix']:] for e in replay])))
            if 'preflight_gradient_check' not in root_report:
                with core.preserve_rng(rng):root_report['preflight_gradient_check']=core.directional_check(actor,world,critic,replay,motions,rng,batch_fn=policy_batch)
                if not root_report['preflight_gradient_check']['passed']:raise FloatingPointError('Initial H32 directional derivative failed')
            world_losses=[];wm=core.train_world(world,optimizers[1],replay,motions,rng,args.world_updates,args.batch,progress=progress,warmup=warm,details=world_losses,sample_fn=sample_windows)
            world_end=time.monotonic();updates=[] if warm else [core.train_actor(actor,world,critic,optimizers[0],optimizers[2],replay,motions,rng,args.batch,progress=progress,batch_fn=policy_batch) for _ in range(args.actor_updates)]
            core.finite_modules(actor,world,critic)
            if warm:state['warm_blocks']+=1
            else:state['joint_blocks']+=1
            validation_world,validation_outcomes=validation_error(world,validation,motions,state['reward_baseline_mean'])
            status=dict(record_type='block',phase='warmup' if warm else 'joint',warm_block=state['warm_blocks'],joint_block=state['joint_blocks'],total_collected=state['total_collected'],resident_transitions=resident,resident_segments=len(replay),collection=collection,episodes=episodes,true_finished_episodes=len(episodes),segments=len(segments),world_loss=wm,world_losses=world_losses,actor_updates=updates,optimization=progress.audit(),heldout_world=validation_world,heldout_outcomes=validation_outcomes,fixed_validation_scope='Same40 recordings; fixed distinct episodes, not heldout motions',timing=dict(collect=collected-begin,world=world_end-collected,other=time.monotonic()-world_end),elapsed_seconds=elapsed(),curriculum=dict(bins=len(curriculum.bins),visited_start_bins=int((curriculum.counts>0).sum()),finished_episode_updates=int(curriculum.counts.sum()),mix=[.5,.4,.1],Gmax=10.,historic_count=curriculum.historic_count))
            state['training']=[status];core.append_log(out/'logs/episodes.jsonl',status)
            actor_mean=float(np.mean([u['actor_loss'] for u in updates])) if updates else None
            critic_mean=float(np.mean([v for u in updates for v in u['critic_losses']])) if updates else None
            line=f"full40 {'warm' if warm else 'joint'} warm={state['warm_blocks']} joint={state['joint_blocks']}/{args.blocks} real={state['total_collected']} done={len(episodes)} segments={len(segments)} WM={wm:.4g} actor={actor_mean if updates else 'n/a'} critic={critic_mean if updates else 'n/a'} elapsed={elapsed():.1f}s"
            tqdm.write(line);bar.update(1);bar.set_postfix_str(line);core.append_log(out/'logs/consol.jsonl',dict(summary=line))
            if warm and state['warm_blocks']>=args.warmup_blocks:
                g=validation_world['groups'];gate=g['root_position_m']['rmse']<g['root_position_m']['persistence_rmse'] and g['joint_position_rad']['rmse']<g['joint_position_rad']['persistence_rmse'] and validation_outcomes['reward_rmse']<validation_outcomes['reward_constant_mean_rmse']
                if gate or not args.warmup_gate:
                    with core.preserve_rng(rng):fd=core.directional_check(actor,world,critic,replay,motions,rng,batch_fn=policy_batch)
                    root_report['warmup_gradient_check']=fd
                    if not fd['passed']:raise FloatingPointError('Warmup H32 directional derivative failed')
                    state['warm_complete']=True;evaluate('warmup-complete')
                elif state['warm_blocks']>=args.warmup_max:state['reason']='warmup_gate_failed';break
            if not warm and state['joint_blocks']%args.eval_every==0:evaluate(f"{state['joint_blocks']:04d}")
            if state['joint_blocks']%args.checkpoint_every==0 or time.monotonic()-last_save>=300:save_full();last_save=time.monotonic()
            report()
        if state['reason']=='running':state['reason']='joint_budget_complete'
        if replay:
            with core.preserve_rng(rng):root_report['final_gradient_check']=core.directional_check(actor,world,critic,replay,motions,rng,batch_fn=policy_batch)
            if not root_report['final_gradient_check']['passed']:state['reason']='gradient_check_failed'
        else:root_report['final_gradient_check']=dict(tested=False,reason='No collected replay before safe stop')
    except Exception as error:
        state['reason']='failure';root_report['failure']=dict(type=type(error).__name__,message=str(error));report();save_full();raise
    finally:bar.close()
    evaluate('final');save_full()
    final_path=out/'evals/final/report.json';final=json.loads(final_path.read_text());old_hash=final['checkpoint_sha256'];final.update(checkpoint_sha256=core.file_hash(out/'ckpt/final.pt'),evaluated_weights_sha256=old_hash);core.atomic_save(final,final_path,True)
    log_path=out/'logs/eval.jsonl'
    rows=[json.loads(line) for line in log_path.open()]
    for row in rows:
        if row['stage']=='final':row.update(checkpoint_sha256=final['checkpoint_sha256'],evaluated_weights_sha256=old_hash)
    temporary=log_path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows));temporary.replace(log_path)
    core.atomic_save(dict(rows[-1],resume_mode=RESUME_MODE),out/'ckpt/final.json',True)
    root_report['final_evaluation']=final['benchmark'];report()


def evaluate_checkpoint(path,output,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=False);args=argparse.Namespace(**saved['config']);args.device=device
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty independent evaluation output')
    out.mkdir(parents=True,exist_ok=True);controller,env,motions,_=setup(args)
    actor,_,_=build_models(env.q_default,args);actor.load_state_dict(saved['actor']);actor.to(core.DEVICE)
    with torch.no_grad():rows,cases,benchmark=evaluate_benchmark(env,motions,actor,np.random.default_rng(args.seed),args.monitor_clip)
    write_packed(out/'trajectories.npz',cases)
    result=dict(dataset_mode='full40_stream',config=vars(args),evaluation=rows,benchmark=benchmark,paper_main=dict(normalized_return=benchmark['macro_normalized_return'],failure_count=benchmark['failure_count'],completed_count=benchmark['completed_count'],definition='Macro over40 full training recordings'),checkpoint=str(Path(path).resolve()),checkpoint_sha256=core.file_hash(path),source_optimization=saved.get('optimization',{'history':'unknown'}),evaluation_only=True,controller=controller)
    core.atomic_save(result,out/'report.json',True);print(json.dumps(benchmark),flush=True)
    return result


if __name__=='__main__':main()
