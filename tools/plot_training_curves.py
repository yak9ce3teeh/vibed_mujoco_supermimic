"""Raw periodic evaluation curves: dimensionless WM ratios and real returns."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from supermimic.run_layout import evaluation_folder, evaluation_folders, stage_iteration, block_records, checkpoint_path, stage_key, load_weights as load_checkpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def aggregate_training_block(row,index):
    """Scalar-only block summary; segment/cap boundaries are never real returns."""
    episodes=row.get('episodes',[])
    ended=[e for e in episodes if e.get('termination_cause') in ('hard_fall','reference_end')]
    def mean(values):return float(np.mean(values)) if values else None
    raw=[float(e['return_sum']) for e in ended] if all('return_sum' in e for e in ended) else []
    normalized=[float(e['normalized_return']) for e in ended] if all('normalized_return' in e for e in ended) else []
    steps=sum(int(e['steps']) for e in ended) if all('steps' in e for e in ended) else 0
    world=[float(v) for v in row.get('world_losses',[])]
    updates=row.get('actor_updates',[])
    actor=[float(u['actor_loss']) for u in updates if 'actor_loss' in u]
    critic=[float(v) for u in updates for v in u.get('critic_losses',[])]
    return dict(global_block_index=index,phase=row.get('phase'),warm_block=row.get('warm_block'),joint_block=row.get('joint_block'),
        finished_count=len(ended),reference_end_count=sum(e['termination_cause']=='reference_end' for e in ended),
        excluded_unknown_or_segment_count=len(episodes)-len(ended),ended_steps_sum=steps,
        normalized_return_mean=mean(normalized),return_sum_mean=mean(raw),
        ended_episode_reward_per_step=float(sum(raw)/steps) if raw and steps else None,
        world_loss_mean=mean(world),actor_loss_mean=mean(actor),critic_loss_mean=mean(critic),
        world_update_count=len(world),actor_update_count=len(actor),critic_update_count=len(critic))


def phase_ewma(values,phases,alpha=.1):
    result=[];previous_phase=None;state=None
    for value,phase in zip(values,phases):
        if phase!=previous_phase:state=None
        previous_phase=phase
        if value is None:result.append(np.nan);state=None;continue
        state=value if state is None else alpha*value+(1-alpha)*state
        result.append(state)
    return result


def plot_training_parameters(points,output):
    if not points:return
    figure,axes=plt.subplots(2,3,sharex=True,figsize=(15,7.8),layout='constrained')
    panels=[('normalized_return_mean','Ended episodes: normalized return','Episode-macro full-reference normalized return'),
            ('return_sum_mean','Ended episodes: raw return','Episode-macro undiscounted return'),
            ('ended_episode_reward_per_step','Ended episodes: pooled per-step reward','Sum episode returns / sum episode steps'),
            ('world_loss_mean','World-model loss','Mean of all WM-update batch/K8 MSEs'),
            ('actor_loss_mean','Actor loss (sign retained)','Mean of all negative imagined-return losses'),
            ('critic_loss_mean','Critic loss','Mean of every critic-update target MSE')]
    x=[p['global_block_index'] for p in points];phases=[p['phase'] for p in points]
    warm=[p['global_block_index'] for p in points if p['phase']=='warmup']
    for axis,(key,title,label) in zip(axes.flat,panels):
        values=[p[key] for p in points];raw=[np.nan if v is None else v for v in values]
        axis.plot(x,raw,color='#2374ab',alpha=.32,lw=.8,label='Raw block aggregate')
        axis.plot(x,phase_ewma(values,phases),color='#2374ab',lw=1.7,label='EWMA alpha=.10; phase reset')
        if warm:
            axis.axvspan(min(warm)-.5,max(warm)+.5,color='#8e969f',alpha=.15)
            axis.axvline(max(warm)+.5,color='#777777',ls=':',lw=.8)
        if all(v is None for v in values):axis.text(.5,.5,'N/A (no observed updates/ended episodes)',transform=axis.transAxes,ha='center',fontsize=9)
        axis.set(title=title,ylabel=label);axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for axis in axes[1]:axis.set_xlabel('Collected block index (warmup + joint, log order)')
    figure.suptitle('Training collection and optimizer diagnostics — not deterministic evaluation',fontsize=14)
    figure.supxlabel('Returns: real episodes ending in this block (terminal or reference end), possibly spanning chunks/policy updates; noisy random resets and changing curriculum.\nPooled reward is not a return. Missing values remain gaps; losses use every update, not the last. Gray region: observed warmup.',fontsize=9)
    temporary=output/'training-params-curves.tmp.png';figure.savefig(temporary,dpi=180);plt.close(figure)
    temporary.replace(output/'training-params-curves.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--through-block',type=int,required=True)
    parser.add_argument('--output',type=Path,help='Optional figure output directory; default RUN/visuals')
    parser.add_argument('--stage',help='Optional explicit terminal final-joint-NNNN; otherwise only current non-signal terminal is considered')
    args=parser.parse_args();run=args.run.resolve();limit=args.through_block
    logfile=run/('logs/episodes.jsonl' if (run/'evals').exists() else 'logs/blocks.jsonl')
    initial_path=evaluation_folder(run,'initial')/'report.json'
    initial=json.loads(initial_path.read_text())
    multi=initial.get('dataset_mode')=='full40_stream'
    initial_main=initial['paper_main']['normalized_return']
    initial_macro=initial['benchmark']['failure_count']/initial['benchmark']['clip_count'] if multi else initial['train_auxiliary_macro']['macro_normalized_return']
    points=[];sources=[dict(path=str(initial_path),sha256=sha(initial_path))]
    paths=[p/'report.json' for p in evaluation_folders(run)
           if p.name.isdigit() or p.name.startswith('joint-')]
    terminal=json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {}
    terminal_path=None
    if args.stage:
        terminal_path=evaluation_folder(run,args.stage)/'report.json'
    elif terminal.get('reason') and terminal['reason'] not in ('signal','running','failure'):
        terminal_path=evaluation_folder(run,'final')/'report.json'
    if terminal_path and terminal_path.exists():paths.append(terminal_path)
    elif args.stage:raise FileNotFoundError(terminal_path)
    wanted={stage_iteration(p.parent.name,json.loads(p.read_text())) for p in paths}
    blocks={};log_digest=hashlib.sha256();training_points=[];global_block_index=0
    with logfile.open('rb') as stream:
        for line in stream:
            log_digest.update(line)
            row=json.loads(line)
            if row.get('record_type')=='block' or 'world_losses' in row:
                global_block_index+=1
                if row.get('phase')=='warmup' or (row.get('phase')=='joint' and row.get('joint_block',0)<=limit):
                    item=aggregate_training_block(row,global_block_index)
                    item['block_row_sha256']=hashlib.sha256(line).hexdigest();training_points.append(item)
            if row.get('phase')=='joint' and row.get('joint_block') in wanted:
                compact=dict(heldout_world=row['heldout_world'],optimization=row['optimization'])
                blocks[row['joint_block']]=(compact,hashlib.sha256(line).hexdigest())
    seen = {}
    for path in paths:
        n=stage_iteration(path.parent.name,json.loads(path.read_text()))
        is_terminal=path.parent.name=='final' or path.parent.name.startswith('final-')
        if n>limit:continue
        report=json.loads(path.read_text())
        if n not in blocks:raise ValueError(f'Missing matching block {n}')
        row,line_sha=blocks[n]
        if row['optimization']['optimizer_steps']!=report['optimization']['optimizer_steps']:
            raise ValueError(f'Optimizer progress mismatch at {n}')
        groups=row['heldout_world']['groups']
        ratios={key:groups[key]['rmse']/groups[key]['persistence_rmse'] for key in ['root_position_m','joint_position_rad','root_angular_velocity_rad_s']}
        if n in seen:
            previous = seen[n]
            # Serialized schema3 payload hashes can differ due to stage metadata;
            # compare the actual saved model tensors, not only container hashes.
            import torch
            old = json.loads(previous.read_text())
            def load_weights(p, r):
                cp = checkpoint_path(p.parent)
                payload=load_checkpoint(cp)
                return {key:payload[key] for key in ['actor','world','critic','initial_actor']}
            a,b=load_weights(previous,old),load_weights(path,report)
            for key in ['actor','world','critic','initial_actor']:
                if a[key].keys()!=b[key].keys() or not all(torch.equal(a[key][k],b[key][k]) for k in a[key]):
                    raise ValueError(f'Same-block regular/final model tensors differ at {n}')
            for key in ['paper_main','train_auxiliary_macro']:
                if old[key] != report[key]:
                    raise ValueError(f'Same-block regular/final evaluation differs at {n}')
            next(p for p in points if p['joint_block']==n)['terminal']=True
            sources.append(dict(path=str(path),sha256=sha(path),deduplicated_same_block=True,model_tensors_equal=True))
            continue
        seen[n]=path
        points.append(dict(joint_block=n,terminal=is_terminal,failure_fraction=report['benchmark']['failure_count']/report['benchmark']['clip_count'] if multi else None,wm_fixed_dev_ratios=ratios,
            wm_fixed_dev_groups=groups,primary_normalized_return=report['paper_main']['normalized_return'],
            auxiliary_three_start_macro=None if multi else report['train_auxiliary_macro']['macro_normalized_return'],
            primary_steps=report['paper_main']['steps'],primary_completed=report['paper_main']['completed'],
            matching_block_row_sha256=line_sha,eval_report_sha256=sha(path),
            checkpoint_sha256=report['checkpoint_sha256']))
        sources.append(dict(path=str(path),sha256=sha(path)))
    points.sort(key=lambda p:p['joint_block'])
    if not points:raise ValueError('No complete periodic evaluations')
    output=args.output.resolve() if args.output else run/'visuals';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,sharex=True,figsize=(13.8,4.8),layout='constrained')
    x=[p['joint_block'] for p in points]
    for key,label,color in [('root_position_m','Root position','#2374ab'),('joint_position_rad','Joint position','#e07a16'),('root_angular_velocity_rad_s','Root angular velocity','#8058a5')]:
        axes[0].plot(x,[p['wm_fixed_dev_ratios'][key] for p in points],marker='o',markersize=3,label=label,color=color)
    terminals=[p for p in points if p['terminal']]
    if terminals:
        for key,color in [('root_position_m','#2374ab'),('joint_position_rad','#e07a16'),('root_angular_velocity_rad_s','#8058a5')]:
            axes[0].scatter([p['joint_block'] for p in terminals],[p['wm_fixed_dev_ratios'][key] for p in terminals],marker='*',s=110,color=color,zorder=5)
    axes[0].axhline(1,color='#666666',ls='--',lw=1,label='Persistence baseline = 1')
    axes[0].set(title='World model: fixed-validation K1–8 window errors',ylabel='RMSE / matching persistence RMSE (unitless)',xlabel='Joint training block (warmup excluded)')
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    axes[1].plot([0]+x,[initial_main]+[p['primary_normalized_return'] for p in points],marker='o',markersize=3,color='#2374ab',label='All40: macro full-reference return' if multi else 'Primary: full reference from frame 0')
    axes[1].plot([0]+x,[initial_macro]+[p['failure_fraction'] if multi else p['auxiliary_three_start_macro'] for p in points],marker='s',markersize=3,color='#e07a16',label='Failure fraction: failed recordings /40' if multi else 'Local auxiliary: mean resets 0/100/200')
    if terminals:
        for key,color in [('primary_normalized_return','#2374ab'),('auxiliary_three_start_macro','#e07a16')]:
            axes[1].scatter([p['joint_block'] for p in terminals],[p['failure_fraction'] if multi and key=='auxiliary_three_start_macro' else p[key] for p in terminals],marker='*',s=110,color=color,zorder=5,label='Terminal evaluation' if key=='primary_normalized_return' else None)
    axes[1].axhline(initial_main,color='#2374ab',ls=':',lw=1,label='Initial primary')
    axes[1].axhline(initial_macro,color='#e07a16',ls=':',lw=1,label='Initial failure fraction' if multi else 'Initial auxiliary macro')
    axes[1].axhline(0,color='#888888',lw=.7)
    axes[1].set(title='MuJoCo imitation: deterministic saved evaluations',ylabel='Macro normalized return / failure fraction' if multi else 'Undiscounted return / (0.1 × planned steps)',xlabel='Joint training block (warmup excluded)')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle(f'{"Full40 recordings" if multi else "Single-clip training"} | raw evaluation points through joint {max(x)} | no smoothing',fontsize=13)
    fig.savefig(output/'training-curves.png',dpi=180)
    plt.close(fig)
    plot_training_parameters(training_points,output)
    data=dict(training_points=training_points,training_definitions=dict(returns='Only real hard_fall/reference_end episodes that ended in this block; may span chunks and policy updates; macro means over ended episodes, noisy random resets/changing curriculum. Not deterministic eval. Unknown/cap/segment entries excluded and counted.',per_step_reward='Sum of raw returns of ended episodes divided by their summed steps; not a return and not all current-segment rewards.',losses='WM mean of every world_losses value; actor mean of every actor_updates.actor_loss; critic mean of every nested critic_losses value. Each panel has its own linear y axis; negative/zero values retained.',missing='No ended episodes or no optimizer updates -> null/gap, never zero-filled.',smoothing='Raw block means in light blue; EWMA alpha=.10 in bold, reset at phase changes and missing values.',axis='Global collected block index in log order; observed warmup shaded, joint/warm counters retained per point.'),created_utc=datetime.now(timezone.utc).isoformat(),requested_through_block=limit,
              actual_last_periodic_block=max((p['joint_block'] for p in points if not p['terminal']),default=0),actual_last_plotted_block=max(x),terminal_stage=str(terminal_path) if terminals else None,points=points,initial_primary=initial_main,
              initial_auxiliary_macro=None if multi else initial_macro,initial_failure_fraction=initial_macro if multi else None,sources=sources,
              block_log=dict(path=str(logfile),sha256_at_read=log_digest.hexdigest()),
              script_sha256=sha(Path(__file__)),
              dataset_mode='full40_stream' if multi else 'single',definitions=dict(wm=('Fixed distinct episodes from same40 training recordings, not heldout motions; ' if multi else '')+'Fixed initial-policy trajectory K1..8 window-aggregated physical RMSE divided by same-group persistence RMSE; not train loss or endpoint-only',
                               returns='Undiscounted reward sum / (.1 * full remaining reference transitions); negative values preserved',
                               points='Initial plus every completed saved regular evaluation; current non-signal terminal is starred. Historical signal finals excluded by default; explicit --stage is opt-in. Same-block regular/final deduplicated only after saved-model equality and score checks',
                               warmup='Warm blocks excluded from joint x axis; WM begins at the first actual saved regular evaluation',
                               interpolation='Straight lines connect measured raw periodic points; no smoothing or missing-value filling'))
    (output/'training-curves-data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(png=str(output/'training-curves.png'),training_params_png=str(output/'training-params-curves.png'),training_blocks=len(training_points),points=len(points),last=max(x))))


if __name__=='__main__':main()
