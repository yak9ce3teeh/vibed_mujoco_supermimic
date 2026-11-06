"""Summarize a matching joint block and immutable evaluation; no training calls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import re
from supermimic.run_layout import evaluation_folder, stage_key, stage_iteration, block_records, prediction_entries, checkpoint_matches


def digest(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    if not values:
        raise ValueError('Missing update losses')
    return dict(count=len(values), mean=statistics.mean(values), min=min(values), max=max(values))


def summarize(run, block=None, stage=None):
    stage = stage or f'joint-{block:04d}'
    folder = evaluation_folder(run,stage)
    report_path=folder/'report.json'
    report=json.loads(report_path.read_text())
    n=stage_iteration(stage,report)
    if block is not None and block != n:raise ValueError('Block/stage mismatch')
    block=n
    source=run/'logs/episodes.jsonl' if (run/'evals').exists() else run/'logs/blocks.jsonl'
    matches = []
    for row,line in block_records(run):
        if row.get('phase')=='joint' and row.get('joint_block')==block:
            if row['optimization']['optimizer_steps']==report['optimization']['optimizer_steps']:
                matches.append((row,line))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one matching block {block}; found {len(matches)}')
    row, line = matches[0]
    losses = dict(world=stats(row['world_losses']),
                  actor=stats([x['actor_loss'] for x in row['actor_updates']]),
                  critic=stats([v for x in row['actor_updates'] for v in x['critic_losses']]))
    expected = report['config']
    if (losses['world']['count'] != expected['world_updates'] or
            losses['actor']['count'] != expected['actor_updates'] or
            losses['critic']['count'] != expected['actor_updates'] * 20):
        raise ValueError('Unexpected optimizer-loss counts; refusing misleading aggregation')
    multi=report.get('dataset_mode')=='full40_stream'
    train_clip = expected.get('monitor_clip',expected.get('train_clip','walk1_subject1_clip0'))
    trained = [x for x in report['evaluation'] if x['clip'] == train_clip and x['policy'] == 'trained']
    actual = {str(x['start']): x for x in trained}
    for item in report["evaluation"] if multi else actual.values():
        if abs(sum(x['sum'] for x in item['reward_terms'].values()) - item['return_sum']) > 1e-8:
            raise ValueError('Reward term sum differs from raw return')
        if abs(item['return_sum'] / (.1 * item['remaining_transitions']) - item['normalized_return']) > 1e-10:
            raise ValueError('Full-reference denominator mismatch')
    h32={};video_sources={}
    for start in (0,200):
        visual=run/'visuals'/stage_key(stage)/f'start{start:04d}'
        if not (visual/'manifest.json').exists():continue
        entries=prediction_entries(visual)
        entry=next((e for e in entries if Path(e['source_npz']).stem==f'{train_clip}-start{start}-trained'),None)
        if entry is None:continue
        if not checkpoint_matches(folder,entry['checkpoint_sha256']):
            raise ValueError('Prediction and evaluation checkpoint differ')
        endpoint=str(entry['horizon'])
        h32[str(start)]=dict(horizon=entry['horizon'],physical_seconds=entry['horizon']*.02,
            endpoint_errors=entry['errors_at_steps'][endpoint],definition='One actual t0 anchor, same executed actions; open-loop endpoint, not train loss or K1..8 aggregate')
        video_sources[str(start)]=dict(manifest=str(visual/'manifest.json'),sha256=digest(visual/'manifest.json'))
    summary = dict(schema_version=1, stage=stage, joint_block=block,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        provenance=dict(block_log=str(source), matching_row_sha256=hashlib.sha256(line.encode()).hexdigest(),
                        eval_report=str(report_path), eval_report_sha256=digest(report_path),
                        checkpoint=report['checkpoint'], checkpoint_sha256=report['checkpoint_sha256'],
                        tool_sha256=digest(Path(__file__))),
        losses=losses,
        loss_definitions=dict(world='Mean of every WM optimizer update; each loss is batch/8-step mean normalized local-state MSE',
                              actor='Mean of every actor update negative imagined return objective',
                              critic='Mean of all 20 critic-step GAE target MSEs per actor update'),
        fixed_dev_world=dict(row['heldout_world'], definition=('Same40 training recordings, fixed initial-policy distinct validation episodes; not heldout motions; ' if multi else 'Fixed initial-policy dev episodes; ')+ ' every same-episode K1..8 unroll state, window-aggregated per-component RMSE; not k8 endpoint; mixed-unit scalar is diagnostic only'),
        fixed_dev_outcomes=dict(**row['heldout_outcomes'], definition='One-step predicted successor through analytic reward/soft-done; fixed train-mean reward baseline; not a learned reward-head loss'),
        **(dict(paper_all40=report['benchmark'],representative_start0=actual['0']) if multi else dict(paper_main_start0=actual['0'],train_three_reset_macro=report['train_auxiliary_macro'])),
        auxiliary_start200=actual.get('200'),
        protocol=dict(normalized_return='undiscounted sum / (0.1 * full planned remaining reference transitions); may be negative',
                      reward_terms='Per-step terms already weighted and multiplied by dt=.02; cumulative sum equals raw return',
                      auxiliary='Representative walk reset200 is excluded from40-recording primary macro' if multi else 'start100/start200 reset evaluations and macro are local diagnostics, not primary first-frame score',
                      diagnostics='MPJPE/joint RMSE are local diagnostics, not claimed paper primary metrics'),
        h32_open_loop=h32, video_sources=video_sources,
        total_collected=row['total_collected'], elapsed_seconds=row['elapsed_seconds'],
        optimization=row['optimization'])
    lines = [f'{stage} | collected {row["total_collected"]} | elapsed {row["elapsed_seconds"]:.1f}s',
             f'Loss means: WM {losses["world"]["mean"]:.6f} ({losses["world"]["count"]} updates); actor {losses["actor"]["mean"]:.6f} ({losses["actor"]["count"]}); critic MSE {losses["critic"]["mean"]:.6f} ({losses["critic"]["count"]} steps)',
             ('WM physical validation = same40-recording fixed episodes, K1..8 aggregate; NOT heldout motions/step8 endpoint:' if multi else 'WM physical validation = fixed-dev K1..8 window aggregate, NOT step8 endpoint:')]
    for name, value in row['heldout_world']['groups'].items():
        lines.append(f'  {name}: RMSE {value["rmse"]:.6f} vs persistence {value["persistence_rmse"]:.6f}')
    o = row['heldout_outcomes']
    lines.append(f'One-step analytic reward RMSE {o["reward_rmse"]:.6f} vs fixed train-mean {o["reward_constant_mean_rmse"]:.6f}; correlation {o["reward_correlation"]:.6f}; soft-done Brier {o["terminal_brier"]:.6f}')
    for start in ['0', '200']:
        if start not in actual: continue
        x = actual[start]
        lines.append(f'{"PRIMARY first-frame" if start == "0" else "AUX reset reference t=4.00s"}: {x["steps"]}/{x["remaining_transitions"]} steps, {x["duration_seconds"]:.2f}s, {x["termination_cause"]}, completed={x["completed"]}; raw return {x["return_sum"]:.6f}, full-reference normalized {x["normalized_return"]:.6f}; local global-MPJPE {100*x["global_mpjpe"]:.3f}cm, jointRMSE {x["joint_rmse"]:.6f}rad')
        lines.append('  Reward term cumulative sums (weights and dt ALREADY included): ' + '; '.join(f'{name}={value["sum"]:.6f}' for name,value in x['reward_terms'].items()))
        if start in h32:
            e = h32[start]['endpoint_errors']
            lines.append(f'  H{h32[start]["horizon"]} same-action open-loop endpoint: root {e["root_position_cm"]:.3f}cm, joint {e["joint_rmse_rad"]:.6f}rad, root rotation {e["root_rotation_deg"]:.3f}deg; separate from WM train loss/K1..8 validation')
    macro = report['train_auxiliary_macro']
    if multi:
        lines.append(f'ALL40 PRIMARY: completed {macro["completed_count"]}/40, failures {macro["failure_count"]}/40, macro full-reference normalized return {macro["macro_normalized_return"]:.6f}, local MPJPE {100*macro["macro_global_mpjpe"]:.3f}cm; all40 are training recordings')
    else:
        lines.append(f'Local three-reset macro: completed {macro["completed_count"]}/3, normalized return {macro["macro_normalized_return"]:.6f}, MPJPE {100*macro["macro_global_mpjpe"]:.3f}cm')
    text = '\n'.join(lines) + '\n'
    print(text, end='')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--block',type=int)
    parser.add_argument('--stage',help='Explicit immutable joint-NNNN or final-joint-NNNN snapshot')
    args=parser.parse_args()
    if args.block is None and args.stage is None: parser.error('--block or --stage is required')
    summarize(args.run.resolve(),args.block,args.stage)
