"""Small versioned run-path and packed-trajectory helpers; legacy reads retained."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np

IO_SCHEMA_VERSION = 2
LEGACY_LAYOUT_SOURCE_SHA256 = '0dd3b540d34a41159bc75c2fa3bd4604e38de114d14b08772615ea54e5ad6887'


def stage_key(stage):
    stage = str(stage)
    if stage in ('initial', 'warmup-complete', 'final') or stage.startswith('interrupt-'):
        return stage
    if stage.startswith('final-joint-'):
        return 'final'
    if stage.startswith('joint-'):
        stage = stage[6:]
    if stage.isdigit():
        return f'{int(stage):04d}'
    raise ValueError(f'Unknown evaluation stage {stage}')


def evaluation_folder(run, stage):
    run = Path(run)
    canonical = run / 'evals' / stage_key(stage)
    legacy = run / 'evaluations' / str(stage)
    if canonical.exists() or not legacy.exists():
        return canonical
    return legacy


def evaluation_folders(run):
    run = Path(run)
    folder = run / ('evals' if (run/'evals').exists() else 'evaluations')
    return sorted(p for p in folder.iterdir() if p.is_dir() and (p/'report.json').exists())


def checkpoint_path(folder):
    folder = Path(folder)
    report = json.loads((folder/'report.json').read_text())
    path = Path(report['checkpoint'])
    local=folder.parent.parent/'ckpt'/path.name
    if local.exists():return local
    if path.is_absolute() and path.exists():
        return path
    # Relative checkpoint paths in v2 are relative to the date-named run root.
    run = folder.parent.parent
    for candidate in (run/path, folder/path, folder/'checkpoint.pt'):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def case_id(clip, start, policy):
    return f'{clip}-start{start}-{policy}'


def load_case(folder, identifier):
    folder = Path(folder)
    bundle = folder/'trajectories.npz'
    if bundle.exists():
        prefix = identifier+'/'
        with np.load(bundle, allow_pickle=False) as packed:
            if identifier not in packed['case_ids'].tolist():
                raise KeyError(identifier)
            return {key[len(prefix):]:packed[key].copy() for key in packed.files if key.startswith(prefix)}
    with np.load(folder/(identifier+'.npz'), allow_pickle=False) as saved:
        return {key:saved[key].copy() for key in saved.files}


def content_digest(record):
    h=hashlib.sha256()
    for key,value in sorted(record.items()):
        a=np.asarray(value)
        h.update(json.dumps([key,a.dtype.str,list(a.shape)],separators=(',',':')).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def case_digest(folder, identifier):
    folder=Path(folder);bundle=folder/'trajectories.npz'
    if bundle.exists():
        with np.load(bundle,allow_pickle=False) as saved:
            if 'case_hashes' in saved:
                return str(saved['case_hashes'][saved['case_ids'].tolist().index(identifier)])
        return content_digest(load_case(folder,identifier))
    return hashlib.sha256((folder/(identifier+'.npz')).read_bytes()).hexdigest()


def write_packed(path, cases):
    path=Path(path)
    arrays={'io_schema_version':np.asarray(IO_SCHEMA_VERSION),
            'case_ids':np.asarray(list(cases)),
            'case_hashes':np.asarray([content_digest(record) for record in cases.values()])}
    for identifier,record in cases.items():
        for field,value in record.items():
            array=np.asarray(value)
            if array.dtype.hasobject:raise ValueError('Packed trajectories must not contain pickle/object arrays')
            arrays[identifier+'/'+field]=array
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def load_weights(path, device='cpu'):
    import torch
    # All project checkpoints are trusted local files. Full resume payloads
    # contain NumPy replay/RNG objects and cannot use weights_only=True.
    return torch.load(path,map_location=device,weights_only=False)


def block_records(run):
    run=Path(run);path=run/'logs/episodes.jsonl'
    if (run/'logs/blocks.jsonl').exists():
        path=run/'logs/blocks.jsonl'
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('record_type')=='block' or 'world_losses' in row:
                yield row,line


def stage_iteration(stage, report=None):
    report=report or {}
    if 'joint_block' in report:
        return int(report['joint_block'])
    match=re.search(r'(\d+)$',str(stage))
    if match:
        return int(match.group(1))
    if str(stage)=='final':
        return int(report['optimization']['joint_iterations']//report['config']['actor_updates'])
    return 0


def prediction_entries(folder):
    """Read both compact current and migrated historical visual manifests."""
    folder=Path(folder); raw=json.loads((folder/'manifest.json').read_text())
    found=[]
    def visit(value):
        if isinstance(value,list):
            for item in value: visit(item)
        elif isinstance(value,dict):
            if 'prediction_npz' in value and 'horizon' in value and ('source_npz' in value or 'trajectory_key' in value):
                item=dict(value)
                if 'source_npz' not in item:
                    bundle=Path(item['trajectory_bundle'])
                    item['source_npz']=str(bundle.parent/(item['trajectory_key']+'.npz'))
                    item['source_sha256']=item['trajectory_source_sha256']
                prediction=Path(item['prediction_npz'])
                if not prediction.exists():prediction=folder/'prediction.npz'
                item['prediction_npz']=str(prediction.resolve())
                found.append(item)
            else:
                for item in value.values():visit(item)
    visit(raw)
    return found


def checkpoint_matches(folder, expected_hash):
    path=checkpoint_path(folder)
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    actual=digest.hexdigest()
    if actual==expected_hash:return True
    report=json.loads((Path(folder)/'report.json').read_text())
    return (report.get('evaluated_weights_sha256')==expected_hash or expected_hash in report.get('checkpoint_alias_hashes',[])) and report.get('checkpoint_sha256')==actual


def archive_terminal_before_resume(run, saved):
    """Move the previous terminal snapshot to a stable interrupt-N boundary."""
    import os
    import torch
    run=Path(run);folder=run/'evals/final'
    if not folder.exists():return None
    report=json.loads((folder/'report.json').read_text())
    n=stage_iteration('final',report);key=f'interrupt-{n:04d}'
    if n != saved['run_state']['joint_blocks']:
        raise ValueError('Resume checkpoint iteration differs from existing final evaluation')
    evaluated=load_weights(checkpoint_path(folder))
    if any(not torch.equal(value.cpu(),evaluated[module][name].cpu())
           for module in ('actor','initial_actor','world','critic')
           for name,value in saved[module].items()):
        raise ValueError('Resume weights differ from existing final boundary')
    target=run/'evals'/key
    if target.exists():raise FileExistsError(f'Previous-final archive already exists: {target}')
    ckpt=run/'ckpt'/(key+'.pt')
    compact={k:saved[k] for k in ('schema_version','state_representation','task_semantics','action_semantics',
              'actor','initial_actor','world','critic','config','controller','optimization') if k in saved}
    temporary=ckpt.with_suffix('.pt.tmp');torch.save(compact,temporary);os.replace(temporary,ckpt)
    old_hash=report['checkpoint_sha256'];new_hash=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    report.update(stage=key,joint_block=n,checkpoint=str(ckpt),checkpoint_sha256=new_hash,checkpoint_kind='weights_only',
        checkpoint_alias_hashes=list({old_hash,report.get('evaluated_weights_sha256',old_hash)}),
        terminal_archive_reason='Previous final boundary archived before explicit continuation')
    folder.rename(target)
    (target/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    metadata=dict(stage=key,joint_block=n,total_collected=report.get('total_collected',saved['run_state']['total_collected']),
        path=str(target),checkpoint=str(ckpt),checkpoint_sha256=new_hash,checkpoint_kind='weights_only',
        score=report['train_auxiliary_macro'],paper_main=report['paper_main'],io_schema_version=2)
    ckpt.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    visual=run/'visuals/final'
    if visual.exists():visual.rename(run/'visuals'/key)
    def remap(value):
        if isinstance(value,dict):
            result={k:remap(v) for k,v in value.items()}
            old_checkpoint=result.get('checkpoint')
            if old_checkpoint in (str(run/'ckpt/final.pt'),'ckpt/final.pt'):
                result.update(checkpoint=str(ckpt),checkpoint_sha256=new_hash,checkpoint_kind='weights_only')
                if 'prediction_npz' in result:
                    result['evaluated_weights_sha256']=value.get('checkpoint_sha256')
            if result.get('stage') in ('final',f'final-joint-{n:04d}'):
                result.update(stage=key,path=str(target))
            return result
        if isinstance(value,list):return [remap(v) for v in value]
        if isinstance(value,str):
            return value.replace(str(folder),str(target)).replace(str(run/'visuals/final'),str(run/'visuals'/key))
        return value
    saved['run_state']=remap(saved['run_state'])
    for path in (run/'best.json',run/'logs/eval.jsonl'):
        if not path.exists():continue
        if path.suffix=='.jsonl':
            rows=[remap(json.loads(line)) for line in path.read_text().splitlines()]
            payload=''.join(json.dumps(row)+'\n' for row in rows)
        else:payload=json.dumps(remap(json.loads(path.read_text())),indent=2)+'\n'
        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(payload);temporary.replace(path)
    for path in (run/'visuals'/key).glob('*/manifest.json'):
        path.write_text(json.dumps(remap(json.loads(path.read_text())),indent=2)+'\n')
    return dict(previous_final_joint_block=n,archived_stage=key,checkpoint=str(ckpt),checkpoint_sha256=new_hash)


def learning_kernel_digest():
    """Pin current learning code; legacy equivalence is separately tested against its archived source."""
    import ast
    source=Path(__file__).parent
    names={'cosine_learning_rate','OptimizationProgress','clip_actor_gradients','tensor','future',
           'noisy_observation','rollout','sample','train_world','imagine','policy_batch','train_actor',
           'world_error','outcome_error','directional_check','summarize','reference_window','trim_replay',
           'rng_state','restore_rng','preserve_rng'}
    tree=ast.parse((source/'training.py').read_text())
    nodes=[ast.dump(node,include_attributes=False) for node in tree.body if getattr(node,'name',None) in names]
    h=hashlib.sha256('\n'.join(nodes).encode())
    for name in ('models.py','dynamics.py','task.py','kinematics.py','motion.py','mujoco_adapter.py'):
        h.update(name.encode());h.update((source/name).read_bytes())
    return h.hexdigest()


def validate_layout_bridge(old_source, new_source):
    path=Path(__file__).resolve().parents[2]/'configs/layout_resume_bridge.json'
    if not path.exists():raise ValueError('Audited layout resume bridge is missing')
    bridge=json.loads(path.read_text())
    if (old_source!=bridge['legacy_source_sha256'] or new_source!=bridge['destination_source_sha256']
            or learning_kernel_digest()!=bridge['learning_kernel_sha256']):
        raise ValueError('Layout resume bridge source/destination/kernel digest mismatch')
    return str(path)
