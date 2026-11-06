import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from supermimic.run_layout import (write_packed,load_case,case_digest,checkpoint_path,
    archive_terminal_before_resume,validate_layout_bridge,LEGACY_LAYOUT_SOURCE_SHA256)


def test_pack_is_exact_pickle_free_and_keeps_case_hashes(tmp_path):
    cases={'clip-start0-trained':{'states':np.arange(142,dtype=np.float64).reshape(2,71),
           'dones':np.array([False,True]),'names':np.array(['weighted','dt'])}}
    write_packed(tmp_path/'trajectories.npz',cases)
    got=load_case(tmp_path,'clip-start0-trained')
    for key in got:np.testing.assert_array_equal(got[key],cases['clip-start0-trained'][key])
    with np.load(tmp_path/'trajectories.npz',allow_pickle=False) as data:
        assert data['case_hashes'][0]==case_digest(tmp_path,'clip-start0-trained')
    with pytest.raises(ValueError):write_packed(tmp_path/'bad.npz',{'bad':{'x':np.array([{}],dtype=object)}})


def test_resume_archives_exact_final_weights_and_all_references(tmp_path):
    folder=tmp_path/'evals/final';folder.mkdir(parents=True)
    (tmp_path/'ckpt').mkdir();(tmp_path/'logs').mkdir()
    saved={k:{'weight':torch.tensor([1.])} for k in ('actor','initial_actor','world','critic')}
    row=dict(stage='final',joint_block=1,path=str(folder),checkpoint=str(tmp_path/'ckpt/final.pt'),checkpoint_kind='full_resume')
    saved.update(run_state=dict(joint_blocks=1,total_collected=9,eval_history=[copy.deepcopy(row)],best_trained=copy.deepcopy(row)),
                 schema_version=3,config={},optimization={})
    torch.save(saved,tmp_path/'ckpt/final.pt')
    original_hash=hashlib.sha256((tmp_path/'ckpt/final.pt').read_bytes()).hexdigest()
    report=dict(stage='final',joint_block=1,checkpoint=row['checkpoint'],checkpoint_sha256=original_hash,
                train_auxiliary_macro={'x':.1},paper_main={'normalized_return':.1})
    (folder/'report.json').write_text(json.dumps(report));(folder/'trajectories.npz').write_bytes(b'exact-pack')
    (tmp_path/'logs/eval.jsonl').write_text(json.dumps(row)+'\n')
    visual=tmp_path/'visuals/final/start0000';visual.mkdir(parents=True)
    (visual/'manifest.json').write_text(json.dumps({'prediction_sources':[[dict(checkpoint=row['checkpoint'],checkpoint_sha256=original_hash,prediction_npz=str(visual/'prediction.npz'))]]}))
    wrong=copy.deepcopy(saved);wrong['run_state']['joint_blocks']=2
    with pytest.raises(ValueError):archive_terminal_before_resume(tmp_path,wrong)
    assert folder.exists()
    result=archive_terminal_before_resume(tmp_path,saved)
    expected=tmp_path/'ckpt/interrupt-0001.pt'
    assert expected.exists() and not folder.exists()
    assert checkpoint_path(tmp_path/'evals/interrupt-0001')==expected
    assert (tmp_path/'evals/interrupt-0001/trajectories.npz').read_bytes()==b'exact-pack'
    for item in [json.loads((tmp_path/'logs/eval.jsonl').read_text()),saved['run_state']['eval_history'][0],saved['run_state']['best_trained']]:
        assert item['checkpoint']==str(expected) and item['checkpoint_kind']=='weights_only'
        assert item['checkpoint_sha256']==result['checkpoint_sha256']
    visual_entry=json.loads((tmp_path/'visuals/interrupt-0001/start0000/manifest.json').read_text())['prediction_sources'][0][0]
    assert visual_entry['checkpoint']==str(expected) and visual_entry['checkpoint_sha256']==result['checkpoint_sha256']
    assert 'replay' not in torch.load(expected,weights_only=True)


def test_layout_bridge_pins_exact_destination_and_kernel():
    source=Path(__file__).resolve().parents[1]/'src/supermimic'
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(source.glob('*.py')))).hexdigest()
    assert validate_layout_bridge(LEGACY_LAYOUT_SOURCE_SHA256,digest)
    with pytest.raises(ValueError):validate_layout_bridge(LEGACY_LAYOUT_SOURCE_SHA256,'0'*64)
    with pytest.raises(ValueError):validate_layout_bridge('1'*64,digest)
