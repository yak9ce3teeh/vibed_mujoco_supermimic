"""Hidden-shape profiles and legacy schema3 compatibility."""
import copy
import tarfile
from pathlib import Path
import types
import pytest
import torch
from supermimic.models import architecture, build_models


def test_joint_profile_counts_and_legacy_shapes():
    for profile, expected in [('legacy',331396),('paper-0.125',702756),('paper-0.25',1840580),('paper-0.5',5421828),('paper-1',17806724)]:
        modules=build_models(torch.zeros(29),dict(width=128,architecture_profile=profile))
        assert sum(p.numel() for m in modules for p in m.parameters())==expected
    config=dict(architecture_profile='paper-1')
    config['architecture']=architecture(config)
    config['architecture']['world'][0]=123
    with pytest.raises(ValueError):build_models(torch.zeros(29),config)


def test_legacy_from_actual_source_snapshot_equal_forward_and_gradients():
    snapshot=Path(__file__).resolve().parents[1]/'logs/2026-11-05_10-56-46/ckpt/source.tar.gz'
    if not snapshot.exists():pytest.skip('Historical source snapshot not installed')
    with tarfile.open(snapshot) as archive:
        member=next(m for m in archive.getmembers() if m.name.endswith('/models.py') or m.name=='models.py')
        source=archive.extractfile(member).read()
    old=types.ModuleType('supermimic._historical_models');old.__package__='supermimic'
    exec(compile(source,'historical/models.py','exec'),old.__dict__)
    torch.manual_seed(7)
    legacy=[old.Actor(torch.zeros(29)),old.WorldModel(),old.Critic()]
    torch.manual_seed(7)
    current=list(build_models(torch.zeros(29),dict(width=128)))
    for before,after in zip(legacy,current):
        assert before.state_dict().keys()==after.state_dict().keys()
        for key,value in before.state_dict().items():assert torch.equal(value,after.state_dict()[key])
    state=torch.randn(2,71);state[:,3:7]=torch.tensor([1.,0.,0.,0.]);state[:,2]=.8
    history=torch.randn(2,10,93);reference=state[:,None].repeat(1,10,1)
    action=torch.randn(2,29)
    inputs=[(state,history,reference),(state,action),(state,reference[:,0])]
    for before,after,args in zip(legacy,current,inputs):
        x=before(*args);y=after(*args);assert torch.equal(x,y)
        gx=torch.autograd.grad(x.sum(),tuple(before.parameters()))
        gy=torch.autograd.grad(y.sum(),tuple(after.parameters()))
        for a,b in zip(gx,gy):assert torch.equal(a,b)


def test_wave_fifo_matches_incremental_whole_episode_suffix():
    import numpy as np
    from supermimic.training import trim_replay, trim_replay_wave
    rng=np.random.default_rng(7)
    for capacity in (1, 5, 29, 100):
        for _ in range(20):
            lengths=rng.integers(1,capacity+1,size=40)
            incremental=[]
            for i,length in enumerate(lengths):
                incremental.append(dict(actions=np.zeros((length,29)),episode_id=i))
                trim_replay(incremental,capacity)
            wave=[dict(actions=np.zeros((length,29)),episode_id=i) for i,length in enumerate(lengths)]
            trim_replay_wave(wave,capacity)
            assert [e['episode_id'] for e in wave]==[e['episode_id'] for e in incremental]
    for function in (trim_replay,trim_replay_wave):
        oversized=[dict(actions=np.zeros((5,29))),dict(actions=np.zeros((11,29)))]
        with pytest.raises(ValueError):function(oversized,10)
        assert len(oversized)==1 and len(oversized[0]['actions'])==11
