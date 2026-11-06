"""Full-recording boundary/history/curriculum contracts, independent of physics."""
from types import SimpleNamespace
import numpy as np
import torch
from supermimic.curriculum import StartBins
from supermimic.multi_replay import history_at,sample_windows,policy_batch,trim_segments
from supermimic.models import Actor


def motion(length,offset=0):
    states=np.zeros((length,71),dtype=np.float32);states[:,0]=np.arange(length)+offset;states[:,3]=1
    return SimpleNamespace(states=states)


def test_bins_mixture_discounted_returns_tail_and_restore():
    c=StartBins({'a':motion(101),'b':motion(60)})
    np.testing.assert_array_equal(c.bins,[[0,0,50],[0,50,100],[1,0,50],[1,50,59]])
    np.testing.assert_allclose(c.probabilities(False),.25)
    G=sum(.99**i*r for i,r in enumerate([.1,-.2,.3]))
    assert abs(G-.19603)<1e-12
    c.update([0,0,2],[G,.4,-.3]);p=c.probabilities(False)
    means=np.asarray([(G+.4)/2,((G+.4)/2-.3)/2,-.3,((G+.4)/2-.3)/2])
    soft=np.exp(-2.5*means);soft/=soft.sum()
    np.testing.assert_allclose(p,.5/4+.4*soft+.1/4)
    c.probabilities();d=StartBins({'a':motion(101),'b':motion(60)});d.load_state_dict(c.state_dict())
    np.testing.assert_array_equal(c.probabilities(False),d.probabilities(False))


def test_prefix_context_windows_and_policy_references():
    observations=np.repeat(np.arange(100,dtype=np.float32)[:,None],93,axis=1)
    # Context ending at original state10; prefix frames10..16 from prior chunk.
    segment=dict(states=motion(100).states[10:74],next_states=motion(100).states[11:75],actions=np.zeros((64,29),np.float32),frames=np.arange(10,74),observations=observations[10:74],history_context=observations[1:11],prefix=7,clip='a')
    for i in (0,1,7,8,9,31,63):np.testing.assert_array_equal(history_at(segment,i),observations[1+i:11+i])
    rng=np.random.default_rng(7)
    for ep,i in sample_windows([segment],100,8,rng):
        assert ep['frames'][i+7]-ep['frames'][i]==7
    actor=Actor(np.zeros(29),width=8)
    states,hist,refs=policy_batch([segment],{'a':motion(120)},actor,rng,128)
    assert int(states[:,0].min())>=17 and int(refs[:,-1,0].max())<=119
    for state,history,reference in zip(states,hist,refs):
        frame=int(state[0]);np.testing.assert_array_equal(history.numpy(),observations[frame-9:frame+1]);assert reference[0,0]==frame and reference[-1,0]==frame+41
    replay=[dict(segment),dict(segment)];assert trim_segments(replay,60)==57 and len(replay)==1


def test_packed_reference_never_leaks_adjacent_record_and_tail_velocity():
    from supermimic.multi_collect import ReferencePool
    a=motion(5);b=motion(7,1000)
    a.states[:,36:]=3;b.states[:,36:]=7
    for m in (a,b):m.body_pos=np.repeat(m.states[:,:3,None],2,axis=-1).transpose(0,2,1)
    pool=ReferencePool({'a':a,'b':b},'cpu')
    records=torch.tensor([[0,0,0],[1,1,1]]);frames=torch.tensor([[-1,4,99],[0,6,99]])
    ref=pool.gather(records,frames)
    np.testing.assert_array_equal(ref[:,:,0].numpy(),[[0,4,4],[1000,1006,1006]])
    assert bool((ref[0,1,36:]==3).all()) and bool((ref[1,1,36:]==7).all())
    assert bool((ref[:,-1,36:]==0).all())
    assert bool((pool.states[4,36:]==3).all())
