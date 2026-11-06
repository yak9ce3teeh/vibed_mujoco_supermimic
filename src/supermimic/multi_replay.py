"""Bounded segments with explicit clean-history context and seven-step WM overlap."""
import numpy as np
import torch


def history_at(segment, index):
    if index >= 9:
        return segment['observations'][index-9:index+1].copy()
    context=segment['history_context']
    return np.concatenate((context,segment['observations'][1:index+1]),axis=0)[-10:].copy()


def sample_windows(replay,batch,length,rng):
    counts=np.asarray([max(0,len(e['actions'])-length+1) for e in replay],dtype=np.int64)
    total=int(counts.sum())
    if not total:raise RuntimeError('No same-episode K8 windows in bounded replay')
    cumulative=counts.cumsum();choices=rng.integers(total,size=batch);samples=[]
    for value in choices:
        k=int(np.searchsorted(cumulative,value,side='right'))
        i=int(value-(cumulative[k-1] if k else 0))
        samples.append((replay[k],i))
    return samples


def policy_batch(replay,motions,actor,rng,batch):
    eligible=[];counts=[]
    for segment in replay:
        indices=np.flatnonzero((np.arange(len(segment['actions']))>=segment.get('prefix',0)) & (segment['frames']+41<=len(motions[segment['clip']].states)-1))
        if len(indices):eligible.append((segment,indices));counts.append(len(indices))
    if not counts:raise RuntimeError('No H32+future10 legal seed; collect additional real chunks')
    cumulative=np.cumsum(counts);states=[];histories=[];refs=[]
    for value in rng.integers(cumulative[-1],size=batch):
        k=int(np.searchsorted(cumulative,value,side='right'));segment,indices=eligible[k]
        i=int(indices[value-(cumulative[k-1] if k else 0)])
        states.append(segment['states'][i]);histories.append(history_at(segment,i))
        refs.append(motions[segment['clip']].states[segment['frames'][i]+np.arange(42)])
    device=next(actor.parameters()).device
    return tuple(torch.as_tensor(np.stack(values),device=device,dtype=torch.float32) for values in (states,histories,refs))


def trim_segments(replay,capacity):
    total=sum(len(e['actions'])-e.get('prefix',0) for e in replay);removed=0
    while removed<len(replay) and total>capacity:
        total-=len(replay[removed]['actions'])-replay[removed].get('prefix',0);removed+=1
    if removed:del replay[:removed]
    return total
