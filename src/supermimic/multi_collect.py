"""Persistent bounded-step GPU collection over complete recordings.

A chunk boundary is neither termination nor reset. Physical resume is explicitly
statistical (reset live lanes), because solver/contact hidden state is not saved.
"""
import time
import numpy as np
import torch
from .task import observation, termination_errors
from .multi_replay import history_at


class ReferencePool:
    def __init__(self,motions,device):
        self.names=list(motions);lengths=[len(motions[name].states) for name in self.names]
        self.offsets=torch.tensor(np.r_[0,np.cumsum(lengths)[:-1]],device=device,dtype=torch.long)
        self.lengths=torch.tensor(lengths,device=device,dtype=torch.long)
        self.states=torch.as_tensor(np.concatenate([motions[name].states for name in self.names]),device=device,dtype=torch.float32)
        self.positions=torch.as_tensor(np.concatenate([motions[name].body_pos for name in self.names]),device=device,dtype=torch.float32)

    def gather(self,records,frames,positions=False):
        last=self.lengths[records]-1
        indices=self.offsets[records]+torch.minimum(frames.clamp_min(0),last)
        if positions:return self.positions[indices]
        state=self.states[indices].clone()
        state[...,36:]=torch.where((frames>last)[...,None],0.,state[...,36:])
        return state


class StreamCollector:
    def __init__(self,adapter,motions,actor,task,curriculum,noisy_observation,next_episode_id=0):
        self.adapter=adapter;self.device=adapter.device;self.batch=adapter.num_envs
        self.actor=actor;self.task=task;self.curriculum=curriculum;self.noisy=noisy_observation
        self.pool=ReferencePool(motions,self.device)
        self.bins=torch.as_tensor(curriculum.bins,device=self.device)
        self.pmf=torch.as_tensor(curriculum.probabilities(False),device=self.device,dtype=torch.float32)
        B=self.batch;D=self.device
        self.records=torch.zeros(B,device=D,dtype=torch.long);self.frames=self.records.clone()
        self.start_frames=self.records.clone();self.start_bins=self.records.clone();self.ids=self.records.clone()
        self.previous=torch.zeros((B,29),device=D);self.history=torch.zeros((B,10,93),device=D);self.clean_history=self.history.clone()
        self.length=torch.zeros(B,device=D,dtype=torch.long);self.discounted=torch.zeros(B,device=D);self.raw_return=torch.zeros(B,device=D)
        self.term_sum=torch.zeros((B,8),device=D);self.metric_sum=torch.zeros((B,4),device=D)
        self.next_episode_id=next_episode_id;self.tails=[None]*B;self.total_collected=0
        self.reset(torch.ones(B,device=D,dtype=torch.bool))

    @torch.no_grad()
    def reset(self,mask):
        count=int(mask.sum())
        if not count:return
        chosen=torch.multinomial(self.pmf,count,replacement=True)
        selected=self.bins[chosen]
        frames=selected[:,1]+(torch.rand(count,device=self.device)*(selected[:,2]-selected[:,1])).long()
        self.records[mask]=selected[:,0];self.frames[mask]=frames;self.start_frames[mask]=frames;self.start_bins[mask]=chosen
        self.ids[mask]=torch.arange(self.next_episode_id,self.next_episode_id+count,device=self.device);self.next_episode_id+=count
        ref=self.pool.gather(self.records,self.frames)
        self.adapter.reset(ref[:,:36],ref[:,36:],mask)
        self.previous[mask]=0;self.length[mask]=0;self.discounted[mask]=0;self.raw_return[mask]=0
        self.term_sum[mask]=0;self.metric_sum[mask]=0;self.history[mask]=0;self.clean_history[mask]=0
        clean=observation(ref[mask],self.previous[mask],self.actor.default)
        self.clean_history[mask,-1]=clean;self.history[mask,-1]=self.noisy(clean)

    @torch.no_grad()
    def collect(self,steps=64):
        started=time.monotonic();context=self.clean_history.cpu().numpy().copy()
        self.pmf=torch.as_tensor(self.curriculum.probabilities(),device=self.device,dtype=torch.float32)
        trace={key:[] for key in ['states','next_states','actions','raw_actions','rewards','dones','truncated','frames','record_ids','episode_ids','observations','reward_terms','metrics']}
        completed=[];names=None
        for _ in range(steps):
            state=self.adapter.state();obs=observation(state,self.previous,self.actor.default)
            future=self.pool.gather(self.records[:,None].expand(-1,10),self.frames[:,None]+torch.arange(1,11,device=self.device))
            raw=self.actor(state,self.history,future)+.1*torch.randn((self.batch,29),device=self.device)
            action=self.actor.execute(raw);successor=self.adapter.step(action)
            target=self.pool.gather(self.records,self.frames+1)
            reward,_,terms=self.task(successor,target,action,self.previous)
            height,tilt=termination_errors(successor,target);hard=(height>.25)|(tilt>.8)
            end=self.frames+1>=self.pool.lengths[self.records]-1;finished=hard|end
            pos=self.task.fk(successor)[0];target_pos=self.pool.gather(self.records,self.frames+1,positions=True)
            metrics=torch.stack((torch.linalg.vector_norm(pos-target_pos,dim=-1).mean(-1),torch.linalg.vector_norm((pos-pos[:,:1])-(target_pos-target_pos[:,:1]),dim=-1).mean(-1),(successor[:,7:36]-target[:,7:36]).square().mean(-1).sqrt(),successor[:,2]),-1)
            term_values=torch.stack(list(terms.values()),-1);names=list(terms)
            values=dict(states=state,next_states=successor,actions=action,raw_actions=raw,rewards=reward,dones=hard,truncated=end&~hard,frames=self.frames,record_ids=self.records,episode_ids=self.ids,observations=obs,reward_terms=term_values,metrics=metrics)
            for key,value in values.items():trace[key].append(value.clone())
            self.discounted+=reward*(.99**self.length);self.raw_return+=reward;self.term_sum+=term_values;self.metric_sum+=metrics;self.length+=1
            if bool(finished.any()):
                record=torch.stack((self.ids[finished].double(),self.records[finished].double(),self.start_frames[finished].double(),self.start_bins[finished].double(),self.length[finished].double(),self.discounted[finished].double(),self.raw_return[finished].double(),hard[finished].double()),-1)
                completed.append((record.cpu().numpy(),self.term_sum[finished].cpu().numpy(),self.metric_sum[finished].cpu().numpy()))
            self.previous.copy_(action);self.frames+=1
            clean=observation(successor,action,self.actor.default)
            self.clean_history=torch.cat((self.clean_history[:,1:],clean[:,None]),1)
            self.history=torch.cat((self.history[:,1:],self.noisy(clean)[:,None]),1)
            self.reset(finished)
        self.adapter.synchronize();arrays={key:torch.stack(values).cpu().numpy() for key,values in trace.items()}
        live_ids=self.ids.cpu().numpy();segments=[]
        for lane in range(self.batch):
            ids=arrays['episode_ids'][:,lane];cuts=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,steps]
            new_tail=None
            for left,right in zip(cuts[:-1],cuts[1:]):
                segment={key:values[left:right,lane].copy() for key,values in arrays.items() if key not in ('episode_ids','record_ids')}
                record_id=int(arrays['record_ids'][left,lane]);episode_id=int(ids[left])
                segment.update(clip=self.pool.names[record_id],record_id=record_id,episode_id=episode_id,prefix=0,reward_term_names=np.asarray(names),segment_boundary=not bool(segment['dones'][-1] or segment['truncated'][-1]))
                segment['history_context']=context[lane].copy() if left==0 else np.concatenate((np.zeros((9,93),dtype=np.float32),segment['observations'][:1]),0)
                tail=self.tails[lane] if left==0 else None
                if tail is not None and tail['episode_id']==episode_id:
                    if tail['clip']!=segment['clip'] or tail['frames'][-1]+1!=segment['frames'][0]:raise AssertionError('Cross-recording/discontinuous chunk boundary')
                    for key in arrays:
                        if key not in ('episode_ids','record_ids'):segment[key]=np.concatenate((tail[key],segment[key]),0)
                    segment['prefix']=len(tail['actions']);segment['history_context']=tail['history_context']
                if not np.array_equal(segment['frames'],segment['frames'][0]+np.arange(len(segment['frames']))):raise AssertionError('Noncontiguous replay segment')
                if episode_id==live_ids[lane] and segment['segment_boundary']:
                    start=max(0,len(segment['actions'])-7)
                    new_tail={key:segment[key][start:].copy() for key in arrays if key not in ('episode_ids','record_ids')}
                    new_tail.update(episode_id=episode_id,clip=segment['clip'],history_context=history_at(segment,start))
                segments.append(segment)
            self.tails[lane]=new_tail
        episodes=[]
        for records,terms,metrics in completed:
            self.curriculum.update(records[:,3].astype(int),records[:,5])
            for r,t,m in zip(records,terms,metrics):
                episode_id,record_id,start,bin_id,length=map(int,r[:5]);name=self.pool.names[record_id]
                expected=int(self.pool.lengths[record_id])-start-1
                episodes.append(dict(episode_id=episode_id,clip=name,start=start,start_bin=bin_id,steps=length,remaining_transitions=expected,discounted_return=float(r[5]),return_sum=float(r[6]),normalized_return=float(r[6])/(.1*expected),completed=not bool(r[7]),termination_cause='hard_fall' if r[7] else 'reference_end',global_mpjpe=float(m[0])/length,joint_rmse=float(m[2])/length,reward_terms={n:dict(sum=float(v),mean=float(v)/length) for n,v in zip(names,t)}))
        n=steps*self.batch;self.total_collected+=n;seconds=time.monotonic()-started
        return segments,episodes,dict(active_transitions=n,control_ticks=steps,aggregate_world_control_ticks=n,active_utilization=1.,collect_seconds=seconds,active_transitions_per_second=n/seconds,completed_episodes=len(episodes),segment_count=len(segments),chunk_boundary_is_not_termination=True,resume_mode='reset_live_lanes_on_resume')

    def live_state(self):
        return dict(next_episode_id=self.next_episode_id,episode_ids=self.ids.cpu().numpy(),record_ids=self.records.cpu().numpy(),frames=self.frames.cpu().numpy(),start_frames=self.start_frames.cpu().numpy(),start_bins=self.start_bins.cpu().numpy(),lengths=self.length.cpu().numpy(),discounted_returns=self.discounted.cpu().numpy(),raw_returns=self.raw_return.cpu().numpy(),previous=self.previous.cpu().numpy(),history=self.history.cpu().numpy(),clean_history=self.clean_history.cpu().numpy(),state71=self.adapter.state().cpu().numpy(),term_sum=self.term_sum.cpu().numpy(),metric_sum=self.metric_sum.cpu().numpy(),resume_mode='reset_live_lanes_on_resume',solver_hidden_state_saved=False)
