"""Synchronous CUDA episode waves with GPU actor/history/task and executed replay.

All lanes start independent episodes together. A completed lane is inactive until
next wave: its later physics ticks are never replay transitions. No cross-episode
windows and no claim of asynchronous episode scheduling.
"""
import time
import numpy as np
import torch
from .motion import HISTORY,OBS_DIM,FUTURE
from .task import observation,termination_errors


@torch.no_grad()
def collect_warp(adapter,motion,actor,rng,episode_count,task,noisy_observation,*,noise=.1,starts_override=None):
    batch=adapter.num_envs
    if episode_count<batch or episode_count%batch:
        raise ValueError('Warp --episodes must be a positive multiple of --num-envs; all initial lanes active')
    if noise<0:raise ValueError("Noise scale must be nonnegative")
    if starts_override is not None and np.asarray(starts_override).shape!=(episode_count,):
        raise ValueError("One fixed reset frame per requested episode required")
    device=adapter.device
    reference=torch.as_tensor(motion.states,device=device,dtype=torch.float32)
    target_body_positions=torch.as_tensor(motion.body_pos,device=device,dtype=torch.float32)
    last=len(reference)-1
    episodes=[];control_ticks=0;started=time.monotonic()
    for wave in range(episode_count//batch):
        starts=(rng.integers(0,last-32-10+1,size=batch) if starts_override is None else np.asarray(starts_override)[wave*batch:(wave+1)*batch])
        if np.any(starts<0) or np.any(starts>last-42):raise ValueError("Reset frames require H32+future10 remaining")
        frames=torch.as_tensor(starts,device=device,dtype=torch.long)
        state=adapter.reset(reference[frames,:36],reference[frames,36:])
        previous=torch.zeros((batch,29),device=device)
        history=torch.zeros((batch,HISTORY,OBS_DIM),device=device)
        initial_obs=observation(state,previous,actor.default)
        history[:,-1]=noisy_observation(initial_obs) if noise else initial_obs
        active=torch.ones(batch,device=device,dtype=torch.bool)
        trace={key:[] for key in ['states','next_states','actions','raw_actions','rewards','dones','frames','reward_terms','active','metrics']}
        term_names=None
        while bool(active.any()):
            indices=frames[:,None]+torch.arange(1,FUTURE+1,device=device)
            goals=reference[indices.clamp_max(last)].clone()
            goals[...,36:]=torch.where((indices>last)[...,None],0.,goals[...,36:])
            raw=actor(state,history,goals)
            if noise:raw=raw+noise*torch.randn((batch,29),device=device)
            executed=actor.execute(raw)
            actual_action=torch.where(active[:,None],executed,torch.zeros_like(executed))
            successor=adapter.step(actual_action)
            control_ticks+=1
            target=reference[(frames+1).clamp_max(last)]
            reward,_,terms=task(successor,target,executed,previous)
            height,tilt=termination_errors(successor,target)
            hard=(height>.25)|(tilt>.8)
            finished=hard|(frames+1>=last)
            pos=task.fk(successor)[0]
            target_pos=target_body_positions[(frames+1).clamp_max(last)]
            metrics=torch.stack([
                torch.linalg.vector_norm(pos-target_pos,dim=-1).mean(-1),
                torch.linalg.vector_norm((pos-pos[:,:1])-(target_pos-target_pos[:,:1]),dim=-1).mean(-1),
                (successor[:,7:36]-target[:,7:36]).square().mean(-1).sqrt(),successor[:,2]],dim=-1)
            if term_names is None:term_names=list(terms)
            values=dict(states=state,next_states=successor,actions=executed,raw_actions=raw,
                        rewards=reward,dones=hard,frames=frames,reward_terms=torch.stack(list(terms.values()),-1),
                        active=active,metrics=metrics)
            for key,value in values.items():trace[key].append(value.clone())
            next_active=active&~finished
            obs=observation(successor,executed,actor.default)
            if noise:obs=noisy_observation(obs)
            next_history=torch.cat([history[:,1:],obs[:,None]],dim=1)
            history=torch.where(next_active[:,None,None],next_history,history)
            previous=torch.where(next_active[:,None],executed,previous)
            frames=frames+active.long();active=next_active;state=successor
        adapter.synchronize();adapter.check()
        arrays={key:torch.stack(values).cpu().numpy() for key,values in trace.items()}
        metric_names=['global_mpjpe','root_relative_mpjpe','joint_rmse','root_height']
        for lane in range(batch):
            length=int(arrays['active'][:,lane].sum())
            ep={key:array[:length,lane].copy() for key,array in arrays.items() if key not in ('active','metrics')}
            hard=bool(ep['dones'][-1]);completed=not hard and int(ep['frames'][-1])+1==last
            truncated=np.zeros(length,dtype=bool);truncated[-1]=not hard
            ep.update(reward_term_names=np.asarray(term_names),truncated=truncated,
                termination_cause='hard_fall' if hard else 'reference_end',completed=completed,
                remaining_transitions=last-int(starts[lane]),action_limit=actor.action_limit,
                metrics=[dict(zip(metric_names,map(float,values))) for values in arrays['metrics'][:length,lane]],
                clip=motion.name,start=int(starts[lane]),mode='actor',
                physics_backend='warp',collection_lane=lane,collection_wave=wave)
            if not np.allclose(ep['reward_terms'].sum(-1),ep['rewards'],atol=2e-6):
                raise AssertionError('GPU actual reward terms do not sum')
            episodes.append(ep)
    active_transitions=sum(len(ep['actions']) for ep in episodes)
    seconds=time.monotonic()-started
    metadata=dict(backend='warp',num_envs=batch,episode_count=len(episodes),
        scheduling='synchronous whole-episode waves; finished lanes inactive until next wave',
        control_ticks=control_ticks,aggregate_world_control_ticks=batch*control_ticks,
        active_transitions=active_transitions,active_utilization=active_transitions/(batch*control_ticks),
        collect_seconds=seconds,active_transitions_per_second=active_transitions/seconds,
        aggregate_world_control_steps_per_second=batch*control_ticks/seconds,
        action_noise_rng='torch CUDA' if noise else 'disabled',action_noise_sigma=noise,observation_noise=bool(noise),history_and_previous_reset='zero per episode; independent per lane',
        terminal_transition_saved_before_reset=True,canonical_eval_backend='native')
    return episodes,metadata
