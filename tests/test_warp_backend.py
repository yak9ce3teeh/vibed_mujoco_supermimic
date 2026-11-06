"""Optional actual-CUDA checks; native-only environments skip this module."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
pytest.importorskip('mujoco_warp')
if not torch.cuda.is_available():pytest.skip('Actual CUDA device required for Warp tests',allow_module_level=True)
from supermimic.warp_adapter import MujocoWarpAdapter
from supermimic.mujoco_adapter import MujocoAdapter
from supermimic.motion import JOINT_NAMES,load_motions
from supermimic.task import TrackingTask,reward,termination_errors

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def setup():
    c=json.loads((ROOT/'configs/g1_controller.json').read_text())
    native=MujocoAdapter('/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml',JOINT_NAMES,c['kp'],c['kd'],c['q_default'],c['action_scale'])
    motions=os.environ.get('SUPERMIMIC_MOTIONS',str(ROOT/'dataset/processed/lafan_29dof_10s-clipped.pkl'))
    motion=load_motions(motions,['walk1_subject1_clip0'],native)['walk1_subject1_clip0']
    warp=MujocoWarpAdapter(native,2,record_torques=True)
    return native,warp,motion


def test_actual_cuda_pd_and_nonfinite_action_rejection(setup):
    native,warp,motion=setup
    warp.reset(motion.states[200,:36],motion.states[200,36:])
    before=warp.state().clone();ctrl=warp.ctrl.clone();time=warp.time.clone()
    invalid=torch.zeros((2,29),device='cuda');invalid[0,0]=float('nan')
    with pytest.raises(ValueError):warp.step(invalid)
    assert torch.equal(before,warp.state()) and torch.equal(ctrl,warp.ctrl) and torch.equal(time,warp.time)
    action=torch.zeros((2,29),device='cuda');action[:,20]=.7;action[:,28]=-.4
    q=warp.qpos[:,warp.qpos_indices].clone();v=warp.qvel[:,warp.qvel_indices].clone()
    expected=torch.as_tensor(native.kp,device='cuda')*(warp.q_default+warp.action_scale*action-q)-torch.as_tensor(native.kd,device='cuda')*v
    limits=torch.as_tensor(native.torque_range,device='cuda')
    expected=expected.clamp(limits[:,0],limits[:,1])
    successor=warp.step(action);warp.synchronize()
    assert successor.is_cuda and successor.shape==(2,71)
    torch.testing.assert_close(warp.torque_trace[0].double(),expected,atol=2e-5,rtol=2e-6)
    assert bool((warp.ctrl[:,:6]==0).all()) and bool((warp.overflow==0).all())
    assert bool((warp.torque_trace<=limits[:,1]+1e-5).all()) and bool((warp.torque_trace>=limits[:,0]-1e-5).all())
    assert warp.time[0].item()==pytest.approx(.02,abs=1e-7)
    assert float(successor[0,42+20].abs())>1e-5


def test_same_successor_state_gpu_task_matches_native_reward(setup):
    native,warp,motion=setup
    warp.reset(motion.states[200,:36],motion.states[200,36:])
    task=TrackingTask(native.model).cuda();previous=np.zeros(29,dtype=np.float32)
    maxerror=0.;done_matches=True
    for k in range(32):
        action=((motion.states[201+k,7:36]-native.q_default)/native.action_scale).astype(np.float32)
        action[20]+=.03
        state=warp.step(torch.as_tensor(action,device='cuda')[None].expand(2,-1))
        if k in (0,4,31):
            actual=state[0].cpu().numpy();native.reset(actual[:36],actual[36:])
            r,done,metrics=reward(native,motion,201+k,action,previous)
            ref=torch.as_tensor(motion.states[201+k],device='cuda',dtype=torch.float32)
            with torch.no_grad():gpu_reward,_,terms=task(state[0],ref,torch.as_tensor(action,device='cuda'),torch.as_tensor(previous,device='cuda'))
            for name,value in metrics['reward_terms'].items():maxerror=max(maxerror,abs(float(terms[name])-value))
            height,tilt=termination_errors(state[0],ref);done_matches &= done==bool((height>.25)|(tilt>.8))
            assert abs(float(gpu_reward)-r)<3e-6
        previous=action.copy()
    assert maxerror<2e-6 and done_matches
