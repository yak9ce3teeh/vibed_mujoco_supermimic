"""Full replay timing and explicit rolling-anchor semantics; no physics rollout."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

spec=importlib.util.spec_from_file_location('full_renderer',Path(__file__).resolve().parents[1]/'tools/render_full_evaluation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Increment(torch.nn.Module):
    def forward(self,state,action):
        result=state.clone();result[:,0]+=action[:,0]
        return result


def states(n):
    x=np.zeros((n+1,71));x[:,0]=1000+10*np.arange(n+1);x[:,3]=1
    return x


def test_every_32_true_anchors_without_teacher_forcing_inside_windows():
    actual=states(70);actions=np.ones((70,29));world=Increment()
    pred,anchors,ends,values,invalid=module.rolling_prediction(world,actual,actions)
    assert not invalid
    assert pred[31,0]==1031 and pred[32,0]==1320 and pred[33,0]==1321
    assert anchors[32]==32 and anchors[70]==64
    np.testing.assert_array_equal(ends,[32,64,70])
    np.testing.assert_array_equal(values[:,0],[1032,1352,1646])
    actual[10,0]=999999
    changed=module.rolling_prediction(world,actual,actions)[0]
    np.testing.assert_array_equal(pred,changed)
    pred,anchors,_,_,_=module.rolling_prediction(world,states(64),actions[:64])
    assert anchors[-1]==32 and pred[-1,0]==1352  # No artificial terminal reanchor.


@pytest.mark.parametrize('n',[2,3,32,33,310,5339,13655])
def test_complete_first_last_real_states_and_one_frame_timing_tolerance(n):
    indices=module.render_indices(n)
    assert indices[0]==0 and indices[-1]==n and bool((np.diff(indices)>0).all())
    assert len(indices)==n//2+1
    assert 0<=len(indices)/25-n/50<=.04000000001
    assert bool((np.diff(indices[:-1])==2).all())


def test_original_height_gravity_termination_and_true_reference_end():
    actual=states(4);reference=actual.astype(np.float32)
    actual[-1,2]=.251
    record=dict(dones=np.array([False,False,False,True]),truncated=np.zeros(4,bool),actions=np.zeros((4,29)),remaining_transitions=8)
    result=module.termination(record,actual,reference,dict(failure=True,termination_cause='hard_fall'))
    assert result['height_trigger'] and not result['gravity_z_trigger']
    assert result['cause']=='tracking_termination'
    actual[-1,2]=0;record['dones'][-1]=False;record['truncated'][-1]=True;record['remaining_transitions']=4
    result=module.termination(record,actual,reference,dict(failure=False,termination_cause='reference_end'))
    assert result['cause']=='reference_end'



def test_large_world_coordinates_match_native_float32_visual_exactly():
    import mujoco
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="63.1464191882 0 1"><freejoint/><geom type="sphere" size=".1" pos=".123 .234 .345"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    scene=mujoco.MjvScene(model,maxgeom=1000)
    result=module.append_full_ghost(model,data,scene,[1,.38,.05,.45])
    assert result['float32_rounding_verified_exact_native_visual']
    assert result['max_native_transform_error']['position_m']>1e-6
    assert scene.ngeom==1 and scene.geoms[0].pos[0]>63
