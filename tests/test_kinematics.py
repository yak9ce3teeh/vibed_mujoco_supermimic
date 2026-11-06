import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.kinematics import Kinematics
from supermimic.task import TrackingTask, tracking_terms, termination_errors, squared_angle

MODEL = os.environ.get(
    "SUPERMIMIC_MODEL", "/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml"
)


@pytest.mark.parametrize("anchor", [False, True])
def test_native_fk_all_body_fields_and_task(anchor):
    if not Path(MODEL).is_file():
        pytest.skip("real G1 resource unavailable")
    m = mujoco.MjModel.from_xml_path(MODEL)
    if anchor:
        # Controlled probe of nonzero hinge anchor and nonzero joint reference.
        m.jnt_pos[2] = [0.02, -0.03, 0.04]
        m.qpos0[m.jnt_qposadr[2]] = 0.1
    fk = Kinematics(m)
    task = TrackingTask(m)
    rng = np.random.default_rng(5)
    states = []
    native = []
    for _ in range(4):
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        d.qpos[:3] = rng.normal(size=3)
        d.qpos[3:7] = rng.normal(size=4)
        d.qpos[3:7] /= np.linalg.norm(d.qpos[3:7])
        d.qpos[7:] = rng.normal(0, 0.2, 29)
        d.qvel[:] = rng.normal(size=35)
        mujoco.mj_forward(m, d)
        sp = np.empty((m.nbody - 1, 6))
        for b in range(1, m.nbody):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, sp[b - 1], 0)
        states.append(np.r_[d.qpos, d.qvel])
        native.append((d.xpos[1:].copy(), d.xquat[1:].copy(), sp[:, 3:].copy(), sp[:, :3].copy()))
    state = torch.tensor(np.array(states), dtype=torch.float64)
    pred = fk(state)
    target = tuple(torch.tensor(np.stack([x[i] for x in native])) for i in range(4))
    for i in [0, 2, 3]:
        torch.testing.assert_close(pred[i], target[i], atol=1e-7, rtol=0)
    assert squared_angle(pred[1], target[1]).max() < 1e-14
    act = torch.randn(4, 29, dtype=state.dtype)
    prev = torch.randn_like(act)
    refs = state.roll(1, 0)
    actualref = tuple(x.roll(1, 0) for x in target)
    expected = tracking_terms(state, refs, target, actualref, act, prev, task.limits, 0.02)
    r, alive, terms = task(state, refs, act, prev)
    for key in terms:
        torch.testing.assert_close(terms[key], expected[key], atol=1e-5, rtol=0)
    assert torch.isfinite(alive).all()
    # Perfect-match and tiny rotational error gradients must remain finite.
    state.requires_grad_(True)
    reward, alive, _ = task(state, state.detach(), act, act)
    gradient = torch.autograd.grad((reward + alive).sum(), state)[0]
    assert torch.isfinite(gradient).all()


def test_soft_boundary_derivative():
    from supermimic.task import termination_errors

    s = torch.zeros(1, 71, dtype=torch.float64)
    s[:, 3] = 1
    s[:, 2] = 1.0
    ref = s.clone()
    s[:, 2] += 0.25
    s.requires_grad_(True)
    height, tilt = termination_errors(s, ref)
    alive = torch.sigmoid((0.25 - height) / 0.02) * torch.sigmoid((0.8 - tilt) / 0.04)
    g = torch.autograd.grad(alive.sum(), s)[0]
    assert 0.49 < float(alive) < 0.51 and float(g[0, 2]) < -12.4 and torch.isfinite(g).all()
