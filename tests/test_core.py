import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from supermimic.motion import JOINT_NAMES
from supermimic.mujoco_adapter import MujocoAdapter
from supermimic.models import Actor, WorldModel, Critic, observation, goal_features, product
from supermimic.training import imagine, noisy_observation

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get(
    "SUPERMIMIC_MODEL", "/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml"
)


@pytest.fixture
def env():
    if not Path(MODEL).is_file():
        pytest.skip("set SUPERMIMIC_MODEL to a real G1 MuJoCo scene")
    c = json.loads((ROOT / "configs/g1_controller.json").read_text())
    return MujocoAdapter(MODEL, JOINT_NAMES, c["kp"], c["kd"], c["q_default"], c["action_scale"])


def test_clock_mapping_reset_and_action_causality(env):
    a = np.zeros(29)
    a[0] = 0.2
    env.reset()
    x = env.step(a)
    trace = env.torque_trace.copy()
    env.reset()
    y = env.step(a)
    np.testing.assert_array_equal(x["qpos"], y["qpos"])
    assert x["time"] == pytest.approx(0.02)
    assert np.ptp(trace[:, 0]) > 0
    np.testing.assert_array_equal(env.data.ctrl[:6], 0)
    env.reset()
    z = env.step(np.zeros(29))
    assert np.linalg.norm(x["qvel"] - z["qvel"]) > 1e-4
    assert np.all(trace <= env.torque_range[:, 1]) and np.all(trace >= env.torque_range[:, 0])
    assert env.model.actuator_trnid[env.actuator_indices, 0].tolist() == [
        env.model.joint(n).id for n in JOINT_NAMES
    ]


def test_root_angular_world_body_and_forward(env):
    q = env.model.qpos0.copy()
    q[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    v = np.zeros(env.model.nv)
    v[3] = 2
    s = env.reset(q, v)
    np.testing.assert_allclose(s["root_angvel_world"], [0, 2, 0], atol=1e-10)
    np.testing.assert_allclose(s["root_angvel_body"], [2, 0, 0], atol=1e-10)
    np.testing.assert_allclose(env.data.xquat[env.root_body], q[3:7])


def test_history_noise_and_heading_equivariance():
    torch.manual_seed(4)
    s = torch.zeros(2, 71)
    s[:, 3] = 1
    s[:, 2] = 0.8
    r = s[:, None].repeat(1, 10, 1)
    r[:, :, 0] = 0.1
    z = torch.tensor([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=s.dtype).expand(2, 4)
    t = s.clone()
    t[:, 3:7] = product(z, s[:, 3:7])
    t[:, :2] += 3
    rr = r.clone()
    rr[:, :, 0] = 3
    rr[:, :, 1] = 3.1
    rr[:, :, 3:7] = t[:, None, 3:7]
    torch.testing.assert_close(goal_features(s, r), goal_features(t, rr), atol=1e-5, rtol=1e-5)
    obs = observation(s, torch.ones(2, 29), torch.zeros(29))
    noisy = noisy_observation(obs)
    torch.testing.assert_close(noisy[:, -29:], obs[:, -29:])
    assert float((noisy[:, :29] - obs[:, :29]).abs().max()) <= 0.01


def test_h32_gradient_flow_and_frozen_world(env):
    torch.manual_seed(5)
    a = Actor(np.zeros(29), 16).double()
    w = WorldModel(16).double()
    c = Critic(16).double()
    s = torch.zeros(2, 71, dtype=torch.float64)
    s[:, 3] = 1
    s[:, 2] = 0.8
    h = torch.zeros(2, 10, 93, dtype=torch.float64)
    r = s[:, None].repeat(1, 44, 1)
    before = [p.clone() for p in w.parameters()]
    for p in w.parameters():
        p.requires_grad_(False)
    from supermimic.task import TrackingTask

    loss = imagine(a, w, c, s, h, r, task=TrackingTask(env.model))[0]
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in a.parameters()) > 0
    assert all(torch.equal(x, p) for x, p in zip(before, w.parameters()))
    assert all(p.grad is None for p in w.parameters())


def test_real_motion_resampling_and_termination(env):
    from supermimic.motion import load_motions
    from supermimic.training import reward, sample

    data = os.environ.get(
        "SUPERMIMIC_MOTIONS",
        "/home/hm/workspace/skate-bfm/train/dataset/base/lafan_29dof_10s-clipped.pkl",
    )
    if not Path(data).is_file():
        pytest.skip("set SUPERMIMIC_MOTIONS for the real motion test")
    motion = load_motions(data, ["walk1_subject1_clip0"], env)["walk1_subject1_clip0"]
    assert len(motion.states) == 499 and motion.dt == 0.02
    assert np.isfinite(motion.states).all()
    assert np.min(np.sum(motion.states[1:, 3:7] * motion.states[:-1, 3:7], -1)) > 0
    reconstructed = np.zeros(36)
    mujoco.mj_integratePos(
        env.model, reconstructed := motion.states[0, :36].copy(), motion.states[0, 36:], 0.02
    )
    np.testing.assert_allclose(reconstructed, motion.states[1, :36], atol=1e-10)
    env.reset(motion.states[0, :36], motion.states[0, 36:])
    _, done, _ = reward(env, motion, 0, np.zeros(29), np.zeros(29))
    assert not done
    q = motion.states[0, :36].copy()
    q[2] -= 0.26
    env.reset(q, motion.states[0, 36:])
    _, done, _ = reward(env, motion, 0, np.zeros(29), np.zeros(29))
    assert done
    # Sampling returns windows contained in one episode, never concatenated falls.
    eps = [{"actions": np.zeros((n, 29)), "episode_id": n} for n in [3, 8, 12]]
    for ep, i in sample(eps, 100, 8, np.random.default_rng(4)):
        assert 0 <= i and i + 8 <= ep["episode_id"]


def test_large_action_still_respects_motor_limits(env):
    env.reset()
    env.step(np.full(29, 20.0))
    np.testing.assert_array_equal(env.data.ctrl[:6], 0)
    assert np.all(env.torque_trace <= env.torque_range[:, 1]) and np.all(
        env.torque_trace >= env.torque_range[:, 0]
    )
