from types import SimpleNamespace
import numpy as np
import torch
from supermimic.training import (
    reference_window,
    trim_replay,
    rng_state,
    restore_rng,
    atomic_save,
    policy_batch,
)
from supermimic.models import Actor


def test_reference_tail_only_zeros_virtual_frame_velocities():
    states = np.ones((50, 71))
    states[:, 0] = np.arange(50)
    result = reference_window(SimpleNamespace(states=states), np.array([48, 49, 50, 51]))
    np.testing.assert_array_equal(result[1], states[49])
    np.testing.assert_array_equal(result[2, :36], states[49, :36])
    assert not result[2, 36:].any()
    assert not result[3, 36:].any()


def test_bounded_replay_keeps_whole_episodes_and_rng_restore(tmp_path):
    replay = [
        dict(episode_id=0, actions=np.zeros((5, 29))),
        dict(episode_id=1, actions=np.zeros((7, 29))),
    ]
    trim_replay(replay, 10)
    assert len(replay) == 1 and replay[0]["episode_id"] == 1
    rng = np.random.default_rng(3)
    state = rng_state(rng)
    a = rng.random(4)
    b = torch.rand(4)
    restore_rng(state, rng)
    np.testing.assert_array_equal(a, rng.random(4))
    torch.testing.assert_close(b, torch.rand(4), rtol=0, atol=0)
    atomic_save({"value": torch.tensor(1)}, tmp_path / "final.pt")
    atomic_save({"value": torch.tensor(2)}, tmp_path / "final.pt")
    assert not (tmp_path / "previous.pt").exists()
    assert torch.load(tmp_path / "final.pt", weights_only=True)["value"] == 2


def test_policy_seed_never_uses_undefined_h32_future():
    states = np.zeros((499, 71))
    states[:, 3] = 1
    states[:, 7] = np.arange(499)
    ep = dict(
        clip="walk",
        frames=np.array([456, 457, 458]),
        states=states[[456, 457, 458]],
        actions=np.zeros((3, 29)),
    )
    actor = Actor(np.zeros(29), 8)
    batch, history, refs = policy_batch(
        [ep], {"walk": SimpleNamespace(states=states)}, actor, np.random.default_rng(8), 100
    )
    assert batch[:, 7].max() <= 457 and refs.shape == (100, 42, 71)
    assert refs[:, -1, 7].max() <= 498


def test_reference_end_is_truncation_and_final_step_fall_is_failure(monkeypatch):
    import os, json, copy
    from pathlib import Path
    import pytest
    from supermimic import training
    from supermimic.motion import load_motions, JOINT_NAMES
    from supermimic.mujoco_adapter import MujocoAdapter

    root = Path(__file__).resolve().parents[1]
    model = Path(
        os.environ.get(
            "SUPERMIMIC_MODEL",
            "/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml",
        )
    )
    data = Path(
        os.environ.get(
            "SUPERMIMIC_MOTIONS", str(root / "dataset/processed/lafan_29dof_10s-clipped.pkl")
        )
    )
    if not model.exists() or not data.exists():
        pytest.skip("real assets unavailable")
    c = json.loads((root / "configs/g1_controller.json").read_text())
    env = MujocoAdapter(model, JOINT_NAMES, c["kp"], c["kd"], c["q_default"], c["action_scale"])
    motion = load_motions(data, ["walk1_subject1_clip0"], env)["walk1_subject1_clip0"]
    motion = copy.copy(motion)
    for key in ["states", "body_pos", "body_quat", "body_vel", "body_omega"]:
        setattr(motion, key, getattr(motion, key)[:2])
    actor = Actor(env.q_default, 8)
    ep = training.rollout(env, motion, 0, actor, "pd", np.random.default_rng(0))
    assert ep["completed"] and not ep["dones"][-1] and ep["truncated"][-1]
    assert ep["termination_cause"] == "reference_end"
    np.testing.assert_allclose(ep["reward_terms"].sum(-1), ep["rewards"], atol=1e-10)
    original = training.reward

    def fall(*args):
        value, done, metrics = original(*args)
        return value, True, metrics

    monkeypatch.setattr(training, "reward", fall)
    ep = training.rollout(env, motion, 0, actor, "pd", np.random.default_rng(0))
    assert not ep["completed"] and ep["dones"][-1] and not ep["truncated"][-1]
    assert (
        training.summarize(ep, 0.02)["completion_fraction"] == 1
        and ep["termination_cause"] == "hard_fall"
    )
