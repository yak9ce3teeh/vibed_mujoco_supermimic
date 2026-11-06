import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch

from supermimic.models import Actor


def test_decoder_is_linear_and_execution_limit_is_shared():
    actor = Actor(np.zeros(29), 16, action_limit=20.0)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.decoder[-1].bias.fill_(30.0)
    state = torch.zeros(1, 71)
    state[:, 3] = 1
    raw = actor(state, torch.zeros(1, 10, 93), state[:, None].repeat(1, 10, 1))
    assert float(raw.min()) == 30.0
    np.testing.assert_array_equal(
        actor.execute(raw.detach().numpy()), actor.execute(raw).detach().numpy()
    )
    assert float(actor.execute(raw).max()) == 20.0
    for limit in [0.0, -1.0, float("inf"), float("nan")]:
        try:
            Actor(np.zeros(29), 16, action_limit=limit)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid action bound accepted")


def test_original_reference_action_coverage():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/g1_controller.json").read_text())
    path = os.environ.get(
        "SUPERMIMIC_MOTIONS",
        "/home/hm/workspace/skate-bfm/train/dataset/base/lafan_29dof_10s-clipped.pkl",
    )
    if not Path(path).is_file():
        import pytest

        pytest.skip("real motion resource unavailable")
    data = joblib.load(path)
    for name, expected in [("walk1_subject1_clip0", 0.2676), ("walk1_subject2_clip0", 0.2217)]:
        actions = (data[name]["dof"] - np.asarray(config["q_default"])) / np.asarray(
            config["action_scale"]
        )
        assert abs(np.mean(np.abs(actions) > 1) - expected) < 0.0001
        assert not np.any(np.abs(actions) > 20)
