import numpy as np
import torch

from supermimic.render import predict_open_loop, prediction_errors


class AddAction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, state, action):
        self.inputs.append(state.clone())
        output = state.clone()
        output[:, 0] += action[:, 0]
        output[:, 7] += action[:, 1]
        return output


def test_prediction_is_recursive_from_only_initial_state_and_executed_actions():
    world = AddAction()
    initial = np.zeros(71)
    initial[3] = 1
    actions = np.zeros((40, 29))
    actions[:, 0] = 0.1
    actions[:, 1] = 0.2
    prediction = predict_open_loop(world, initial, actions)
    assert prediction.shape == (33, 71) and len(world.inputs) == 32
    np.testing.assert_array_equal(prediction[0], initial)
    np.testing.assert_allclose(prediction[:, 0], np.arange(33) * 0.1, atol=2e-6)
    for step in range(1, 32):
        np.testing.assert_array_equal(world.inputs[step][0].numpy(), prediction[step])
    assert not torch.is_grad_enabled() or all(not state.requires_grad for state in world.inputs)


def test_metric_units_and_quaternion_sign():
    actual = np.zeros((2, 71))
    actual[:, 3] = 1
    prediction = actual.copy()
    prediction[1, 0] = 0.1
    prediction[1, 7:36] = 0.2
    prediction[1, 3] = -1
    error = prediction_errors(prediction, actual)
    np.testing.assert_allclose(error["root_position_cm"], [0, 10])
    np.testing.assert_allclose(error["joint_rmse_rad"], [0, 0.2])
    np.testing.assert_allclose(error["root_rotation_deg"], [0, 0])


def test_overlay_uses_native_robot_geometry_without_changing_model():
    import os
    from pathlib import Path
    import mujoco
    import pytest
    from supermimic.render import append_robot_ghost

    path = Path(
        os.environ.get(
            "SUPERMIMIC_MODEL",
            "/home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml",
        )
    )
    if not path.is_file():
        pytest.skip("real model unavailable")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[7:] = 0.1
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=1000)
    rgba_before = model.geom_rgba.copy()
    qpos_before = data.qpos.copy()
    result = append_robot_ghost(model, data, scene, [1.0, 0.38, 0.05, 0.45])
    assert result["geometry_ids"]
    assert len(set(result["geometry_ids"])) == len(result["geometry_ids"])
    assert all(model.geom_bodyid[i] > 0 for i in result["geometry_ids"])
    assert max(result["max_native_transform_error"].values()) <= 1e-6
    np.testing.assert_array_equal(model.geom_rgba, rgba_before)
    np.testing.assert_array_equal(data.qpos, qpos_before)
    for geom in scene.geoms[: scene.ngeom]:
        np.testing.assert_allclose(geom.rgba, [1.0, 0.38, 0.05, 0.45], atol=1e-7)
        assert geom.transparent
