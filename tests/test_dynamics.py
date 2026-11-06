import torch

from supermimic.dynamics import (
    apply_difference,
    local_difference,
    local_state,
    rotation_exp,
    rotation_log,
)
from supermimic.models import WorldModel
from supermimic.task import rotate, product


def states(n=8):
    s = torch.randn(n, 71, dtype=torch.float64) * 0.2
    s[:, 3:7] = rotation_exp(torch.randn(n, 3, dtype=s.dtype))
    return s


def transform(s, yaw, xy):
    result = s.clone()
    q = rotation_exp(s.new_tensor([0.0, 0.0, yaw])).expand(len(s), 4)
    result[:, :3] = rotate(q, s[:, :3]) + s.new_tensor([*xy, 0.0])
    result[:, 3:7] = product(q, s[:, 3:7])
    result[:, 36:39] = rotate(q, s[:, 36:39])
    return result


def test_local_inverse_quaternion_sign_and_se2_equivariance():
    torch.manual_seed(8)
    s = states()
    n = states()
    d = local_difference(s, n)
    recovered = apply_difference(s, d)
    torch.testing.assert_close(recovered[:, :3], n[:, :3], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(recovered[:, 7:], n[:, 7:], atol=1e-12, rtol=1e-12)
    assert ((recovered[:, 3:7] * n[:, 3:7]).sum(-1).abs() - 1).abs().max() < 1e-12
    flipped = s.clone()
    flipped[:, 3:7] *= -1
    torch.testing.assert_close(local_state(s), local_state(flipped))
    torch.testing.assert_close(local_difference(s, n), local_difference(flipped, n))
    world = WorldModel(16).double()
    action = torch.randn(8, 29, dtype=s.dtype)
    shifted = transform(s, 1.2, [5.0, -3.0])
    torch.testing.assert_close(local_state(shifted), local_state(s), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        world(shifted, action),
        transform(world(s, action), 1.2, [5.0, -3.0]),
        atol=1e-12,
        rtol=1e-12,
    )


def test_rotation_zero_and_small_gradients():
    for size in [0.0, 1e-12, 1e-5]:
        v = torch.full((3,), size, dtype=torch.float64, requires_grad=True)
        x = rotation_log(rotation_exp(v))
        g = torch.autograd.grad(x.sum(), v)[0]
        torch.testing.assert_close(x, v, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g, torch.ones_like(g), atol=1e-10, rtol=1e-10)
