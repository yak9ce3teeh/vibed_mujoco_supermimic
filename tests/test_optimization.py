import numpy as np
import pytest
import torch

from supermimic.training import (
    OptimizationProgress,
    clip_actor_gradients,
    cosine_learning_rate,
    train_world,
)


class SpyAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=2e-4)
        self.used_lr = []

    def step(self, closure=None):
        self.used_lr.append(self.param_groups[0]["lr"])
        return super().step(closure)


class TinyWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, state, action):
        return state + self.weight * action

    def error(self, prediction, target):
        return (prediction - target).square().mean()


def test_norm_then_elementwise_clipping_is_distinguishable():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([100.0, 1.0])
    clip_actor_gradients([parameter])
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([0.1, 1 / np.sqrt(10001)], dtype=parameter.dtype),
        rtol=1e-6,
        atol=1e-8,
    )
    assert parameter.grad[1] < 0.011  # Reversed order would produce [.1,.1].


@pytest.mark.parametrize(
    "iteration,expected", [(0, 2e-4), (125000, 1.03e-4), (250000, 6e-6), (300000, 6e-6)]
)
def test_cosine_boundaries(iteration, expected):
    assert cosine_learning_rate(iteration) == pytest.approx(expected, rel=1e-12)


def test_warmup_and_two_block_counts_use_joint_actor_iterations():
    world = TinyWorld()
    world_opt = SpyAdam(world.parameters())
    actor_opt = SpyAdam([torch.nn.Parameter(torch.zeros(()))])
    critic_opt = SpyAdam([torch.nn.Parameter(torch.zeros(()))])
    progress = OptimizationProgress(world_updates=20, actor_updates=2)
    replay = [dict(states=np.zeros((9, 1)), actions=np.ones((9, 1)), next_states=np.ones((9, 1)))]
    rng = np.random.default_rng(4)
    train_world(world, world_opt, replay, {}, rng, 20, 2, progress=progress, warmup=True)
    audit = progress.audit()
    assert audit["joint_iterations"] == 0
    assert audit["optimizer_steps"] == dict(world=20, actor=0, critic=0)
    assert audit["last_used_lr"] == dict(world=2e-4, actor=None, critic=None)
    assert audit["last_used_iteration"] == dict(world=None, actor=None, critic=None)
    for _ in range(2):
        train_world(world, world_opt, replay, {}, rng, 20, 2, progress=progress)
        for _ in range(2):
            iteration = progress.joint_iterations
            progress.step(actor_opt, "actor", iteration)
            for _ in range(20):
                progress.step(critic_opt, "critic", iteration)
            progress.joint_iterations += 1
    audit = progress.audit()
    assert audit["joint_iterations"] == 4
    assert audit["optimizer_steps"] == dict(world=60, actor=4, critic=80)
    assert audit["warmup_world_optimizer_steps"] == 20
    np.testing.assert_allclose(
        world_opt.used_lr[20:],
        [cosine_learning_rate(n) for n in range(4) for _ in range(10)],
        rtol=1e-12,
        atol=0,
    )
    assert audit["last_used_lr"] == dict.fromkeys(
        ["world", "actor", "critic"], cosine_learning_rate(3)
    )
    assert audit["next_joint_lr"] == cosine_learning_rate(4)
    # Snapshots must not mutate when future updates change progress.
    progress.steps["actor"] += 1
    assert audit["optimizer_steps"]["actor"] == 4


@pytest.mark.parametrize("world,actor", [(0, 1), (1, 0), (-1, 2)])
def test_positive_update_counts_required(world, actor):
    with pytest.raises(ValueError):
        OptimizationProgress(world, actor)


def test_nondivisible_world_actor_ratio_uses_floor_mapping():
    world = TinyWorld()
    optimizer = SpyAdam(world.parameters())
    progress = OptimizationProgress(3, 2)
    replay = [dict(states=np.zeros((9, 1)), actions=np.ones((9, 1)), next_states=np.ones((9, 1)))]
    train_world(world, optimizer, replay, {}, np.random.default_rng(2), 3, 2, progress=progress)
    np.testing.assert_allclose(
        optimizer.used_lr, [cosine_learning_rate(n) for n in [0, 0, 1]], rtol=1e-12, atol=0
    )
    assert progress.joint_iterations == 0  # Only actor completion advances this clock.
