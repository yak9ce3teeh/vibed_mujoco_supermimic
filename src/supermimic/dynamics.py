"""SE(2)-equivariant dynamics coordinates; public state retains SI/WXYZ layout."""

import math

import torch
from torch.nn import functional as F

from .task import conjugate, heading, product, rotate

DELTA_DIM = 70


def rotation_exp(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return torch.cat(((angle * 0.5).cos(), 0.5 * torch.sinc(angle / (2 * math.pi)) * vector), -1)


def rotation_log(quaternion):
    q = F.normalize(quaternion, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    length = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, q[..., :1])
    ratio = torch.where(length > 1e-6, angle / length.clamp_min(1e-12), 2 + length.square() / 3)
    return q[..., 1:] * ratio


def local_state(state):
    q = state[..., 3:7]
    local_q = product(heading(q), q)
    x = rotate(local_q, torch.zeros_like(state[..., :3]) + state.new_tensor([1.0, 0.0, 0.0]))
    z = rotate(local_q, torch.zeros_like(state[..., :3]) + state.new_tensor([0.0, 0.0, 1.0]))
    return torch.cat(
        (
            state[..., 2:3],
            x,
            z,
            rotate(conjugate(q), state[..., 36:39]),
            state[..., 39:42],
            state[..., 7:36],
            state[..., 42:71],
        ),
        -1,
    )


def local_difference(state, successor):
    q, next_q = state[..., 3:7], successor[..., 3:7]
    inv = conjugate(q)
    angular_difference_world = rotate(next_q, successor[..., 39:42]) - rotate(q, state[..., 39:42])
    return torch.cat(
        (
            rotate(heading(q), successor[..., :3] - state[..., :3]),
            rotation_log(product(inv, next_q)),
            rotate(inv, successor[..., 36:39] - state[..., 36:39]),
            rotate(inv, angular_difference_world),
            successor[..., 7:36] - state[..., 7:36],
            successor[..., 42:71] - state[..., 42:71],
        ),
        -1,
    )


def apply_difference(state, delta):
    q = state[..., 3:7]
    next_q = F.normalize(product(q, rotation_exp(delta[..., 3:6])), dim=-1)
    omega_world = rotate(q, state[..., 39:42]) + rotate(q, delta[..., 9:12])
    return torch.cat(
        (
            state[..., :3] + rotate(conjugate(heading(q)), delta[..., :3]),
            next_q,
            state[..., 7:36] + delta[..., 12:41],
            state[..., 36:39] + rotate(q, delta[..., 6:9]),
            rotate(conjugate(next_q), omega_world),
            state[..., 42:71] + delta[..., 41:70],
        ),
        -1,
    )
