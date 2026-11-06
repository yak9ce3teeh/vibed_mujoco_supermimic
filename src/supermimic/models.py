"""Differentiable compact CPU profile of SuperMimic components.

State layout: MuJoCo qpos36 + qvel35. Networks and state parameterization are
local implementation choices where the original paper leaves details open.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .motion import JOINT_DIM, STATE_DIM, OBS_DIM, GOAL_DIM, HISTORY, FUTURE, ROOT_QUAT
from .task import goal_features, observation, rotate, conjugate, product
from .dynamics import DELTA_DIM, local_state, local_difference, apply_difference


def architecture(config):
    """Explicit hidden shapes; absent profiles retain schema3 legacy weights."""
    if not isinstance(config, dict):
        config = vars(config)
    profile = config.get("architecture_profile", "legacy")
    if profile == "legacy":
        hidden = [int(config.get("width", 128))] * 2
        result = {name: list(hidden) for name in ("encoder", "decoder", "world", "critic")}
    else:
        scale = float(profile.removeprefix("paper-"))
        if scale not in (.125, .25, .5, 1.):
            raise ValueError(f"Unknown architecture profile: {profile}")
        result = {
            "encoder": [round(n * scale) for n in (2048, 1024, 512, 256)],
            "decoder": [round(n * scale) for n in (2048, 1024, 512, 256)],
            "world": [round(n * scale) for n in (2048, 2048, 2048)],
            "critic": [round(n * scale) for n in (256, 128)],
        }
    saved = config.get("architecture")
    if saved is not None and saved != result:
        raise ValueError("Saved hidden shapes do not match architecture profile")
    return result


def mlp(inputs, outputs, hidden):
    hidden = [hidden, hidden] if isinstance(hidden, int) else list(hidden)
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    return nn.Sequential(*layers, nn.Linear(inputs, outputs))


def build_models(default, config):
    config = config if isinstance(config, dict) else vars(config)
    shapes = architecture(config)
    actor = Actor(default, action_limit=config.get("action_limit", 20.), hidden=shapes)
    return actor, WorldModel(hidden=shapes["world"]), Critic(hidden=shapes["critic"])


class Actor(nn.Module):
    def __init__(self, default, width=128, latent=32, action_limit=20.0, hidden=None):
        super().__init__()
        if not math.isfinite(action_limit) or action_limit <= 0:
            raise ValueError("action_limit must be positive and finite")
        self.action_limit = float(action_limit)
        self.register_buffer("default", torch.as_tensor(default, dtype=torch.float32))
        self.encoder = mlp(FUTURE * GOAL_DIM, latent, width if hidden is None else hidden["encoder"])
        self.decoder = mlp(HISTORY * OBS_DIM + latent, JOINT_DIM, width if hidden is None else hidden["decoder"])

    def forward(self, state, history, reference):
        goals = goal_features(state, reference).flatten(-2)
        z = self.encoder(goals).tanh()
        return self.decoder(torch.cat((history.flatten(-2), z), -1))

    def execute(self, action):
        return action.clip(-self.action_limit, self.action_limit)


class WorldModel(nn.Module):
    def __init__(self, width=128, hidden=None):
        super().__init__()
        self.transition = mlp(STATE_DIM + JOINT_DIM, DELTA_DIM, width if hidden is None else hidden)
        self.register_buffer("mean", torch.zeros(STATE_DIM))
        self.register_buffer("scale", torch.ones(STATE_DIM))
        self.register_buffer("delta_scale", torch.ones(DELTA_DIM) * 0.1)

    def fit_scaling(self, states, next_states):
        with torch.no_grad():
            inputs = local_state(states)
            self.mean.copy_(inputs.mean(0))
            self.scale.copy_(inputs.std(0).clamp_min(0.05))
            self.delta_scale.copy_(local_difference(states, next_states).std(0).clamp_min(0.005))

    def normalize(self, state):
        return (local_state(state) - self.mean) / self.scale

    def forward(self, state, action):
        delta = self.transition(torch.cat((self.normalize(state), action), -1)) * self.delta_scale
        return apply_difference(state, delta)

    def error(self, prediction, target):
        return (local_difference(prediction, target) / self.delta_scale).square().mean()


class Critic(nn.Module):
    def __init__(self, width=128, hidden=None):
        super().__init__()
        self.net = mlp(STATE_DIM + GOAL_DIM, 1, width if hidden is None else hidden)

    def forward(self, state, reference):
        return self.net(
            torch.cat((state, goal_features(state, reference.unsqueeze(-2)).squeeze(-2)), -1)
        ).squeeze(-1)
