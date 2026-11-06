"""One CLI for online collection, learned dynamics, BPTT, and true MuJoCo eval."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import hashlib
import sys
import tarfile
from importlib.metadata import version
from contextlib import contextmanager
from tqdm import tqdm
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from .models import Actor, Critic, WorldModel, architecture, build_models
from .task import observation, reward, TrackingTask
from .motion import (
    JOINT_NAMES,
    load_motions,
    HISTORY,
    FUTURE,
    HORIZON,
    WORLD_WINDOW,
    QPOS_DIM,
    QVEL_DIM,
    OBS_DIM,
)
from .mujoco_adapter import MujocoAdapter
from .run_layout import (IO_SCHEMA_VERSION, LEGACY_LAYOUT_SOURCE_SHA256, stage_key,
    evaluation_folder, checkpoint_path as evaluation_checkpoint_path, write_packed, load_weights, archive_terminal_before_resume, validate_layout_bridge)


DEVICE = torch.device("cpu")
TASK = None


def cosine_learning_rate(iteration):
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    fraction = min(iteration, 250_000) / 250_000
    return 2e-4 * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * fraction)))


class OptimizationProgress:
    """Local joint-actor iteration clock, separate from actual optimizer steps."""

    def __init__(self, world_updates, actor_updates):
        if world_updates <= 0 or actor_updates <= 0:
            raise ValueError("world_updates and actor_updates must be positive")
        self.world_updates, self.actor_updates = world_updates, actor_updates
        self.joint_iterations = 0
        self.steps = dict(world=0, actor=0, critic=0)
        self.warmup_world_steps = 0
        self.last_used_lr = dict.fromkeys(self.steps)
        self.last_used_iteration = dict.fromkeys(self.steps)

    def step(self, optimizer, name, iteration, warmup=False):
        if warmup and name != "world":
            raise ValueError("warmup updates only the world model")
        lr = 2e-4 if warmup else cosine_learning_rate(iteration)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        self.steps[name] += 1
        self.warmup_world_steps += int(warmup)
        self.last_used_lr[name] = lr
        self.last_used_iteration[name] = None if warmup else iteration

    def audit(self, mode="training"):
        return dict(
            mode=mode,
            counting="local logical iteration mapping; one completed joint actor update",
            formula="base_lr*(floor_ratio+(1-floor_ratio)*0.5*(1+cos(pi*min(n,decay_iterations)/decay_iterations)))",
            base_lr=2e-4,
            decay_iterations=250_000,
            floor_ratio=0.03,
            minimum_lr=6e-6,
            warmup="WM-only constant base_lr; excluded from joint_iterations (local choice)",
            block_order="W world steps then A actor updates; retained local batching, not paper alternation",
            world_iteration_mapping="n_at_block_start + floor(j*A/W), zero-based world step j",
            actor_critic_mapping="actor iteration n; all associated 20 critic steps share n",
            world_updates_per_block=self.world_updates,
            actor_updates_per_block=self.actor_updates,
            joint_iterations=self.joint_iterations,
            optimizer_steps=dict(self.steps),
            warmup_world_optimizer_steps=self.warmup_world_steps,
            last_used_lr=dict(self.last_used_lr),
            last_used_iteration=dict(self.last_used_iteration),
            next_joint_lr=cosine_learning_rate(self.joint_iterations),
            actor_gradient_clipping="global norm 1.0, then elementwise [-0.1,0.1]",
            adam_betas=dict(actor=[0.9, 0.95], world=[0.9, 0.999], critic=[0.9, 0.999]),
            beta_provenance="actor from paper; WM/critic PyTorch defaults, not disclosed paper values",
            schedule_scope="shared cosine is an interpretation of the common paper schedule; paper total 320k is not a run budget here",
            resumable=False,
        )


def clip_actor_gradients(parameters):
    parameters = list(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.clamp_(-0.1, 0.1)


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)


def future(motion, frames):
    indices = np.asarray(frames)[..., None] + np.arange(1, FUTURE + 1)
    return tensor(reference_window(motion, indices))


def noisy_observation(obs):
    bounds = obs.new_tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
    return obs + (2 * torch.rand_like(obs) - 1) * bounds


def rollout(env, motion, start, actor, mode, rng, noise=0.0, max_steps=None):
    state = env.reset(motion.states[start, :36], motion.states[start, 36:])
    previous = np.zeros(29)
    history = torch.zeros(1, HISTORY, OBS_DIM, device=actor.default.device)
    current = tensor(np.r_[state["qpos"], state["qvel"]]).unsqueeze(0)
    obs = observation(current, tensor(previous).unsqueeze(0), actor.default)
    history[:, -1] = noisy_observation(obs) if noise else obs
    states, actions, next_states, rewards, dones, frames, metrics, raw_actions, reward_terms = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    term_names = None
    steps = len(motion.states) - start - 1
    if steps <= 0:
        raise ValueError("rollout start needs at least one remaining transition")
    if max_steps is not None:
        steps = min(steps, max_steps)
    for frame in range(start, start + steps):
        with torch.no_grad():
            if mode == "actor":
                action = actor(current, history, future(motion, [frame]))[0].cpu().numpy()
                if noise:
                    action = action + rng.normal(0, noise, 29)
            elif mode == "pd":
                action = (motion.states[frame + 1, 7:36] - env.q_default) / env.action_scale
            elif mode == "zero":
                action = np.zeros(29)
            else:
                raise ValueError(mode)
        raw_actions.append(action.copy())
        action = actor.execute(action)
        old = np.r_[env.data.qpos.copy(), env.data.qvel.copy()]
        state = env.step(action)
        next_state = np.r_[state["qpos"], state["qvel"]]
        r, done, metric = reward(env, motion, frame + 1, action, previous)
        terms = metric.pop("reward_terms")
        term_names = list(terms)
        reward_terms.append(list(terms.values()))
        if not np.isclose(sum(terms.values()), r, atol=1e-10):
            raise AssertionError("reward terms do not sum to actual reward")
        states.append(old)
        actions.append(action.copy())
        next_states.append(next_state)
        rewards.append(r)
        dones.append(done)
        frames.append(frame)
        metrics.append(metric)
        previous = action.copy()
        current = tensor(next_state).unsqueeze(0)
        obs = observation(current, tensor(previous).unsqueeze(0), actor.default)
        if noise:
            obs = noisy_observation(obs)
        history = torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
        if done:
            break
    terminated = bool(dones[-1])
    completed = not terminated and frames[-1] + 1 == len(motion.states) - 1
    cause = "hard_fall" if terminated else "reference_end" if completed else "collection_cap"
    truncated = np.zeros(len(dones), dtype=bool)
    truncated[-1] = not terminated
    return dict(
        reward_terms=np.asarray(reward_terms),
        reward_term_names=np.asarray(term_names),
        termination_cause=cause,
        completed=completed,
        truncated=truncated,
        remaining_transitions=len(motion.states) - start - 1,
        raw_actions=np.asarray(raw_actions),
        action_limit=actor.action_limit,
        states=np.asarray(states),
        actions=np.asarray(actions),
        next_states=np.asarray(next_states),
        rewards=np.asarray(rewards),
        dones=np.asarray(dones),
        frames=np.asarray(frames),
        metrics=metrics,
        clip=motion.name,
        start=start,
        mode=mode,
    )


def sample(replay, batch, length, rng):
    eligible = [e for e in replay if len(e["actions"]) >= length]
    if not eligible:
        raise RuntimeError(f"no episode has {length} contiguous transitions")
    samples = []
    for _ in range(batch):
        ep = eligible[rng.integers(len(eligible))]
        i = int(rng.integers(len(ep["actions"]) - length + 1))
        samples.append((ep, i))
    return samples


def train_world(
    world,
    optimizer,
    replay,
    motions,
    rng,
    updates,
    batch,
    progress=None,
    warmup=False,
    details=None,
    sample_fn=sample,
):
    losses = []
    for update in range(updates):
        samples = sample_fn(replay, batch, WORLD_WINDOW, rng)
        state = tensor(np.stack([e["states"][i] for e, i in samples]))
        loss = 0.0
        for k in range(WORLD_WINDOW):
            action = tensor(np.stack([e["actions"][i + k] for e, i in samples]))
            target = tensor(np.stack([e["next_states"][i + k] for e, i in samples]))
            state = world(state, action)  # No teacher forcing after the first state.
            loss = loss + world.error(state, target)
        optimizer.zero_grad()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite world-model loss")
        (loss / WORLD_WINDOW).backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1.0, error_if_nonfinite=True)
        if progress is None:
            optimizer.step()
        else:
            iteration = progress.joint_iterations + update * progress.actor_updates // updates
            progress.step(optimizer, "world", iteration, warmup=warmup)
        losses.append(float(loss.detach() / WORLD_WINDOW))
    if details is not None:
        details.extend(losses)
    return float(np.mean(losses))


def imagine(
    actor,
    world,
    critic,
    state,
    history,
    reference,
    horizon=HORIZON,
    obs_noise=False,
    task=None,
    statistics=None,
):
    task = TASK if task is None else task
    if task is None:
        raise ValueError("imagination requires the compiled-model TrackingTask")
    previous = history[:, -1, -29:]
    states, successors, actions, previous_actions, goals, raw_actions = [], [], [], [], [], []
    for k in range(horizon):
        raw = actor(state, history, reference[:, k + 1 : k + FUTURE + 1])
        raw_actions.append(raw)
        action = actor.execute(raw)
        next_state = world(state, action)
        states.append(state)
        successors.append(next_state)
        actions.append(action)
        previous_actions.append(previous)
        goals.append(reference[:, k + 1])
        obs = observation(next_state, action, actor.default)
        if obs_noise:
            obs = noisy_observation(obs)
        history = torch.cat((history[:, 1:], obs[:, None]), 1)
        state, previous = next_state, action
    if statistics is not None:
        raw = torch.stack(raw_actions).detach()
        statistics["imagined_action_clip_fraction"] = float(
            (raw.abs() > actor.action_limit).to(torch.float32).mean()
        )
        statistics["imagined_raw_action_abs_max"] = float(raw.abs().max())
    batch = len(state)
    rewards, alive, _ = task(
        torch.stack(successors).flatten(0, 1),
        torch.stack(goals).flatten(0, 1),
        torch.stack(actions).flatten(0, 1),
        torch.stack(previous_actions).flatten(0, 1),
    )
    rewards, alive = rewards.reshape(horizon, batch), alive.reshape(horizon, batch)
    survival = torch.cumprod(torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0), 0)
    discounts = state.new_tensor(0.99).pow(
        torch.arange(horizon, device=state.device, dtype=state.dtype)
    )
    bootstrap = critic(state, reference[:, horizon + 1])
    objective = (discounts[:, None] * survival * rewards).sum(0) + 0.99**horizon * alive.prod(
        0
    ) * bootstrap
    records = [(states[k], goals[k], rewards[k], alive[k]) for k in range(horizon)]
    return -objective.mean(), records, bootstrap


def policy_batch(replay, motions, actor, rng, batch):
    eligible = []
    for episode in replay:
        last = len(motions[episode["clip"]].states) - 1
        ids = np.flatnonzero(episode["frames"] + HORIZON + FUTURE - 1 <= last)
        if len(ids):
            eligible.append((episode, ids))
    if not eligible:
        raise RuntimeError("no valid H32+future10 policy seed; collect more real transitions")
    samples = []
    for _ in range(batch):
        episode, ids = eligible[rng.integers(len(eligible))]
        samples.append((episode, int(ids[rng.integers(len(ids))])))
    states, histories, refs = [], [], []
    for ep, i in samples:
        state = ep["states"][i]
        history = np.zeros((10, 93), dtype=np.float32)
        for slot, j in enumerate(range(max(0, i - 9), i + 1), start=10 - min(i + 1, 10)):
            prev = ep["actions"][j - 1] if j else np.zeros(29)
            history[slot] = (
                observation(tensor(ep["states"][j]), tensor(prev), actor.default).cpu().numpy()
            )
        frame = ep["frames"][i]
        indices = frame + np.arange(HORIZON + FUTURE)
        states.append(state)
        histories.append(history)
        refs.append(motions[ep["clip"]].states[indices])
    return tensor(states), tensor(histories), tensor(refs)


def train_actor(
    actor, world, critic, actor_opt, critic_opt, replay, motions, rng, batch, progress=None, batch_fn=policy_batch
):
    frozen = {k: v.clone() for k, v in world.state_dict().items()}
    for p in world.parameters():
        p.requires_grad_(False)
    for p in critic.parameters():
        p.requires_grad_(False)
    states, histories, references = batch_fn(replay, motions, actor, rng, batch)
    statistics = {}
    loss, records, bootstrap = imagine(
        actor,
        world,
        critic,
        states,
        noisy_observation(histories),
        references,
        obs_noise=True,
        statistics=statistics,
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite actor loss")
    actor_opt.zero_grad()
    loss.backward()
    grad = float(
        torch.sqrt(sum(p.grad.square().sum() for p in actor.parameters() if p.grad is not None))
    )
    clip_actor_gradients(actor.parameters())
    iteration = progress.joint_iterations if progress is not None else None
    if progress is None:
        actor_opt.step()
    else:
        progress.step(actor_opt, "actor", iteration)
    for p in critic.parameters():
        p.requires_grad_(True)
    # Imagined GAE(lambda=.95), frozen targets, five epochs x four minibatches.
    with torch.no_grad():
        values = [critic(s, r) for s, r, _, _ in records] + [bootstrap.detach()]
        gae = torch.zeros_like(values[0])
        targets = []
        for k in reversed(range(len(records))):
            _, _, reward_k, continuation = records[k]
            delta = reward_k.detach() + 0.99 * continuation.detach() * values[k + 1] - values[k]
            gae = delta + 0.99 * 0.95 * continuation.detach() * gae
            targets.append(gae + values[k])
        targets = torch.stack(list(reversed(targets))).flatten()
    ss = torch.stack([x[0].detach() for x in records]).flatten(0, 1)
    rr = torch.stack([x[1].detach() for x in records]).flatten(0, 1)
    critic_losses = []
    for _ in range(5):
        for ids in torch.randperm(len(ss)).chunk(4):
            critic_opt.zero_grad()
            closs = F.mse_loss(critic(ss[ids], rr[ids]), targets[ids])
            critic_losses.append(float(closs.detach()))
            if not bool(torch.isfinite(closs)):
                raise FloatingPointError("nonfinite critic loss")
            closs.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            if progress is None:
                critic_opt.step()
            else:
                progress.step(critic_opt, "critic", iteration)
    if progress is not None:
        progress.joint_iterations += 1
    unchanged = all(torch.equal(v, frozen[k]) for k, v in world.state_dict().items())
    for p in world.parameters():
        p.requires_grad_(True)
    return dict(
        critic_losses=critic_losses,
        critic_loss_mean=float(np.mean(critic_losses)),
        critic_loss_min=float(np.min(critic_losses)),
        critic_loss_max=float(np.max(critic_losses)),
        actor_loss=float(loss.detach()),
        actor_gradient_norm=grad,
        world_unchanged=unchanged,
        **statistics,
    )


def world_error(world, episodes, horizon=8):
    errors, persistence = [], []
    with torch.no_grad():
        for ep in episodes:
            for i in range(0, len(ep["actions"]) - horizon + 1, horizon):
                state = tensor(ep["states"][i : i + 1])
                initial = state.clone()
                for k in range(horizon):
                    state = world(state, tensor(ep["actions"][i + k : i + k + 1]))
                    target = tensor(ep["next_states"][i + k : i + k + 1])
                    errors.append((state - target).square().cpu().numpy())
                    persistence.append((initial - target).square().cpu().numpy())
    if not errors:
        return dict(count=0, rmse=None, persistence_rmse=None)
    groups = {
        "root_position_m": slice(0, 3),
        "quaternion_unitless": slice(3, 7),
        "joint_position_rad": slice(7, 36),
        "root_linear_velocity_m_s": slice(36, 39),
        "root_angular_velocity_rad_s": slice(39, 42),
        "joint_velocity_rad_s": slice(42, 71),
    }
    grouped = {
        name: dict(
            rmse=float(np.sqrt(np.mean(np.asarray(errors)[..., ids]))),
            persistence_rmse=float(np.sqrt(np.mean(np.asarray(persistence)[..., ids]))),
        )
        for name, ids in groups.items()
    }
    return dict(
        count=len(errors),
        mixed_unit_rmse=float(np.sqrt(np.mean(errors))),
        mixed_unit_persistence_rmse=float(np.sqrt(np.mean(persistence))),
        groups=grouped,
    )


def outcome_error(world, episodes, motions, reward_mean=None):
    predictions, targets, probabilities, labels = [], [], [], []
    with torch.no_grad():
        for ep in episodes:
            predicted_state = world(tensor(ep["states"]), tensor(ep["actions"]))
            previous = np.concatenate((np.zeros((1, 29)), ep["actions"][:-1]), axis=0)
            reference = tensor(motions[ep["clip"]].states[ep["frames"] + 1])
            r, alive, _ = TASK(predicted_state, reference, tensor(ep["actions"]), tensor(previous))
            predictions.extend(r.cpu().tolist())
            probabilities.extend((1 - alive).cpu().tolist())
            targets.extend(ep["rewards"])
            labels.extend(ep["dones"])
    predictions, targets, probabilities, labels = map(
        np.asarray, (predictions, targets, probabilities, labels)
    )
    mean = float(targets.mean()) if reward_mean is None else float(reward_mean)
    correlation = (
        float(np.corrcoef(predictions, targets)[0, 1])
        if predictions.std() > 0 and targets.std() > 0
        else None
    )
    return dict(
        count=len(labels),
        reward_rmse=float(np.sqrt(np.mean((predictions - targets) ** 2))),
        reward_correlation=correlation,
        reward_constant_mean=mean,
        reward_constant_mean_rmse=float(np.sqrt(np.mean((targets - mean) ** 2))),
        terminal_positive_count=int(labels.sum()),
        terminal_negative_count=int((~labels.astype(bool)).sum()),
        terminal_brier=float(np.mean((probabilities - labels) ** 2)),
        terminal_positive_mean_probability=float(probabilities[labels.astype(bool)].mean())
        if labels.any()
        else None,
        terminal_negative_mean_probability=float(probabilities[~labels.astype(bool)].mean()),
    )


def directional_check(actor, world, critic, replay, motions, rng, batch_fn=policy_batch):
    a, w, c = (
        copy.deepcopy(actor).double(),
        copy.deepcopy(world).double(),
        copy.deepcopy(critic).double(),
    )
    state, history, refs = [x.double() for x in batch_fn(replay, motions, actor, rng, 2)]
    params = list(a.parameters())
    directions = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum(d.square().sum() for d in directions))
    directions = [d / norm for d in directions]
    loss = imagine(a, w, c, state, history, refs)[0]
    gradients = torch.autograd.grad(loss, params)
    analytic = float(sum((g * d).sum() for g, d in zip(gradients, directions)))
    eps = 1e-5
    with torch.no_grad():
        for p, d in zip(params, directions):
            p.add_(eps * d)
        plus = float(imagine(a, w, c, state, history, refs)[0])
        for p, d in zip(params, directions):
            p.add_(-2 * eps * d)
        minus = float(imagine(a, w, c, state, history, refs)[0])
    finite_difference = (plus - minus) / (2 * eps)
    absolute = abs(analytic - finite_difference)
    relative = absolute / max(abs(analytic), abs(finite_difference), 1e-12)
    return dict(
        analytic=analytic,
        finite_difference=finite_difference,
        absolute_error=absolute,
        relative_error=relative,
        passed=relative <= 0.001 or absolute <= 1e-6,
        horizon=32,
    )


def summarize(ep, dt):
    terms = ep.get("reward_terms")
    term_summary = (
        {}
        if terms is None
        else {
            str(name): dict(sum=float(terms[:, i].sum()), mean=float(terms[:, i].mean()))
            for i, name in enumerate(ep["reward_term_names"])
        }
    )
    return dict(
        reward_terms=term_summary,
        clip=ep["clip"],
        start=ep["start"],
        mode=ep["mode"],
        steps=len(ep["actions"]),
        duration_seconds=len(ep["actions"]) * dt,
        return_sum=float(ep["rewards"].sum()),
        terminated=bool(ep["dones"][-1]),
        termination_cause=ep.get(
            "termination_cause", "hard_fall" if ep["dones"][-1] else "unknown"
        ),
        completed=ep.get("completed", False),
        remaining_transitions=ep.get("remaining_transitions", len(ep["actions"])),
        completion_fraction=len(ep["actions"])
        / ep.get("remaining_transitions", len(ep["actions"])),
        normalized_return=float(ep["rewards"].sum())
        / (0.1 * ep.get("remaining_transitions", len(ep["actions"]))),
        reward_per_step=float(ep["rewards"].mean()),
        action_clip_fraction=float(np.mean(np.abs(ep["raw_actions"]) > ep["action_limit"])),
        **{k: float(np.mean([m[k] for m in ep["metrics"]])) for k in ep["metrics"][0]},
    )


def reference_window(motion, indices):
    indices = np.asarray(indices)
    result = motion.states[np.minimum(indices, len(motion.states) - 1)].copy()
    result[..., QPOS_DIM:] = np.where(
        (indices >= len(motion.states))[..., None], 0.0, result[..., QPOS_DIM:]
    )
    return result


def trim_replay(replay, capacity):
    while len(replay) > 1 and sum(len(e["actions"]) for e in replay) > capacity:
        replay.pop(0)
    if sum(len(e["actions"]) for e in replay) > capacity:
        raise ValueError(
            "one episode exceeds replay capacity; increase capacity, never split episodes"
        )


def trim_replay_wave(replay, capacity):
    """Same whole-episode FIFO suffix, counting the complete wave only once."""
    total = sum(len(ep["actions"]) for ep in replay)
    removed = 0
    while len(replay) - removed > 1 and total > capacity:
        total -= len(replay[removed]["actions"])
        removed += 1
    if removed:
        del replay[:removed]
    if total > capacity:
        raise ValueError("one episode exceeds replay capacity; increase capacity, never split episodes")


def rng_state(rng):
    return dict(
        numpy=copy.deepcopy(rng.bit_generator.state),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state, rng):
    rng.bit_generator.state = copy.deepcopy(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextmanager
def preserve_rng(rng):
    state = rng_state(rng)
    try:
        yield
    finally:
        restore_rng(state, rng)


def atomic_save(value, path, json_format=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        if json_format:
            handle.write(json.dumps(value, indent=2, allow_nan=False).encode())
        else:
            torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_log(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_identity(args, env):
    buffer = np.empty(mujoco.mj_sizeModel(env.model), dtype=np.uint8)
    mujoco.mj_saveModel(env.model, buffer=buffer)
    return dict(
        model_xml=file_hash(args.model),
        compiled_model=hashlib.sha256(buffer.tobytes()).hexdigest(),
        controller=file_hash(args.controller),
        motions=file_hash(args.motions),
    )


def foot_friction_one(env):
    roots = [env.model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    bodies = []
    for body in range(env.model.nbody):
        parent = body
        while parent and parent not in roots:
            parent = int(env.model.body_parentid[parent])
        if parent in roots:
            bodies.append(body)
    ids = np.flatnonzero(np.isin(env.model.geom_bodyid, bodies))
    if not np.allclose(env.model.geom_friction[ids, 0], 1.0):
        raise ValueError(
            "paper eval needs feet friction=1; explicit protocol change required, training is not silently modified"
        )
    return dict(
        foot_geom_ids=ids.tolist(),
        sliding_friction=env.model.geom_friction[ids, 0].tolist(),
        randomization=False,
        observation_noise=False,
        action_noise=False,
    )


def weights_payload(actor, initial, world, critic, args, controller, progress):
    return dict(
        schema_version=3,
        state_representation="public71_local_input71_delta70",
        task_semantics="analytic_global_body_v1",
        action_semantics="linear_decoder_clip",
        actor=actor.state_dict(),
        initial_actor=initial.state_dict(),
        world=world.state_dict(),
        critic=critic.state_dict(),
        config=vars(args),
        controller=controller,
        optimization=progress.audit(),
    )


def evaluate_snapshot(
    env,
    motions,
    actor,
    initial,
    world,
    critic,
    args,
    controller,
    progress,
    folder,
    stage,
    source_optimization=None,
    checkpoint_reference=None,
    total_collected=0,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = weights_payload(actor, initial, world, critic, args, controller, progress)
    if source_optimization is not None:
        payload["optimization"] = source_optimization
    checkpoint_reference = Path(checkpoint_reference) if checkpoint_reference else folder.parent.parent / "ckpt" / (stage_key(stage)+".pt")
    checkpoint_reference.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_reference.exists():
        previous = load_weights(checkpoint_reference)
        if any(not torch.equal(value.cpu(), previous[key][name].cpu())
               for key in ("actor", "initial_actor", "world", "critic")
               for name,value in payload[key].items()):
            raise ValueError("Existing checkpoint weights differ from evaluated state")
    else:
        atomic_save(payload, checkpoint_reference)
    packed_cases = {}
    physics = foot_friction_one(env)
    rows = []
    for clip, motion in motions.items():
        for start in (0, 100, 200):
            if start >= len(motion.states) - 1:
                continue
            for mode, policy in [
                ("initial", initial),
                ("trained", actor),
                ("pd", actor),
                ("zero", actor),
            ]:
                ep = rollout(
                    env,
                    motion,
                    start,
                    policy,
                    "actor" if mode in ("initial", "trained") else mode,
                    np.random.default_rng(0),
                )
                row = summarize(ep, env.control_dt)
                row["policy"] = mode
                row["protocol"] = (
                    "paper_main_first_frame"
                    if start == 0 and clip == args.train_clip
                    else "local_auxiliary"
                    if clip == args.train_clip
                    else "development_diagnostic"
                )
                rows.append(row)
                identifier = f"{clip}-start{start}-{mode}"
                row.update(trajectory_key=identifier, trajectory_bundle="trajectories.npz")
                packed_cases[identifier] = {k:v for k,v in ep.items() if isinstance(v,np.ndarray)}
    write_packed(folder / "trajectories.npz", packed_cases)
    train = [r for r in rows if r["clip"] == args.train_clip and r["policy"] == "trained"]
    score = dict(
        completed_count=sum(r["completed"] for r in train),
        macro_normalized_return=float(np.mean([r["normalized_return"] for r in train])),
        macro_global_mpjpe=float(np.mean([r["global_mpjpe"] for r in train])),
        median_steps=float(np.median([r["steps"] for r in train])),
    )
    report = dict(
        config=vars(args),
        controller=controller,
        stage=stage_key(stage),
        io_schema_version=IO_SCHEMA_VERSION,
        joint_block=progress.joint_iterations // args.actor_updates,
        total_collected=total_collected,
        physics=physics,
        canonical_eval_backend="native",
        collection_backend=getattr(args,"physics_backend","native"),
        evaluation=rows,
        train_auxiliary_macro=score,
        paper_main=next(r for r in train if r["start"] == 0),
        checkpoint=str(checkpoint_reference),
        checkpoint_sha256=file_hash(checkpoint_reference),
        evaluated_weights_sha256=file_hash(checkpoint_reference),
        normalized_return_definition="undiscounted sum(reward)/(.1*(T-1-start)); full planned denominator, no clipping; .1=5*.02",
        diagnostics_note="MPJPE/joint RMSE and 3-start macro are local diagnostics, not paper main score",
        optimization=progress.audit(),
        provenance=asset_identity(args, env),
    )
    if source_optimization is not None:
        report["optimization"] = source_optimization
        report["optimizer_steps_this_eval"] = dict(world=0, actor=0, critic=0)
    atomic_save(report, folder / "report.json", json_format=True)
    checkpoint_json = dict(stage=report["stage"], joint_block=report["joint_block"],
        total_collected=total_collected, path=str(folder), checkpoint=str(checkpoint_reference),
        checkpoint_sha256=report["checkpoint_sha256"], score=score, paper_main=report["paper_main"],
        checkpoint_kind="full_resume" if "train_state_version" in load_weights(checkpoint_reference) else "weights_only",
        io_schema_version=IO_SCHEMA_VERSION)
    if source_optimization is None:
        atomic_save(checkpoint_json, checkpoint_reference.with_suffix(".json"), json_format=True)
    return report


def load_progress(progress, audit):
    progress.joint_iterations = audit["joint_iterations"]
    progress.steps = dict(audit["optimizer_steps"])
    progress.warmup_world_steps = audit["warmup_world_optimizer_steps"]
    progress.last_used_lr = dict(audit["last_used_lr"])
    progress.last_used_iteration = dict(audit["last_used_iteration"])


def resume_identity(args, assets):
    keys = (
        "train_clip",
        "heldout_clip",
        "width",
        "batch",
        "episodes",
        "world_updates",
        "actor_updates",
        "seed",
        "action_limit",
        "warmup_blocks",
        "warmup_max",
        "warmup_gate",
        "replay_capacity",
        "training_gates",
        "physics_backend",
        "num_envs",
    )
    return dict(
        configuration={**{key: getattr(args, key) for key in keys}, **({"architecture": architecture(args)} if getattr(args, "architecture_profile", "legacy") != "legacy" else {})},
        assets=assets,
        protocol="full_clip_tail_zero_velocity_valid_policy_seed_v1",
        sampler=(dict(backend="warp",num_envs=args.num_envs,versions={name:version(name) for name in ['mujoco','mujoco-warp','warp-lang']},
                      physics_dtype="float32",nconmax=128,njmax=512,scheduling="synchronous episode waves")
                 if args.physics_backend=="warp" else dict(backend="native",num_envs=1,physics_dtype="float64")),
    )


def finite_modules(*modules):
    if not all(bool(torch.isfinite(p).all()) for m in modules for p in m.parameters()):
        raise FloatingPointError("nonfinite model parameters")


def evaluate_checkpoint(checkpoint_path, output, device="cpu"):
    global DEVICE, TASK
    DEVICE = torch.device(device)
    torch.set_num_threads(2)
    if (Path(output) / "report.json").exists() or (Path(output) / "checkpoint.pt").exists():
        raise FileExistsError("evaluation output already contains results; use a fresh directory")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if checkpoint.get("schema_version") != 3:
        raise ValueError("evaluation requires schema3 weights")
    if checkpoint["config"].get("dataset_mode")=="full40_stream":
        from .multi_training import evaluate_checkpoint as evaluate_full40
        return evaluate_full40(checkpoint_path, output, device)
    args = argparse.Namespace(**checkpoint["config"])
    controller = checkpoint.get("controller") or json.loads(Path(args.controller).read_text())
    env = MujocoAdapter(
        args.model,
        JOINT_NAMES,
        controller["kp"],
        controller["kd"],
        controller["q_default"],
        controller["action_scale"],
    )
    TASK = TrackingTask(env.model).to(DEVICE)
    motions = load_motions(args.motions, [args.train_clip, args.heldout_clip], env)
    actor, world, critic = (module.to(DEVICE) for module in build_models(env.q_default, args))
    initial = copy.deepcopy(actor)
    for module, key in [
        (actor, "actor"),
        (initial, "initial_actor"),
        (world, "world"),
        (critic, "critic"),
    ]:
        module.load_state_dict(checkpoint[key])
    progress = OptimizationProgress(args.world_updates, args.actor_updates)
    if "joint_iterations" in checkpoint.get("optimization", {}):
        load_progress(progress, checkpoint["optimization"])
    report = evaluate_snapshot(
        env,
        motions,
        actor,
        initial,
        world,
        critic,
        args,
        controller,
        progress,
        Path(output) / "evals" / "final",
        "final",
        source_optimization=checkpoint.get(
            "optimization", {"status": "legacy historical optimization unknown"}
        ),
        checkpoint_reference=checkpoint_path,
        total_collected=checkpoint.get("run_state", {}).get("total_collected", 0),
    )
    report["source_checkpoint"] = str(checkpoint_path)
    report["source_checkpoint_sha256"] = file_hash(checkpoint_path)
    report["source_optimization"] = checkpoint.get(
        "optimization", {"status": "legacy historical optimization unknown"}
    )
    atomic_save(report, Path(output) / "report.json", json_format=True)
    return report


def main():
    global DEVICE, TASK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--motions", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-clip", default="walk1_subject1_clip0")
    parser.add_argument("--heldout-clip", default="walk1_subject2_clip0")
    parser.add_argument("--warmup-blocks", type=int, default=5)
    parser.add_argument("--warmup-max", type=int)
    parser.add_argument("--warmup-gate", action="store_true")
    parser.add_argument("--training-gates", action="store_true")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--world-updates", type=int, default=100)
    parser.add_argument("--actor-updates", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--architecture-profile", choices=["legacy", "paper-0.125", "paper-0.25", "paper-0.5", "paper-1"], default="legacy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-limit", type=float, default=20.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--physics-backend", choices=["native","warp"], default="native")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--resume", help="Full ckpt/final.pt, never a weights-only checkpoint")
    parser.add_argument("--allow-layout-resume", action="store_true", help="Explicitly migrate only the audited legacy IO source hash; algorithm identity stays strict")
    parser.add_argument("--eval-checkpoint")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--replay-capacity", type=int, default=32768)
    args = parser.parse_args()
    args.architecture = architecture(args)
    args.warmup_max = args.warmup_max or args.warmup_blocks
    if (
        min(
            args.world_updates,
            args.actor_updates,
            args.episodes,
            args.eval_every,
            args.checkpoint_every,
        )
        <= 0
    ):
        parser.error("update/episode/period counts must be positive")
    if args.num_envs<1 or (args.physics_backend=="native" and args.num_envs!=1):
        parser.error('Native collection requires num-envs1; Warp num-envs must be positive')
    if args.physics_backend=="warp" and (args.device!="cuda" or args.episodes<args.num_envs or args.episodes%args.num_envs):
        parser.error('Warp requires --device cuda and --episodes a positive multiple of --num-envs')
    if args.warmup_max < args.warmup_blocks:
        parser.error("warmup-max must be >= warmup-blocks")
    if args.eval_checkpoint:
        print(
            json.dumps(
                evaluate_checkpoint(args.eval_checkpoint, args.output, args.device)["paper_main"]
            ),
            flush=True,
        )
        return
    DEVICE = torch.device(args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    if (
        any((out / name).exists() for name in ("latest.pt", "report.json", "checkpoint.pt"))
        and not args.resume
    ):
        parser.error("output already has training state; choose a fresh directory or --resume")
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    controller = json.loads(Path(args.controller).read_text())
    env = MujocoAdapter(
        args.model,
        JOINT_NAMES,
        controller["kp"],
        controller["kd"],
        controller["q_default"],
        controller["action_scale"],
    )
    if env.model.nq != 36 or not np.array_equal(env.qpos_indices, np.arange(7, 36)):
        parser.error("requires the declared G1 state71 order")
    physics = foot_friction_one(env)
    TASK = TrackingTask(env.model).to(DEVICE)
    collector=None
    if args.physics_backend=="warp":
        try:
            from .warp_adapter import MujocoWarpAdapter
            from .warp_collect import collect_warp
        except ImportError as error:
            raise RuntimeError('Warp backend dependencies unavailable; use isolated pinned environment, no CPU fallback') from error
        collector=MujocoWarpAdapter(env,args.num_envs,DEVICE)
    sampler_metadata=collector.metadata if collector else dict(backend="native",num_envs=1,physics_dtype="float64",canonical_eval_backend="native")
    motions = load_motions(args.motions, [args.train_clip, args.heldout_clip], env)
    if len(motions[args.train_clip].states) < HORIZON + FUTURE + 1:
        parser.error("train clip too short for H32+future10")
    actor, world, critic = (module.to(DEVICE) for module in build_models(env.q_default, args))
    initial = copy.deepcopy(actor)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=2e-4, betas=(0.9, 0.95))
    world_opt = torch.optim.Adam(world.parameters(), lr=2e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=2e-4)
    progress = OptimizationProgress(args.world_updates, args.actor_updates)
    assets = asset_identity(args, env)
    identity = resume_identity(args, assets)
    source_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
    ).hexdigest()
    started = time.monotonic()
    elapsed_before = 0.0
    state = dict(
        warm_blocks=0,
        joint_blocks=0,
        warm_complete=args.warmup_blocks == 0,
        total_collected=0,
        next_episode_id=0,
        eval_history=[],
        best_initial=None,
        best_trained=None,
        reason=None,
        training=[],
    )
    replay = []
    validation = []
    initial_critic = {k: v.clone() for k, v in critic.state_dict().items()}
    if args.resume:
        saved = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        if saved.get("train_state_version") != 1:
            parser.error("checkpoint is weights-only or unsupported; cannot resume")
        layout_migration = None
        if saved.get("source_sha256") != source_hash:
            if args.allow_layout_resume and args.physics_backend!="native":
                parser.error("Legacy native replay/checkpoint cannot resume into Warp; start a fresh Warp run")
            if args.allow_layout_resume and args.physics_backend=="native" and args.num_envs==1 and saved.get("source_sha256") == LEGACY_LAYOUT_SOURCE_SHA256:
                bridge = validate_layout_bridge(saved["source_sha256"],source_hash)
                layout_migration = dict(bridge=bridge,old_source_sha256=saved["source_sha256"], new_source_sha256=source_hash,
                    scope="Explicit audited legacy-profile migration; initialization/forward/gradients tested equal; optimizer/replay/RNG identity unchanged")
                tqdm.write(json.dumps(dict(layout_resume_migration=layout_migration)))
            else:
                parser.error("resume source hash differs; restore its source or explicitly use --allow-layout-resume for the single audited legacy IO hash")
        saved_identity=copy.deepcopy(saved["identity"])
        if args.physics_backend=="native" and args.num_envs==1 and layout_migration:
            # A historical native run had no backend fields. This is the only
            # explicit identity upgrade; legacy replay can never become Warp.
            saved_identity["configuration"].setdefault("physics_backend","native")
            saved_identity["configuration"].setdefault("num_envs",1)
            saved_identity.setdefault("sampler",dict(backend="native",num_envs=1,physics_dtype="float64"))
        if saved_identity != identity:
            parser.error(
                "resume identity mismatch: assets/architecture/optimization/seed/clip/sampling must match"
            )
        terminal_archive = archive_terminal_before_resume(out,saved)
        for module, key in [
            (actor, "actor"),
            (initial, "initial_actor"),
            (world, "world"),
            (critic, "critic"),
        ]:
            module.load_state_dict(saved[key])
        for optimizer, key in [(actor_opt, "actor"), (world_opt, "world"), (critic_opt, "critic")]:
            optimizer.load_state_dict(saved["optimizers"][key])
        load_progress(progress, saved["optimization"])
        state = saved["run_state"]
        for row in state["eval_history"]:
            old_stage = row["stage"]
            if old_stage.startswith("final-joint-"):
                new_stage = f"interrupt-{row['joint_block']:04d}"
            else:
                new_stage = stage_key(old_stage)
            row.update(stage=new_stage, path=str(out/"evals"/new_stage))
        for best_key in ("best_initial", "best_trained"):
            row = state.get(best_key)
            if row:
                key = stage_key(row["stage"])
                row.update(stage=key, path=str(out/"evals"/key))
        state["reason"] = None
        replay = saved["replay"]
        validation = saved["validation"]
        initial_critic = saved["initial_critic"]
        elapsed_before = saved["elapsed_seconds"]
        restore_rng(saved["rng"], rng)
    else:
        layout_migration = None
        terminal_archive = None
        with tarfile.open(out / "ckpt/source.tar.gz", "w:gz") as archive:
            for p in sorted(Path(__file__).parent.glob("*.py")):
                archive.add(p, arcname="supermimic/" + p.name)
        validation = [
            rollout(env, motions[args.heldout_clip], s, initial, "actor", np.random.default_rng(0))
            for s in (0, 100, 200)
        ]
    previous_report = json.loads((out/"report.json").read_text()) if (out/"report.json").exists() else {}
    run_report = dict(previous_report)
    run_report.update(io_schema_version=IO_SCHEMA_VERSION, reason="running", checkpoint="ckpt/final.pt",
        physics_sampler=sampler_metadata, canonical_eval_backend="native",
        config=vars(args), optimization=progress.audit(), source_sha256=source_hash)
    run_report.setdefault("manifest", dict(config=vars(args), assets=assets, source_sha256=source_hash,
        pid=os.getpid(), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), physics=physics, command=sys.argv))
    run_report.setdefault("resume_parameters", [])
    if isinstance(run_report["resume_parameters"],dict):
        run_report["resume_parameters"]=[dict(historical_records=run_report["resume_parameters"])]
    if args.resume:
        run_report["resume_parameters"].append(dict(config=vars(args), layout_migration=layout_migration,terminal_archive=terminal_archive))
    if layout_migration:
        with tarfile.open(out/"ckpt/source-layout-v2.tar.gz","w:gz") as archive:
            for p in sorted(Path(__file__).parent.glob("*.py")):
                archive.add(p,arcname="supermimic/"+p.name)
        run_report["layout_source_archive"]="ckpt/source-layout-v2.tar.gz"
    atomic_save(run_report, out/"report.json", json_format=True)
    attempt = time.time_ns()

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def save_latest():
        payload = weights_payload(actor, initial, world, critic, args, controller, progress)
        payload.update(
            train_state_version=1,
            identity=identity,
            source_sha256=source_hash,
            optimizers=dict(
                actor=actor_opt.state_dict(),
                world=world_opt.state_dict(),
                critic=critic_opt.state_dict(),
            ),
            replay=replay,
            validation=validation,
            rng=rng_state(rng),
            run_state=copy.deepcopy(state),
            initial_critic=initial_critic,
            elapsed_seconds=elapsed(),
        )
        payload["io_schema_version"] = IO_SCHEMA_VERSION
        atomic_save(payload, out / "ckpt/final.pt")
        current = next((r for r in reversed(state["eval_history"]) if r["joint_block"] == state["joint_blocks"]), None)
        metadata = dict(current or {}, joint_block=state["joint_blocks"], total_collected=state["total_collected"],
            checkpoint="ckpt/final.pt", checkpoint_kind="full_resume", io_schema_version=IO_SCHEMA_VERSION,
            checkpoint_sha256=file_hash(out/"ckpt/final.pt"))
        atomic_save(metadata, out/"ckpt/final.json", json_format=True)
        run_report.update(reason=state["reason"] or "running", elapsed_seconds=elapsed(), total_collected=state["total_collected"],
            joint_blocks=state["joint_blocks"], warm_blocks=state["warm_blocks"], optimization=progress.audit())
        atomic_save(run_report, out/"report.json", json_format=True)

    def evaluate(stage):
        stage = stage_key(stage)
        folder = out / "evals" / stage
        if (folder / "report.json").exists():
            report = json.loads((folder / "report.json").read_text())
            cached = load_weights(evaluation_checkpoint_path(folder), DEVICE)
            if any(
                not torch.equal(value, cached[key][name])
                for module, key in [(actor, "actor"), (world, "world"), (critic, "critic")]
                for name, value in module.state_dict().items()
            ):
                raise ValueError("cached evaluation weights differ from this training boundary")
        else:
            with preserve_rng(rng):
                report = evaluate_snapshot(
                    env,
                    motions,
                    actor,
                    initial,
                    world,
                    critic,
                    args,
                    controller,
                    progress,
                    folder,
                    stage,
                    checkpoint_reference=out / "ckpt" / ("final.pt" if stage == "final" else stage+".pt"),
                    total_collected=state["total_collected"],
                )
        score = report["train_auxiliary_macro"]
        row = dict(
            stage=stage,
            joint_block=state["joint_blocks"],
            path=str(folder),
            score=score,
            paper_main=report["paper_main"],
            total_collected=state["total_collected"],
            checkpoint=report["checkpoint"],
            checkpoint_sha256=report["checkpoint_sha256"],
            checkpoint_kind="full_resume" if stage == "final" else "weights_only",
            io_schema_version=IO_SCHEMA_VERSION,
        )
        if not any(x["stage"] == stage for x in state["eval_history"]):
            state["eval_history"].append(row)
            append_log(logs / "eval.jsonl", dict(attempt=attempt, **row))
        if stage == "initial":
            state["best_initial"] = row
        elif state["joint_blocks"] > 0:
            rank = lambda r: (
                r["score"]["completed_count"],
                r["score"]["macro_normalized_return"],
                -r["score"]["macro_global_mpjpe"],
                -r["joint_block"],
            )
            if state["best_trained"] is None or rank(row) > rank(state["best_trained"]):
                state["best_trained"] = row
                atomic_save(
                    weights_payload(actor, initial, world, critic, args, controller, progress),
                    out / "best.pt",
                )
                best_metadata=dict(row,checkpoint="best.pt",checkpoint_kind="weights_only",
                    checkpoint_sha256=file_hash(out/"best.pt"),evaluation_checkpoint=report["checkpoint"])
                atomic_save(best_metadata,out/"best.json",json_format=True)
        return report

    stop = dict(requested=False, signal=None)

    def request_stop(number, frame):
        stop.update(requested=True, signal=number)

    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    last_saved = time.monotonic()
    bar = tqdm(
        total=args.warmup_max + args.blocks,
        initial=state["warm_blocks"] + state["joint_blocks"],
        desc="MuJoCo training blocks",
        disable=not sys.stderr.isatty(),
        mininterval=1.0,
    )
    try:
        if not args.resume:
            evaluate("initial")
            with preserve_rng(rng):
                preflight = directional_check(actor, world, critic, validation, motions, rng)
            if not preflight["passed"]:
                raise FloatingPointError("preflight directional gradient failed")
            run_report["preflight"] = preflight
            atomic_save(run_report, out/"report.json", json_format=True)
            save_latest()
        while not state["warm_complete"] or state["joint_blocks"] < args.blocks:
            if stop["requested"]:
                state["reason"] = "signal"
                break
            if args.max_hours and elapsed() >= args.max_hours * 3600 - 120:
                state["reason"] = "wall_time_budget"
                break
            warm = not state["warm_complete"]
            block_start = time.monotonic()
            episodes = []
            if collector:
                collected_episodes,collection_metrics=collect_warp(collector,motions[args.train_clip],actor,rng,args.episodes,TASK,noisy_observation)
            else:
                collected_episodes=[]
                for _ in range(args.episodes):
                    last = len(motions[args.train_clip].states) - 1
                    start = int(rng.integers(0, last - HORIZON - FUTURE + 1))
                    collected_episodes.append(rollout(env,motions[args.train_clip],start,actor,"actor",rng,noise=0.1))
                collection_metrics=dict(backend="native",num_envs=1,active_transitions=sum(len(e["actions"]) for e in collected_episodes))
            for ep in collected_episodes:
                ep["episode_id"] = state["next_episode_id"]
                state["next_episode_id"] += 1
                replay.append(ep)
                if not collector or len(ep["actions"]) > args.replay_capacity:
                    trim_replay(replay, args.replay_capacity)
                state["total_collected"] += len(ep["actions"])
                summary = summarize(ep, env.control_dt)
                summary["episode_id"] = ep["episode_id"]
                episodes.append(summary)
            if collector:
                trim_replay_wave(replay, args.replay_capacity)
            state.setdefault(
                "reward_baseline_mean",
                float(
                    sum(e["rewards"].sum() for e in replay) / sum(len(e["rewards"]) for e in replay)
                ),
            )
            collected = time.monotonic()
            if warm or not progress.steps["world"]:
                world.fit_scaling(
                    tensor(np.concatenate([e["states"] for e in replay])),
                    tensor(np.concatenate([e["next_states"] for e in replay])),
                )
            world_losses = []
            wm_loss = train_world(
                world,
                world_opt,
                replay,
                motions,
                rng,
                args.world_updates,
                args.batch,
                progress=progress,
                warmup=warm,
                details=world_losses,
            )
            world_end = time.monotonic()
            updates = (
                []
                if warm
                else [
                    train_actor(
                        actor,
                        world,
                        critic,
                        actor_opt,
                        critic_opt,
                        replay,
                        motions,
                        rng,
                        args.batch,
                        progress=progress,
                    )
                    for _ in range(args.actor_updates)
                ]
            )
            finite_modules(actor, world, critic)
            if warm:
                state["warm_blocks"] += 1
            else:
                state["joint_blocks"] += 1
            with preserve_rng(rng):
                wm_metrics = world_error(world, validation)
                outcomes = outcome_error(
                    world,
                    validation,
                    motions,
                    state["reward_baseline_mean"],
                )
            status = dict(
                attempt=attempt,
                phase="warmup" if warm else "joint",
                warm_block=state["warm_blocks"],
                joint_block=state["joint_blocks"],
                total_collected=state["total_collected"],
                resident_transitions=sum(len(e["actions"]) for e in replay),
                resident_episodes=len(replay),
                collection=collection_metrics,
                physics_sampler=sampler_metadata,
                canonical_eval_backend="native",
                world_loss=wm_loss,
                world_losses=world_losses,
                actor_updates=updates,
                episodes=episodes,
                heldout_world=wm_metrics,
                heldout_outcomes=outcomes,
                optimization=progress.audit(),
                elapsed_seconds=elapsed(),
                timing=dict(
                    collect=collected - block_start,
                    world=world_end - collected,
                    other=time.monotonic() - world_end,
                ),
                transition_weighted_reward_mean=float(
                    sum(e["rewards"].sum() for e in replay[-args.episodes :])
                    / sum(len(e["rewards"]) for e in replay[-args.episodes :])
                ),
                episode_macro_normalized_return=float(
                    np.mean([e["normalized_return"] for e in episodes])
                ),
            )
            if warm:
                status["actor_unchanged"] = all(
                    torch.equal(v, initial.state_dict()[k]) for k, v in actor.state_dict().items()
                )
                status["critic_unchanged"] = all(
                    torch.equal(v, initial_critic[k]) for k, v in critic.state_dict().items()
                )
                if state["warm_blocks"] >= args.warmup_blocks:
                    groups = wm_metrics["groups"]
                    gate = (
                        all(
                            groups[k]["rmse"] < groups[k]["persistence_rmse"]
                            for k in ("root_position_m", "joint_position_rad")
                        )
                        and outcomes["reward_rmse"] < outcomes["reward_constant_mean_rmse"]
                    )
                    status["warmup_gate_passed"] = gate
                    if gate or not args.warmup_gate:
                        state["warm_complete"] = True
                        with preserve_rng(rng):
                            fd = directional_check(actor, world, critic, replay, motions, rng)
                        if not fd["passed"]:
                            raise FloatingPointError("warmup gradient check failed")
                        status["directional_derivative"] = fd
                        evaluate("warmup-complete")
                        bar.total = state["warm_blocks"] + args.blocks
                    elif state["warm_blocks"] >= args.warmup_max:
                        state["reason"] = "warmup_gate_failed"
            if collector:
                state["training"] = [status]  # Complete block history remains in episodes.jsonl.
            else:
                state["training"].append(status)
            append_log(logs / "episodes.jsonl", dict(record_type="block", **status))
            losses = [u["actor_loss"] for u in updates]
            critic_losses = [x for u in updates for x in u["critic_losses"]]
            actor_display = f"{np.mean(losses):.4g}" if losses else "n/a"
            critic_display = f"{np.mean(critic_losses):.4g}" if critic_losses else "n/a"
            line = f"{status['phase']} warm={state['warm_blocks']} joint={state['joint_blocks']}/{args.blocks} real={state['total_collected']} WM={wm_loss:.4g} actor={actor_display} critic={critic_display} elapsed={elapsed():.1f}s"
            bar.update(1)
            bar.set_postfix_str(line)
            tqdm.write(line)
            append_log(logs / "consol.jsonl", dict(attempt=attempt, summary=line))
            if not warm and state["joint_blocks"] % args.eval_every == 0:
                evaluate(f"joint-{state['joint_blocks']:04d}")
                history = [r for r in state["eval_history"] if r["stage"].isdigit()]
                if args.training_gates:
                    current = history[-1]["score"]
                    passes = lambda r: (
                        r["score"]["completed_count"] == 3
                        and r["score"]["macro_normalized_return"] >= 0.5
                        and r["score"]["macro_global_mpjpe"] <= 0.15
                    )
                    if len(history) >= 2 and passes(history[-1]) and passes(history[-2]):
                        state["reason"] = "task_gate_passed"
                    elif state["joint_blocks"] >= 600 and not any(
                        r["paper_main"]["completed"]
                        or (
                            r["score"]["median_steps"] >= 150
                            and r["score"]["macro_normalized_return"] >= 0.2
                        )
                        for r in history
                    ):
                        state["reason"] = "600_block_no_progress_gate"
                    elif state["joint_blocks"] >= 300 and not any(
                        r["score"]["median_steps"] >= 75
                        and r["score"]["macro_normalized_return"] >= 0.07
                        for r in history
                    ):
                        state["reason"] = "300_block_no_progress_gate"
            if (
                (state["warm_blocks"] + state["joint_blocks"]) % args.checkpoint_every == 0
                or time.monotonic() - last_saved >= 300
                or state["reason"]
                or stop["requested"]
            ):
                save_latest()
                last_saved = time.monotonic()
            if state["reason"]:
                break
        state["reason"] = state["reason"] or (
            "signal" if stop["requested"] else "joint_budget_complete"
        )
        # Save training RNG and state before diagnostics; evaluations never change the training stream.
        save_latest()
        final_stage = f"interrupt-{state['joint_blocks']:04d}" if state["reason"] == "signal" else "final"
        final_report = evaluate(final_stage)
        with preserve_rng(rng):
            final_fd = directional_check(actor, world, critic, replay or validation, motions, rng)
        save_latest()
        if final_stage == "final":
            final_report["checkpoint_sha256"] = file_hash(out/"ckpt/final.pt")
        atomic_save(final_report, out/"evals"/final_stage/"report.json", json_format=True)
        run_report.update(reason=state["reason"], elapsed_seconds=elapsed(), total_collected=state["total_collected"],
            joint_blocks=state["joint_blocks"], warm_blocks=state["warm_blocks"], best_initial=state["best_initial"],
            best_trained=state["best_trained"], optimization=progress.audit(), final_gradient_check=final_fd,
            final_evaluation=str(out/"evals"/final_stage), checkpoint="ckpt/final.pt")
        atomic_save(run_report, out/"report.json", json_format=True)
    except Exception as error:
        run_report.update(reason="failure", error=repr(error), elapsed_seconds=elapsed(),
            last_safe_checkpoint=str(out/"ckpt/final.pt"))
        atomic_save(run_report, out/"report.json", json_format=True)
        raise
    finally:
        bar.close()
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
