"""Render saved MuJoCo trajectories and genuine checkpoint world-model predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .models import WorldModel, architecture
from .motion import JOINT_NAMES, load_motions
from .mujoco_adapter import MujocoAdapter
from .run_layout import load_case, case_digest, checkpoint_path as evaluation_checkpoint_path, load_weights, prediction_entries, checkpoint_matches


CLIPS = ("walk1_subject1_clip0", "walk1_subject2_clip0")


def digest(path):
    path=Path(path)
    if not path.exists() and path.suffix == '.npz' and (path.parent/'trajectories.npz').exists():
        return case_digest(path.parent,path.stem)
    hashed=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):hashed.update(chunk)
    return hashed.hexdigest()


def recorded_states(path):
    record = load_case(Path(path).parent, Path(path).stem)
    if not np.array_equal(record["states"][1:], record["next_states"][:-1]):
        raise ValueError("recorded trajectory is not contiguous")
    sequence = np.concatenate((record["states"][:1], record["next_states"]), axis=0)
    return sequence, record


def predict_open_loop(world, initial_state, actions, horizon=32):
    """The only real-state input is s0; every later input is the last prediction."""
    count = min(horizon, len(actions))
    state = torch.as_tensor(initial_state, dtype=torch.float32).unsqueeze(0)
    prediction = [np.asarray(initial_state).copy()]
    with torch.no_grad():
        for action in actions[:count]:
            state = world(state, torch.as_tensor(action, dtype=torch.float32).unsqueeze(0))
            prediction.append(state[0].cpu().numpy().copy())
    return np.stack(prediction)


def prediction_errors(prediction, actual):
    p, q = prediction[:, 3:7], actual[:, 3:7]
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return dict(
        root_position_cm=(
            100 * np.linalg.norm(prediction[:, :3] - actual[:, :3], axis=-1)
        ).tolist(),
        joint_rmse_rad=np.sqrt(
            np.mean((prediction[:, 7:36] - actual[:, 7:36]) ** 2, axis=-1)
        ).tolist(),
        root_rotation_deg=np.rad2deg(
            2 * np.arccos(np.clip(np.abs((p * q).sum(-1)), 0, 1))
        ).tolist(),
    )


def write_video(model, sequences, labels, output, title, subtitle, status, hold_label):
    """All panels use one fixed world camera; rendering calls mj_forward only."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    datas = [mujoco.MjData(model) for _ in sequences]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.30, 0, 0.60]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135.0, -15.0
    panel_width = 512 if len(sequences) == 3 else 640
    renderer = mujoco.Renderer(model, height=480, width=panel_width)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, title_font, small = (ImageFont.truetype(font_path, size) for size in (20, 24, 17))
    count, width = max(map(len, sequences)), panel_width * len(sequences)
    output_fps, repeat, final_hold = 25, 2, 50
    first = last = None
    try:
        with imageio.get_writer(
            str(output),
            fps=output_fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=16,
        ) as writer:
            for index in range(count):
                image = Image.new("RGB", (width, 608), (19, 25, 34))
                draw = ImageDraw.Draw(image)
                draw.text((20, 12), title, font=title_font, fill="white")
                draw.text((20, 43), subtitle, font=small, fill=(185, 199, 216))
                for column, (sequence, data, label) in enumerate(zip(sequences, datas, labels)):
                    frame = min(index, len(sequence) - 1)
                    state = sequence[frame]
                    data.qpos[:], data.qvel[:] = state[: model.nq], state[model.nq :]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    image.paste(Image.fromarray(renderer.render()), (column * panel_width, 64))
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(
                        (column * panel_width + 10, 74, (column + 1) * panel_width - 10, 108),
                        fill=(19, 25, 34),
                    )
                    draw.text((column * panel_width + 20, 79), label, font=font, fill="white")
                    line1, line2 = status(index, column, frame)
                    draw.text(
                        (column * panel_width + 20, 552), line1, font=small, fill=(194, 216, 235)
                    )
                    draw.text(
                        (column * panel_width + 20, 579), line2, font=small, fill=(255, 175, 143)
                    )
                if first is None:
                    first = image.copy()
                last = image.copy()
                for _ in range(repeat):
                    writer.append_data(np.asarray(image))
            hold = last.copy()
            draw = ImageDraw.Draw(hold)
            draw.rectangle((0, 0, width, 64), fill=(19, 25, 34))
            draw.text(
                (20, 15),
                hold_label + " | image held 2 s, no extra motion",
                font=title_font,
                fill="white",
            )
            for _ in range(final_hold):
                writer.append_data(np.asarray(hold))
    finally:
        renderer.close()
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(output), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(
        video=str(output),
        sha256=digest(output),
        physical_fps=50,
        playback_speed=0.25,
        output_fps=output_fps,
        output_frames=count * repeat + final_hold,
        video_duration_seconds=(count * repeat + final_hold) / output_fps,
        final_hold_seconds=2,
        camera=dict(
            lookat=camera.lookat.tolist(),
            distance=camera.distance,
            azimuth=camera.azimuth,
            elevation=camera.elevation,
        ),
        renderer="native MuJoCo EGL, state restoration and mj_forward only",
        decoder_verified=True,
    )


def render_pair(model_path, run, clip, output):
    paths = [run / f"{clip}-start0-{policy}.npz" for policy in ("initial", "trained")]
    records = [recorded_states(path) for path in paths]
    sequences = [x[0] for x in records]

    def status(index, column, frame):
        final = frame == len(sequences[column]) - 1
        return (
            f"t={frame * 0.02:.2f} s | "
            + ("TERMINATED / held" if final else "recorded simulation"),
            "Saved actual state; no post-terminal simulation",
        )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    metadata = write_video(
        model,
        sequences,
        ["Initial policy", "Trained policy"],
        output,
        f"MuJoCo saved rollouts | {clip}",
        "Same reset and fixed camera | 0.25x speed",
        status,
        "Final recorded states",
    )
    metadata["sources"] = [
        dict(
            path=str(p),
            sha256=digest(p),
            transitions=len(record["actions"]),
            rendered_states=len(sequence),
            physical_duration_seconds=len(record["actions"]) * 0.02,
            terminated=bool(record["dones"][-1]),
        )
        for p, (sequence, record) in zip(paths, records)
    ]
    return metadata


def render_world(model_path, run, clip, output, world, motion, env, checkpoint_path):
    path = run / f"{clip}-start0-trained.npz"
    actual, record = recorded_states(path)
    horizon = min(32, len(record["actions"]))
    if not np.array_equal(record["frames"][:horizon], record["frames"][0] + np.arange(horizon)):
        raise ValueError("source motion frame indices must be contiguous")
    actual = actual[: horizon + 1]
    actions = record["actions"][:horizon]
    predicted = predict_open_loop(world, actual[0], actions, horizon)
    frames = record["frames"][0] + np.arange(horizon + 1)
    reference = motion.states[frames]
    errors = prediction_errors(predicted, actual)
    endpoint = (
        "RECORDED TERMINATION" if bool(record["dones"][horizon - 1]) else f"H{horizon} HORIZON"
    )

    def status(index, column, frame):
        end = endpoint if frame == horizon else "open-loop window"
        timing = f"step {frame}/{horizon} | t={frame * 0.02:.2f} s | {end}"
        details = [
            "Kinematic target only; not a simulated rollout",
            "Actual recorded actions and MuJoCo states",
            f"WM-MJ: root {errors['root_position_cm'][frame]:.2f} cm | q {errors['joint_rmse_rad'][frame]:.3f} rad | rot {errors['root_rotation_deg'][frame]:.1f} deg",
        ]
        return timing, details[column]

    metadata = write_video(
        env.model,
        [reference, actual, predicted],
        [
            "Reference target (kinematic)",
            "MuJoCo actual (recorded)",
            "World model prediction (open-loop)",
        ],
        output,
        f"Reference / actual / trained world model | {clip} | 0.25x speed",
        "Actual + WM use the same recorded executed actions | WM anchored at t0 only | no resets or teacher forcing",
        status,
        endpoint + " reached",
    )
    prediction_path = output.parent / "prediction.npz"
    np.savez_compressed(
        prediction_path,
        actual_states=actual,
        predicted_states=predicted,
        reference_states=reference,
        executed_actions=actions,
        source_transition_frames=record["frames"][:horizon],
        state_reference_frames=frames,
        physical_dt=np.asarray(0.02),
        checkpoint_sha256=np.asarray(digest(checkpoint_path)),
        source_sha256=np.asarray(digest(path)),
    )
    model_buffer = np.empty(mujoco.mj_sizeModel(env.model), dtype=np.uint8)
    mujoco.mj_saveModel(env.model, buffer=model_buffer)
    metadata.update(
        dict(
            source_npz=str(path),
            source_sha256=digest(path),
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=digest(checkpoint_path),
            model=str(model_path),
            model_xml_sha256=digest(model_path),
            compiled_model_sha256=hashlib.sha256(model_buffer.tobytes()).hexdigest(),
            prediction_npz=str(prediction_path),
            prediction_npz_sha256=digest(prediction_path),
            horizon=horizon,
            physical_dt=0.02,
            physical_duration_seconds=horizon * 0.02,
            initial_anchor="one actual s0; no later true states provided to predictor",
            inference="schema3 WorldModel.forward CPU float32; stored 71D state; 70D local deltas integrated recursively",
            action_source="unchanged recorded executed actions, not raw policy outputs",
            reference_frames=frames.tolist(),
            source_total_transitions=len(record["actions"]),
            source_termination_in_window=bool(record["dones"][:horizon].any()),
            horizon_endpoint_label=endpoint,
            errors_per_state=errors,
            errors_at_steps={
                str(k): {key: value[k] for key, value in errors.items()}
                for k in [1, 8, 32]
                if k <= horizon
            },
        )
    )
    return metadata


def append_robot_ghost(model, data, scene, color):
    """Append native robot geoms only. Alter display material, never transforms."""
    first = scene.ngeom
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    native = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(
        model,
        data,
        options,
        mujoco.MjvPerturb(),
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_DYNAMIC,
        native,
    )
    native_ids = {
        g.objid: (g.type, g.dataid, g.size.copy())
        for g in native.geoms[: native.ngeom]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
    }
    mujoco.mjv_addGeoms(
        model, data, options, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
    )
    errors = dict(position_m=0.0, rotation_matrix=0.0, geometry_size=0.0)
    geometry_ids = []
    for geom in scene.geoms[first : scene.ngeom]:
        gid = geom.objid
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or model.geom_bodyid[gid] == 0:
            raise ValueError("ghost unexpectedly contains non-robot geometry")
        if geom.type != model.geom_type[gid] or (geom.type, geom.dataid) != native_ids[gid][:2]:
            raise ValueError("ghost mesh/type identity changed")
        errors["position_m"] = max(
            errors["position_m"], float(np.max(abs(geom.pos - data.geom_xpos[gid])))
        )
        errors["rotation_matrix"] = max(
            errors["rotation_matrix"], float(np.max(abs(geom.mat.reshape(9) - data.geom_xmat[gid])))
        )
        errors["geometry_size"] = max(
            errors["geometry_size"], float(np.max(abs(geom.size - native_ids[gid][2])))
        )
        geometry_ids.append(int(gid))
        geom.rgba[:] = color
        geom.matid, geom.texid, geom.transparent = -1, -1, 1
        geom.reflectance, geom.emission = 0.0, 0.15
    if max(errors.values()) > 1e-6:
        raise ValueError(f"ghost changed native geometry: {errors}")
    return dict(geometry_ids=geometry_ids, max_native_transform_error=errors)


def render_overlay(model_path, run, clip, output, predictions):
    source_manifest_path = predictions / "manifest.json"
    entries = prediction_entries(predictions)
    entry = next(e for e in entries if Path(e["source_npz"]).name == f"{clip}-start0-trained.npz")
    prediction_path = Path(entry["prediction_npz"])
    if digest(prediction_path) != entry["prediction_npz_sha256"]:
        raise ValueError("prediction NPZ hash differs from verified world-view manifest")
    if not checkpoint_matches(run, entry["checkpoint_sha256"]):
        raise ValueError("checkpoint differs from cached prediction")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with np.load(prediction_path) as saved:
        actual, predicted = (saved[k].copy() for k in ("actual_states", "predicted_states"))
        actions = saved["executed_actions"].copy()
    source_path = run / f"{clip}-start0-trained.npz"
    actual_source, record = recorded_states(source_path)
    horizon = len(actions)
    if (
        digest(source_path) != entry["source_sha256"]
        or not np.array_equal(actual, actual_source[: horizon + 1])
        or not np.array_equal(actions, record["actions"][:horizon])
    ):
        raise ValueError("cached actual/action data no longer match the real rollout")
    errors = prediction_errors(predicted, actual)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data, pred_data = (mujoco.MjData(model) for _ in range(2))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.3, 0, 0.6]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135.0, -15.0
    # Enlarge only the in-memory offscreen buffer; model physics and poses are unchanged.
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 512
    renderer = mujoco.Renderer(model, height=512, width=960)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font, font, small = (ImageFont.truetype(font_path, size) for size in (24, 20, 17))
    colors = dict(world_model=[1.0, 0.38, 0.05, 0.45])
    audits = []
    snapshots = {}
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    try:
        with imageio.get_writer(
            str(output),
            fps=25,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=16,
        ) as writer:
            for step in range(horizon + 1):
                for state, target in zip((actual[step], predicted[step]), (data, pred_data)):
                    target.qpos[:], target.qvel[:] = state[: model.nq], state[model.nq :]
                    mujoco.mj_forward(model, target)
                renderer.update_scene(data, camera=camera, scene_option=options)
                # One actual scene/floor; ghost geometry is visual only, with native world transforms.
                pred_audit = append_robot_ghost(
                    model, pred_data, renderer.scene, colors["world_model"]
                )
                floor_count = sum(
                    g.objtype == mujoco.mjtObj.mjOBJ_GEOM and model.geom_bodyid[g.objid] == 0
                    for g in renderer.scene.geoms[: renderer.scene.ngeom]
                )
                if floor_count != 1:
                    raise ValueError(f"expected one floor, found {floor_count}")
                audits.append(
                    dict(
                        step=step,
                        world_model=pred_audit,
                        floor_count=int(floor_count),
                    )
                )
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                image = Image.new("RGB", (960, 704), (19, 25, 34))
                image.paste(Image.fromarray(renderer.render()), (0, 96))
                draw = ImageDraw.Draw(image)
                draw.text(
                    (20, 10),
                    f"Same-scene overlay | {clip} | 0.25x speed",
                    font=title_font,
                    fill="white",
                )
                draw.text((20, 43), "Actual: solid white", font=font, fill="white")
                draw.text((260, 43), "WM: orange transparent", font=font, fill=(255, 155, 80))
                draw.text(
                    (20, 73),
                    "Same world coordinates / camera | t0 overlaps exactly | no position offsets",
                    font=small,
                    fill=(185, 199, 216),
                )
                endpoint = (
                    entry["horizon_endpoint_label"] if step == horizon else "open-loop window"
                )
                draw.text(
                    (20, 618),
                    f"step {step}/{horizon} | t={step * 0.02:.2f} s | {endpoint}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (20, 647),
                    f"WM-MJ: root {errors['root_position_cm'][step]:.2f} cm | joint RMSE {errors['joint_rmse_rad'][step]:.3f} rad | root rotation {errors['root_rotation_deg'][step]:.1f} deg",
                    font=small,
                    fill=(255, 175, 143),
                )
                draw.text(
                    (20, 677),
                    "Recorded executed actions; WM anchored only at t0. Ghosts have no physics or forces.",
                    font=small,
                    fill=(185, 199, 216),
                )
                if step in (0, horizon // 2, horizon):
                    snapshots[step] = image.copy()
                for _ in range(2):
                    writer.append_data(np.asarray(image))
            hold = image.copy()
            draw = ImageDraw.Draw(hold)
            draw.rectangle((0, 0, 960, 37), fill=(19, 25, 34))
            draw.text(
                (20, 9),
                "H32 window complete | image held 2 s, no extra motion",
                font=title_font,
                fill="white",
            )
            for _ in range(50):
                writer.append_data(np.asarray(hold))
    finally:
        renderer.close()
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(output))
    if frames != (horizon + 1) * 2 + 50:
        raise ValueError("decoded overlay frame count mismatch")
    # Geometry checks remain enforced in memory; no screenshot/audit sidecars.
    result = dict(entry)
    result.update(
        dict(
            video=str(output),
            sha256=digest(output),
            layout="single native MuJoCo scene with visual-only robot ghosts",
            reused_prediction_npz=str(prediction_path),
            source_world_manifest=str(source_manifest_path),
            source_world_manifest_sha256=digest(source_manifest_path),
            colors_rgba=colors,
            show_reference=False,
            world_alignment="unmodified native world poses; no recentering, translation offsets or rotation alignment",
            camera=dict(
                lookat=camera.lookat.tolist(),
                distance=camera.distance,
                azimuth=camera.azimuth,
                elevation=camera.elevation,
            ),

            single_floor_all_frames=True,
            visual_overrides=dict(
                offwidth=960, offheight=512, site_markers=False, reflections=False
            ),
            output_frames=frames,
            output_fps=25,
            video_duration_seconds=seconds,
            final_hold_seconds=2,
            preview_steps=list(snapshots),
            renderer="MuJoCo update_scene(actual) + mjv_addGeoms(DYNAMIC) for prediction",
        )
    )
    result.pop("decoder_verification", None)
    result["source_prediction_compiled_model_sha256"] = result.pop("compiled_model_sha256")
    result["compiled_hash_scope"] = (
        "Source prediction model; overlay only changes visual buffer/sites/reflections and never steps physics."
    )
    result["geometry_verification_summary"] = dict(
        frames_checked=len(audits),
        ghosts_per_frame=1,
        robot_geometries_per_ghost=len(audits[0]["world_model"]["geometry_ids"]),
        maximum_errors={
            key: max(
                a[ghost]["max_native_transform_error"][key]
                for a in audits
                for ghost in ("world_model",)
            )
            for key in ("position_m", "rotation_matrix", "geometry_size")
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip", default="walk1_subject1_clip0")
    parser.add_argument("--view", choices=["world", "policy", "overlay"], default="world")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Existing verified world-view directory for overlay; default RUNROOT/visuals/STAGE/start0000",
    )
    args = parser.parse_args()
    global CLIPS
    CLIPS=(args.clip,)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("output already contains a manifest; choose a fresh directory")
    model_path, run = args.model.resolve(), args.run.resolve()
    if args.view == "overlay":
        predictions = args.predictions.resolve() if args.predictions else run.parent.parent / "visuals" / run.name / "start0000"
        metadata = [
            render_overlay(
                model_path, run, clip, args.output / f"{clip}-same-scene-overlay.mp4", predictions
            )
            for clip in CLIPS
        ]
    elif args.view == "world":
        checkpoint_path = evaluation_checkpoint_path(run)
        checkpoint = load_weights(checkpoint_path)
        if checkpoint.get("schema_version") != 3:
            parser.error("world prediction rendering requires a schema3 checkpoint")
        config = checkpoint["config"]
        report = json.loads((run / "report.json").read_text())
        controller = report["controller"]
        env = MujocoAdapter(
            model_path,
            JOINT_NAMES,
            controller["kp"],
            controller["kd"],
            controller["q_default"],
            controller["action_scale"],
        )
        motions = load_motions(config["motions"], CLIPS, env)
        world = WorldModel(hidden=architecture(config)["world"]).eval()
        world.load_state_dict(checkpoint["world"])
        metadata = [
            render_world(
                model_path,
                run,
                clip,
                args.output / f"{clip}-reference-actual-world.mp4",
                world,
                motions[clip],
                env,
                checkpoint_path,
            )
            for clip in CLIPS
        ]
        for item in metadata:
            item["motion_library"] = config["motions"]
            item["motion_library_sha256"] = digest(config["motions"])
    else:
        metadata = [
            render_pair(model_path, run, clip, args.output / f"{clip}-initial-vs-trained.mp4")
            for clip in CLIPS
        ]
    for entry in metadata:
        if "source_npz" in entry and (run/"trajectories.npz").exists():
            entry["trajectory_key"]=Path(entry.pop("source_npz")).stem
            entry["trajectory_bundle"]=str(run/"trajectories.npz")
            entry["trajectory_source_sha256"]=entry.pop("source_sha256")
    document=dict(io_schema_version=2,videos=metadata,prediction_sources=[metadata] if args.view=="world" else [])
    (args.output / "manifest.json").write_text(json.dumps(document, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
