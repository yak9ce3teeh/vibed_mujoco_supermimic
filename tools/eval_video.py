"""Same-world-coordinate three-layer eval video, using frozen render helpers.
This peripheral presentation adapter preserves src and its running-training hash.
"""
import argparse
import json
from pathlib import Path
from supermimic.run_layout import prediction_entries, checkpoint_matches
from supermimic.render import (np, mujoco, imageio, imageio_ffmpeg, Image,
    ImageDraw, ImageFont, digest, recorded_states, prediction_errors, append_robot_ghost)

def render_overlay(model_path, run, clip, output, predictions, start=0):
    source_manifest_path = predictions / "manifest.json"
    entries = prediction_entries(predictions)
    entry = next(e for e in entries if Path(e["source_npz"]).name == f"{clip}-start{start}-trained.npz")
    prediction_path = Path(entry["prediction_npz"])
    if digest(prediction_path) != entry["prediction_npz_sha256"]:
        raise ValueError("prediction NPZ hash differs from verified world-view manifest")
    if not checkpoint_matches(run,entry["checkpoint_sha256"]):
        raise ValueError("checkpoint differs from cached prediction")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with np.load(prediction_path) as saved:
        actual, predicted, reference = (saved[k].copy() for k in ("actual_states", "predicted_states", "reference_states"))
        actions = saved["executed_actions"].copy()
    source_path = run / f"{clip}-start{start}-trained.npz"
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
    data, pred_data, ref_data = (mujoco.MjData(model) for _ in range(3))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.3, 0, 0.6]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135.0, -15.0
    # Enlarge only the in-memory offscreen buffer; model physics and poses are unchanged.
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 512
    renderer = mujoco.Renderer(model, height=512, width=960)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font, font, small = (ImageFont.truetype(font_path, size) for size in (24, 20, 17))
    colors = dict(world_model=[1.0, 0.38, 0.05, 0.45], reference=[0.65, 0.25, 1.0, 0.30])
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
                for state, target in zip((actual[step], predicted[step], reference[step]), (data, pred_data, ref_data)):
                    target.qpos[:], target.qvel[:] = state[: model.nq], state[model.nq :]
                    mujoco.mj_forward(model, target)
                renderer.update_scene(data, camera=camera, scene_option=options)
                # One actual scene/floor; ghost geometry is visual only, with native world transforms.
                pred_audit = append_robot_ghost(
                    model, pred_data, renderer.scene, colors["world_model"]
                )
                ref_audit = append_robot_ghost(model, ref_data, renderer.scene, colors["reference"])
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
                        reference=ref_audit,
                        floor_count=int(floor_count),
                    )
                )
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                image = Image.new("RGB", (960, 704), (19, 25, 34))
                image.paste(Image.fromarray(renderer.render()), (0, 96))
                draw = ImageDraw.Draw(image)
                draw.text(
                    (20, 10),
                    f"RESET to reference t={start * .02:.2f}s | auxiliary evaluation | 0.25x" if start else f"Same-scene overlay | {clip} | 0.25x speed",
                    font=title_font,
                    fill="white",
                )
                draw.text((20, 43), "Actual: solid white", font=font, fill="white")
                draw.text((260, 43), "WM prediction: orange", font=font, fill=(255, 155, 80))
                draw.text((590, 43), "Reference goal: purple", font=font, fill=(195, 140, 255))
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
                    f"step {step}/{horizon} | elapsed={step*.02:.2f}s | reference t={(start+step)*.02:.2f}s | {endpoint}",
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
    # Native geometry assertions run in memory; no audit/screenshots sidecars.
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
            show_reference=True,
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
        ghosts_per_frame=2,
        robot_geometries_per_ghost=len(audits[0]["world_model"]["geometry_ids"]),
        maximum_errors={
            key: max(
                a[ghost]["max_native_transform_error"][key]
                for a in audits
                for ghost in ("world_model", "reference")
            )
            for key in ("position_m", "rotation_matrix", "geometry_size")
        },
    )
    return result



def prepare_prediction(model_path, snapshot, clip, start, output):
    from supermimic.run_layout import checkpoint_path, load_weights
    from supermimic.models import WorldModel, architecture
    from supermimic.motion import JOINT_NAMES, load_motions
    from supermimic.mujoco_adapter import MujocoAdapter
    from supermimic.render import predict_open_loop
    checkpoint=checkpoint_path(snapshot); saved=load_weights(checkpoint)
    report=json.loads((snapshot/'report.json').read_text()); c=report['controller']
    env=MujocoAdapter(model_path,JOINT_NAMES,c['kp'],c['kd'],c['q_default'],c['action_scale'])
    motion=load_motions(saved['config']['motions'],[clip],env)[clip]
    world=WorldModel(hidden=architecture(saved['config'])["world"]).eval();world.load_state_dict(saved['world'])
    source=snapshot/f'{clip}-start{start}-trained.npz'
    actual,record=recorded_states(source);horizon=min(32,len(record['actions']))
    if not np.array_equal(record['frames'][:horizon],start+np.arange(horizon)):
        raise ValueError('Recorded reference frames are not contiguous')
    actual=actual[:horizon+1];actions=record['actions'][:horizon]
    predicted=predict_open_loop(world,actual[0],actions,horizon)
    frames=start+np.arange(horizon+1);reference=motion.states[frames]
    path=output/'prediction.npz'
    np.savez_compressed(path,actual_states=actual,predicted_states=predicted,
        reference_states=reference,executed_actions=actions,state_reference_frames=frames,
        source_transition_frames=record['frames'][:horizon])
    errors=prediction_errors(predicted,actual)
    buffer=np.empty(mujoco.mj_sizeModel(env.model),dtype=np.uint8);mujoco.mj_saveModel(env.model,buffer=buffer)
    import hashlib
    return dict(source_npz=str(source),source_sha256=digest(source),checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),model=str(model_path),model_xml_sha256=digest(model_path),
        compiled_model_sha256=hashlib.sha256(buffer.tobytes()).hexdigest(),
        prediction_npz=str(path),prediction_npz_sha256=digest(path),horizon=horizon,
        physical_dt=.02,physical_fps=50,physical_duration_seconds=horizon*.02,playback_speed=.25,
        reference_frames=frames.tolist(),reset_reference_frame=start,
        initial_anchor='one actual reset s0; no later true-state input',
        inference='schema3 WorldModel CPU float32 recursive open-loop',
        action_source='unchanged recorded executed actions',source_total_transitions=len(record['actions']),
        source_termination_in_window=bool(record['dones'][horizon-1]),
        horizon_endpoint_label='RECORDED TERMINATION' if record['dones'][horizon-1] else f'H{horizon} HORIZON',
        errors_per_state=errors,errors_at_steps={str(k):{name:v[k] for name,v in errors.items()} for k in (1,min(8,horizon),horizon)},
        motion_library=saved['config']['motions'],motion_library_sha256=digest(Path(saved['config']['motions'])))


def render_stage(snapshot, output, start=0, clip='walk1_subject1_clip0', model=None):
    from supermimic.run_layout import checkpoint_path, stage_key
    snapshot=Path(snapshot).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=json.loads((snapshot/'report.json').read_text())
    model=Path(model or report['config']['model']).resolve()
    video=output/'actual-wm-reference.mp4';manifest=output/'manifest.json'
    if video.exists() and manifest.exists():return str(video)
    if video.exists():raise FileExistsError('Video without complete manifest; review before retry')
    entries=prediction_entries(output) if manifest.exists() else []
    entry=next((e for e in entries if Path(e['source_npz']).stem==f'{clip}-start{start}-trained'),None)
    if entry is None:entry=prepare_prediction(model,snapshot,clip,start,output)
    # In-progress manifest retains exact source evidence if rendering fails.
    document=dict(io_schema_version=2,stage_key=stage_key(snapshot.name),reset_reference_frame=start,
                  prediction_sources=[[entry]],videos=[])
    manifest.write_text(json.dumps(document,indent=2)+'\n')
    result=render_overlay(model,snapshot,clip,video,output,start)
    result.pop('source_world_manifest',None)
    result.pop('source_world_manifest_sha256',None)
    result['evaluation_role']='auxiliary reset evaluation' if start else 'start0 primary episode window'
    result['presentation_adapter_sha256']=digest(Path(__file__))
    document['videos']=[result]
    # Packed cases have virtual legacy paths only inside implementation helpers.
    for item in [entry,result]:
        if (snapshot/'trajectories.npz').exists():
            item['trajectory_key']=Path(item.pop('source_npz')).stem
            item['trajectory_bundle']=str(snapshot/'trajectories.npz')
            item['trajectory_source_sha256']=item.pop('source_sha256')
    manifest.write_text(json.dumps(document,indent=2)+'\n')
    return str(video)


if __name__ == '__main__':
    from supermimic.run_layout import stage_key
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path,help='Immutable evals/STAGE folder')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--clip',default='walk1_subject1_clip0')
    parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args()
    output=args.output or args.run.resolve().parent.parent/'visuals'/stage_key(args.run.name)/f'start{args.start:04d}'
    print(render_stage(args.run,output,args.start,args.clip,args.model),flush=True)
