"""Validate and split a trusted, retargeted G1 joblib library into full windows.

This is offline dataset preparation. Runtime resampling/velocities/FK remain in
supermimic.motion. It does not convert raw human mocap or retarget a robot.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def prepare(source, seconds=10.0):
    result = {}
    for name, clip in source.items():
        dof = np.asarray(clip["dof"])
        pos = np.asarray(clip["root_trans_offset"])
        quat = np.asarray(clip["root_rot"])
        frames = len(dof)
        fps = float(clip["fps"])
        window = int(round(seconds * fps))
        if (
            window < 2
            or dof.shape != (frames, 29)
            or pos.shape != (frames, 3)
            or quat.shape != (frames, 4)
        ):
            raise ValueError(f"{name}: invalid G1 shapes/fps/window")
        if not all(np.isfinite(x).all() for x in (dof, pos, quat)) or not np.allclose(
            np.linalg.norm(quat, axis=1), 1, atol=1e-4
        ):
            raise ValueError(f"{name}: nonfinite state or nonunit XYZW quaternion")
        for number, start in enumerate(range(0, frames - window + 1, window)):
            key = f"{name}_clip{number}"
            result[key] = {
                field: (
                    np.asarray(value)[start : start + window].copy()
                    if hasattr(value, "shape") and len(value.shape) > 0 and len(value) == frames
                    else value
                )
                for field, value in clip.items()
            }
            result[key]["motion_name"] = key
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to overwrite existing dataset")
    clips = prepare(joblib.load(args.input), args.seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clips, args.output)
    metadata = dict(
        input=str(args.input.resolve()),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        output=str(args.output.resolve()),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        seconds=args.seconds,
        clips=len(clips),
        remainder="discarded incomplete final window",
        quaternion="XYZW",
        joint_order="declared G1 29DoF source order",
        source="already-retargeted G1 library, not raw human mocap",
    )
    args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
