"""Run unmodified official phase-all inference, then export with explicit mapping."""
import argparse
import os
import subprocess
import sys
import cv2
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=Path, default=base.parent/'GVHMRplusGMR/inputs/hybrik_dance_0000_0400.mp4')
    p.add_argument('--output', type=Path, default=base/'runs/hybrik_dance_0000_0400')
    a = p.parse_args()
    capture = cv2.VideoCapture(str(a.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if abs(fps - 30) > .01:
        raise SystemExit('This entry requires a documented CFR30 input; resample VFR/other rates using actual timestamps first.')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, 'app/infer/infer_video.py', '--video', str(a.video.resolve()),
        '--robots', 'g1', '--fps', '30', '--phase', 'all', '--output_dir', str(a.output.resolve())],
        cwd=base/'vendor/BeyondRetarget', env=env, check=True)
    prediction = a.output/'predictions/postprocess'/a.video.stem/'g1/g1_raw_pred.npz'
    subprocess.run([sys.executable, str(base/'export_motion.py'), '--prediction', str(prediction.resolve()),
                   '--output', str(a.output.resolve()/'canonical')], env=env, check=True)


if __name__ == '__main__':
    main()
