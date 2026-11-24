"""Fail-fast launcher for the unmodified official fixed-camera GVHMR demo."""
import argparse
import os
import subprocess
import sys
import cv2
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--video', type=Path, default=base/'inputs/hybrik_dance_0000_0400.mp4')
    p.add_argument('--output', type=Path, default=base/'runs/hybrik_dance_0000_0400/gvhmr')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--preprocess-only', action='store_true', help='Official tracking/2D pose/features only; does not recover a human or require SMPL')
    a = p.parse_args()
    capture = cv2.VideoCapture(str(a.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if abs(fps - 30) > .01:
        raise SystemExit('Official GVHMR/GMR demo assumes30fps; supply a documented CFR30 video, not a VFR nominal rate.')
    vendor = base/'vendor/GVHMR'
    required = [a.video, vendor/'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz',
                vendor/'inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl',
                vendor/'inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt',
                vendor/'inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth']
    if a.preprocess_only:
        required = [x for x in required if 'body_models' not in x.parts]
    missing = [str(x) for x in required if not x.is_file()]
    if missing:
        raise SystemExit('Missing assets (licensed SMPL/SMPL-X must be user-provided):\n'+'\n'.join(missing))
    if a.check_only:
        print('Required input/body/checkpoint files present; use actual demo to validate inference.')
        return
    env = os.environ.copy()
    # Absolute-python execution does not activate the prefix's ffmpeg on PATH.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = str(vendor) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    entry = ['tools/demo/demo.py']
    if a.preprocess_only:
        entry = ['-c', 'from tools.demo.demo import parse_args_to_cfg, run_preprocess; run_preprocess(parse_args_to_cfg())']
    subprocess.run([sys.executable, *entry, '--video', str(a.video.resolve()),
                    '--output_root', str(a.output.resolve()), '--static_cam'], cwd=vendor, env=env, check=True)


if __name__ == '__main__':
    main()
