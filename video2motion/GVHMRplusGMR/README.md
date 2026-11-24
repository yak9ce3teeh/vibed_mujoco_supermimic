# GVHMR → GMR → native G1 motion

Official source checkouts live in `vendor/GVHMR` and `vendor/GMR`; run artifacts,
inputs and licensed models remain local and ignored by Git. The two independent
prefixes are `/home/hm/.conda/envs/video2motion-gvhmr` (Python3.10) and
`/home/hm/.conda/envs/video2motion-gmr` (Python3.11). Existing training environments
are not upgraded. GMR imports and its named29-hinge G1 model load were tested.

Required user-obtained licensed files:

- `vendor/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`
- `vendor/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`
- GMR accepts the same licensed SMPL-X directory via `--body-models`.

No unlicensed body-model mirror is used. Register/download under the official
SMPL/SMPL-X terms, then place or link the files at those paths. Public neural
checkpoints are separate from body models; source acquisition records identify
their origin and SHA256. Missing files trigger a clear failure before inference.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/hm/.conda/envs/video2motion-gvhmr/bin/python video2motion/GVHMRplusGMR/run_gvhmr.py --check-only
PYTHONDONTWRITEBYTECODE=1 /home/hm/.conda/envs/video2motion-gvhmr/bin/python video2motion/GVHMRplusGMR/run_gvhmr.py
PYTHONDONTWRITEBYTECODE=1 /home/hm/.conda/envs/video2motion-gmr/bin/python video2motion/GVHMRplusGMR/retarget.py --prediction video2motion/GVHMRplusGMR/runs/hybrik_dance_0000_0400/gvhmr/hybrik_dance_0000_0400/hmr4d_results.pt --body-models video2motion/GVHMRplusGMR/vendor/GVHMR/inputs/checkpoints/body_models --output video2motion/GVHMRplusGMR/runs/hybrik_dance_0000_0400/gmr
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/hm/.conda/envs/supermimic-warp/bin/python video2motion/check_motion.py --motion video2motion/GVHMRplusGMR/runs/hybrik_dance_0000_0400/gmr/motion_source.pkl --output video2motion/GVHMRplusGMR/runs/hybrik_dance_0000_0400/gmr --render
```

The sole selected input is HybrIK's official example dance, source seconds0–4,
CFR30/120frames with full original framing. Media license is not established;
local research only, not distributed. See `inputs/source.json`.

`retarget.py` calls official GMR APIs without a viewer, preserves all frames
(including frame0), saves original full qpos and maps every native hinge by name.
The official GVHMR→GMR helper rotates human coordinates by Rx(+90°): x,−z,y;
configured human scaling, target offsets and IK remain upstream behavior. Export
only changes named joint ordering and WXYZ→XYZW storage, without adding root or
joint corrections. Converted motion stores actual aligned FPS; native loader
resampling to50Hz is a separate, documented step. A finite reference replay is
kinematic validation, not a dynamics or tracking result.

Actual completed smoke: the same120-frame input passed official GVHMR human recovery and headless GMR retargeting, using the user-provided licensed neutral files. Official preprocessing took20.62s and cached human prediction took0.45s. GMR preserved all120 source frames at30Hz; native resampling produced199 states at50Hz (3.96s sample span). `runs/hybrik_dance_0000_0400/gmr/` contains original qpos, canonical motion, conversion/QC and reference.mp4. Minimum target foot collision-surface height is−0.04439m; peak joint speed16.806rad/s. Joint limits hold to numerical rounding. These issues are reported without manual root/joint correction; this is kinematic validation, not physical feasibility or policy tracking. `--preprocess-only` remains available for non-SMPL preprocessing. The launcher explicitly adds its environment bin directory to PATH so the official final merge can find ffmpeg.

Environment-specific fixes: install chumpy with `--no-build-isolation`; pin `setuptools==80.9.0` because Lightning2.3 still imports pkg_resources. Torch2.3/CUDA12.1, torchvision0.18 and PyTorch3D0.7.6 were cloned from the compatible existing environment, not upgraded in place. The actual package versions and fixed vendor revisions are in the two environment JSON files.

The HMR2 checkpoint is shared locally at `../shared_assets/beyondretarget/hmr2/`; the GVHMR vendor link resolves there. The removed outer BeyondRetarget workspace is not required by either environment or runner. Historical acquisition paths remain provenance only.

Both launchers reject non30fps nominal input. The entry contract requires actual CFR30; this is not a VFR timestamp validator. The selected input has documented timestamp-based CFR preprocessing in inputs/source.json. Vendors, inputs, licensed assets, shared_assets and generated runs are Git-ignored; only the self-written entries and explanations are intended as project source.
