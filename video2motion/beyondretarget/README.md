# BeyondRetarget → native G1 motion

This pipeline uses the official `bear-ty/BeyondRetarget` checkout in `vendor/`.
The independent environment is `/home/hm/.conda/envs/video2motion-beyond`.
It reuses locally available public checkpoints through read-only links; local
SHA256 values identify the files, but are not authoritative publisher checksums.
The official code/weights license is CC BY-NC 4.0; retain upstream terms.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/hm/.conda/envs/video2motion-beyond/bin/python video2motion/beyondretarget/run.py
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/hm/.conda/envs/supermimic-warp/bin/python video2motion/check_motion.py --motion video2motion/beyondretarget/runs/hybrik_dance_0000_0400/canonical/motion_source.pkl --output video2motion/beyondretarget/runs/hybrik_dance_0000_0400/canonical --render
```

The shared HybrIK demo crop is 120 frames at CFR30, local research only; see
`../GVHMRplusGMR/inputs/source.json`. It is not redistributed. Full phase-all
inference has run successfully on that input, including official filtering,
contact IK and root grounding. Both raw and official postprocessed outputs remain
under `runs/`. The exporter adds no root lift, recentering, clipping or wrist zeros.

G1's executable `model_to_xml_dof` scatter defines the network-index → XML-joint
mapping. The upstream YAML's `model_dof_order` labels conflict with this scatter;
`conversion.json` records that conflict and every actual named mapping. Output is
29 radians, root metres, XYZW quaternions and source FPS. `source_qpos.npz` also
preserves the source-model free-pose WXYZ qpos and network DoFs.

The target model shares named joint axes and pelvis frame, but four static link
positions differ from the source robot. Native QC therefore reports target-model
geometry, rather than claiming identical source and target skeleton positions.
For this clip: 199 states at 50Hz (3.96s span), finite FK; minimum physical foot
surface −0.02030m, maximum joint-limit excess 0.003683rad. These are reported, not
corrected. The reference MP4 is kinematic only, not proof of physical feasibility
or policy tracking. Fixed-camera monocular root trajectories remain estimates.

Current resource storage: `../shared_assets/beyondretarget/` holds the three neural checkpoints and source G1 robot assets; vendor assets use local shared links. `migration-manifest.json` there records the byte/hash-preserving migration from the removed outer workspace checkout. Historical run manifests retain their original acquisition paths as provenance, not runtime dependencies.
