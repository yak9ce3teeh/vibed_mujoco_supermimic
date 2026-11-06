# Third-party notices

## SuperMimic_vibed

Implementation reference/adaptation source:
[zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed/tree/0525a23a2dcfa61f8543f98b84c075fe66d9d286),
fixed commit `0525a23a2dcfa61f8543f98b84c075fe66d9d286`.
The local heading-relative dynamics, differentiable kinematics/task construction
and rollout/survival implementation choices were checked against that source;
they are not all uniquely specified by the paper. Its full MIT notice, including
Copyright (c) 2026 ETH Zurich, Chong Zhang, is preserved in
[supermimic-vibed-LICENSE.txt](supermimic-vibed-LICENSE.txt).

## mjlab

The controller constants in `configs/g1_controller.json` adapt the G1 gain,
default-pose and action-scale conventions from
[mujocolab/mjlab](https://github.com/mujocolab/mjlab), tested with installed
version 1.6.0. The installed distribution's complete Apache License 2.0 text is
preserved in [mjlab-LICENSE.txt](mjlab-LICENSE.txt). No additional NOTICE file was
present in that installed distribution. This is not a claim that these gains
are the unpublished exact SuperMimic controller parameters.

These notices apply to their respective third-party material. They do not assign
a new project-wide license, relicense datasets or robot assets, or imply upstream
author endorsement. Source motion data, model XML/meshes and trained checkpoints
are not redistributed here.
