# vibed_mujoco_supermimic

A small standalone implementation of the training loop in
[SuperMimic](https://arxiv.org/abs/2610.28480): online simulation, a learned
world model trained on consecutive transitions, deterministic motion encoding,
and policy gradients through imagined trajectories. The current full-recording
path uses stock MuJoCo Warp GPU collection, with native CPU MuJoCo evaluation
of a floating-base G1 on flat ground; networks support CPU/CUDA where applicable. This project is
independent of the BFM/HUSKY runtime and implements no skateboarding, HOI, or new
foot-contact objective.

## Current result and entry points

The current full40 implementation uses paper-sized hidden layers and persistent
2,048-lane GPU collection with 64-control-step chunks. See
[recorded results, two curves and three representative videos](docs/results/README.md).
Paper-level performance has **not** been achieved. All40 LAFAN1 recordings were
used for training and in-distribution evaluation; two short BONES examples do not
establish broad generalization. [notes.md](notes.md) is the author's historical
log, with language edits only; measured results and technical corrections are
kept separately in the result page.

Use [Complete LAFAN1 multi-recording run](#complete-lafan1-multi-recording-run)
for the current training command and `configs/nightly-full40-warp.json` for the
recorded configuration. Older sections below describe earlier experiments.
Model XML/meshes, source motion data, checkpoints and local run logs are not
bundled. Obtain compatible G1 assets and motion data separately under their
upstream terms. Public source revisions are in
[dataset/public-sources.json](dataset/public-sources.json).

### Portable setup

Python3.11+ is required. `python -m pip install -e .` installs the native path;
GPU physics uses the optional `[warp]` dependencies and a compatible CUDA-enabled
PyTorch build. Tested versions are recorded in `configs/warp-environment.json`
(MuJoCo/MJWarp3.11.0, Warp1.17.0, PyTorch2.7.0+cu128). Existing host paths in
configs and historical commands identify the original machine; replace them
with your own paths. Do not assume a generic G1 XML has the required joint/motor
mapping. The adapter validates names, transmissions and clock settings.

[Third-party notices and original license texts](third_party/README.md) are
preserved. No project-wide license is assigned by these notices.

## Historical single-clip/native usage

Python 3.11+. Install with `python -m pip install -e .`. The current machine's
`/home/hm/.conda/envs/skatebfm/bin/python` already provides runtime and test
requirements. The robot model remains external. The trusted joblib motion
library is stored in the project dataset directory; see dataset/provenance.json:

```bash
cd /home/hm/workspace/vibed_mujoco_supermimic
PYTHONPATH=src /home/hm/.conda/envs/skatebfm/bin/python -m supermimic \
  --model /home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml \
  --motions /home/hm/workspace/vibed_mujoco_supermimic/dataset/processed/lafan_29dof_10s-clipped.pkl \
  --controller configs/g1_controller.json --device cuda \
  --warmup-blocks 5 --blocks 30 --episodes 16 --world-updates 100 \
  --actor-updates 10 --batch 64 --width 128 --seed 7 --action-limit 20 \
  --output logs/YYYY-MM-DD_HH-MM-SS
```

Use `--device cpu` without CUDA and choose a fresh output directory per run.
`--warmup-blocks 1 --blocks 1 --episodes 4 --world-updates 2 --actor-updates 1
--batch 4 --width 32` provides a smaller smoke profile. The pure world-model
warmup updates neither actor nor critic; exact parameter comparisons are saved.

Evaluate a saved checkpoint without updating weights:

```bash
PYTHONPATH=src /home/hm/.conda/envs/skatebfm/bin/python -m supermimic.eval \
  --checkpoint logs/YYYY-MM-DD_HH-MM-SS/ckpt/final.pt \
  --output /tmp/supermimic-independent-eval --device cuda
PYTHONPATH=src /home/hm/.conda/envs/skatebfm/bin/python -m pytest tests -q
```

Evaluation snapshots contain `report.json` and one pickle-free `trajectories.npz`
bundle. Checkpoints are referenced from the run's `ckpt/` directory; `final.pt`
is complete resumable state, while intermediate evaluation weights and `best.pt`
are compact. Every path below is inside `logs/YYYY-MM-DD_HH-MM-SS/`.
Set `SUPERMIMIC_MODEL` and `SUPERMIMIC_MOTIONS` for integration tests on another
machine. Asset-dependent tests skip when resources are unavailable.

## Contracts and local choices

The clock is 50 Hz with four 5 ms physical substeps. The adapter resolves each
joint's torque motor by transmission, recomputes limited PD at every substep,
and leaves six root-force motors at zero. Quaternions use WXYZ at runtime;
source XYZW rotations are SLERP-resampled from 30 Hz. Public state71 contains
world root position, quaternion, 29 joint positions, world root linear velocity,
body root angular velocity, and joint velocities. Dataset joint order is explicit.

The motion encoder receives ten future reference frames; its tanh latent has
32 dimensions. The decoder receives the latent and ten observation frames and
has a linear output. The default ±20 execution protection is an implementation
choice, not a published paper action bound. Independent action noise is added
before clipping; replay/history retain executed values. Real, imagined and direct
reference-PD actions use the same execution protection. The physical torque
limits remain authoritative. A zero action holds the configured default pose.

World-model input removes root xy and heading. Its 70D targets represent local
translation, rotation-vector increments, local linear/angular velocity increments,
and joint increments. Repeated predictions integrate into the public state.
Training uses eight consecutive transitions without teacher forcing. Imagination
uses 32 steps, a frozen world model with input derivatives, differentiable soft
survival, and end-value bootstrap. The critic uses imagined GAE.

One differentiable FK derives body transforms, joint axes/anchors/references and
velocities from the compiled MuJoCo model. Actual native body fields and imagined
FK fields feed the same analytic tracking kernels. The retained evaluation task
tracks all 40 non-world bodies in world coordinates and hard XML joint limits;
these are local choices where the paper does not specify a body set or frame.
Real termination remains hard; imagined survival uses sigmoid temperatures .02 m
for height error and .04 for projected-gravity error. There are no learned reward
or termination heads. Their diagnostic RMSE/Brier fields now describe the task
applied to predicted states, not a learned classifier's calibration.

The local-delta design, differentiable kinematics and soft termination were checked
against the reference implementation at
[supermimic_vibed commit 0525a23](https://github.com/zita-ch/supermimic_vibed/tree/0525a23a2dcfa61f8543f98b84c075fe66d9d286).
They are implementation choices beyond the paper's abstract interface. The code
here implements the compiled-model contract locally; it does not copy that
repository's fixed 14-body or heading-aligned reward definition. The reference
repository carries its own MIT license. The paper does not publish exact low-PD
gains or action scales; the supplied controller is an attributed adaptation of
[mjlab G1 constants](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py).
Physics is fixed and widths/budgets are reduced. This is not a claim to recover
unpublished author parameters or the paper's full-scale performance.

Checkpoint schema3 explicitly rejects earlier raw-state/learned-head checkpoints
and legacy tanh action semantics. Old reports and outputs are retained; the
pre-change source snapshot was local to the validation session and is not repository evidence.

## Historical mini-run observations (artifacts not bundled)

The fixed seed7, five-warmup plus thirty-joint-block CUDA run produced 24,033 real
transitions in 286 seconds, with 3,500 world-model updates (500 warmup) and 300
actor updates. All 24 evaluation episodes still fell. Compared with the previous
trained model, all six matched starts survived longer and each clip's average
MPJPE decreased. Compared with this run's initial actor, mean MPJPE increased on
both clips. This establishes partial improvement over the earlier failed model,
not stable imitation or generalization. Training uses subject 1; subject 2 is held
out, each evaluated at three fixed phases. Those historical local outputs are
not distributed; the old `outputs/local-analytic-warm5-budget30` path is not
available evidence in this repository. Current recorded results are indexed above.
Mixed-unit state RMSE is labeled; use per-group physical-unit RMSE for interpretation.
Existing joblib data triggers upstream NumPy 2.5 deprecation warnings on load.

## Saved-state visualization

The default watcher produces only actual/WM/reference overlays, with white actual,
orange predicted and purple reference geometry in shared native world coordinates.
WM prediction starts from one actual state and recursively applies the same recorded
executed actions for at most32 steps. Reference frames use the same clip and reset
index. Start200 is explicitly a reset at reference4s, not survival from frame0.

```bash
PYTHONPATH=src python tools/eval_video.py \
  --run logs/YYYY-MM-DD_HH-MM-SS/evals/0050 --start 0
PYTHONPATH=src python tools/watch_eval_videos.py \
  --run logs/YYYY-MM-DD_HH-MM-SS --from-block 50
PYTHONPATH=src python tools/plot_training_curves.py \
  --run logs/YYYY-MM-DD_HH-MM-SS --through-block 50
PYTHONPATH=src python tools/eval_log_summary.py \
  --run logs/YYYY-MM-DD_HH-MM-SS --stage 0050
```

Videos, prediction arrays and one manifest are written to
`visuals/0050/start0000/` or `start0200/`. The root of `visuals/` contains only the
latest `training-curves.png` and `training-curves-data.json`. There are no automatic
screenshots, montage, geometry sidecars, SVG or evaluation-log copies. Numeric
geometry assertions and complete video decode checks still run. The summaries
print to the terminal. Source trajectories are restored with `mj_forward`, not
new physics steps; playback speed and any final image hold are labeled.

Initial baselines remain in numerical evaluation and curves, but initial-versus-
trained videos are not generated by default. An explicit `--view policy` in
`python -m supermimic.render` remains available when that comparison is requested.
Its default `--view world` shows reference, actual and predicted three-panel views.
For a reference-only1x kinematic video use `tools/reference_video.py --run RUN`;
this has no policy rollout and clearly labels its camera following behavior.

## Optimization settings and iteration accounting

New training runs clip actor gradients by global norm 1 first, then elementwise
to [-.1,.1]. All three Adam optimizers use the shared schedule
`lr(n)=2e-4 * [.03 + .97*.5*(1+cos(pi*min(n,250000)/250000))]`. It starts at
2e-4, is 1.03e-4 at n=125000, and stays at 6e-6 from n=250000 onward.

Here n is the number of completed joint actor updates, a local logical-iteration
choice. WM-only warmup uses constant 2e-4 and does not advance n. The existing
block ordering (W world updates, then A actor updates) is retained: world step j
uses `n_block_start + floor(j*A/W)`; actor update i and its 20 critic steps share
`n_block_start+i`. Default W/A=100/10 corresponds to ten world updates per logical
actor iteration. This batching is not the paper's strict alternation, and the
paper's total 320k iterations is not imposed as this CLI's training budget.

The actor's Adam betas remain (.9,.95). WM/critic retain PyTorch defaults
(.9,.999); those values are not claimed as disclosed paper parameters. Report
and schema3 checkpoint metadata store schedule provenance, logical iterations,
actual optimizer step counts, last-used rates and the next joint rate separately.
Optimization metadata alone is not a resumable checkpoint; complete training state is described below. Evaluation performs zero optimizer
updates and preserves the source checkpoint's historical optimization metadata;
older checkpoints lacking it are marked unknown rather than relabeled cosine.

A historical small verification run (not included in this repository) was
only a numerical/pipeline check: one warmup plus one joint block, W=20/A=2,
40 total WM steps (20 warmup), 2 actor and 40 critic steps. Earlier trained
results and videos were not retrained with these settings, and no behavioral
improvement is established by this change.

## Resumable single-clip training and run files

Use a date-and-time run name, for example `logs/2026-11-05_10-56-46/`.
The directory layout is:

```text
logs/YYYY-MM-DD_HH-MM-SS/
  report.json
  best.pt
  best.json
  ckpt/initial.pt, initial.json, 0050.pt, 0050.json, final.pt, final.json, source.tar.gz
  logs/episodes.jsonl, eval.jsonl, consol.jsonl
  evals/initial/, warmup-complete/, 0050/, final/
  visuals/training-curves.png, training-curves-data.json, 0050/start0000/, 0050/start0200/
```

Each `evals/STAGE/` contains only `report.json` plus `trajectories.npz`. Bundle keys
are `case_ids`, `case_hashes`, `io_schema_version`, then `<case-id>/<original-field>`;
a case ID is e.g. `walk1_subject1_clip0-start0-trained`. Arrays load with
`allow_pickle=False`; all24 evaluated cases, reward terms and actual states/actions
are retained. Each report row names `trajectory_key` and `trajectory_bundle`.
Stage names are `initial`, `warmup-complete`, a four-digit joint iteration, or
`final`; a prior interrupted/continued terminal boundary is `interrupt-NNNN`.

`episodes.jsonl` has one `record_type="block"` row per warm/joint block. It retains
all episode summaries and every WM/actor/critic update loss, without duplicated
flat episode rows. `eval.jsonl` records evaluated boundaries; `consol.jsonl` stores
plain tqdm summaries and video-worker events. Reward terms include their weights
and dt already. Normalized return divides by the full planned remaining-reference
maximum, retains negative values, and penalizes early termination. Primary score is
frame0 full-reference evaluation; resets100/200 and MPJPE are local diagnostics.

The atomic rolling `ckpt/final.pt` contains models, three Adam states, replay,
fixed validation episodes, normalization, RNG states, counters and elapsed budget.
Intermediate evaluated checkpoints and root `best.pt` are weights-only. Their JSON
sidecars include stage/joint block, total collected transitions, metrics and kind.
The `resumable=false` field inside an optimization-audit subdictionary describes
that audit alone, not a full checkpoint containing `train_state_version`.

Resume with the same critical settings and `--resume
logs/YYYY-MM-DD_HH-MM-SS/ckpt/final.pt`. Source/asset/learning configuration checks
remain strict. Before continuing an existing run, its previous terminal evaluation,
matching compact weights and visual references are archived as `interrupt-NNNN`;
new final state cannot overwrite the meaning of an earlier evaluated boundary.
No separate latest/previous/manifest/preflight files are written. Root `report.json`
combines manifest/source/assets/configuration, explicit resume parameters, total
transitions, optimization counters, preflight/final gradient checks and stop reason.
Source snapshots remain inside `ckpt/` for reproducibility.

A one-time legacy layout migration requires explicit `--allow-layout-resume`.
`configs/layout_resume_bridge.json` pins the single approved old source SHA, exact
new destination source SHA and unchanged learning-kernel digest. Any future source
change invalidates that bridge; identity and full-state checks are not bypassed.
Legacy files may be read, but future writes use this layout. Training data and its
preparation/acquisition records live under `dataset/`, not run-specific audit folders.

## Optional GPU physics

The tested isolated environment is `/home/hm/.conda/envs/supermimic-warp`, cloned
from `skate-sim-mjlab`; the original environments were not upgraded. Exact tested
versions are recorded in `configs/warp-environment.json`. Stock MuJoCo Warp is
used directly; no skateboard controller or contact-order patches are imported.

A short CUDA pipeline (not a performance-training run) is:

```bash
PYTHONPATH=src /home/hm/.conda/envs/supermimic-warp/bin/python -m supermimic \
  --model /home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml \
  --motions dataset/processed/lafan_29dof_10s-clipped.pkl \
  --controller configs/g1_controller.json --output logs/YYYY-MM-DD_HH-MM-SS \
  --device cuda --physics-backend warp --num-envs 128 --episodes 128 \
  --width 128 --batch 64 --world-updates 2 --actor-updates 1 \
  --warmup-blocks 1 --warmup-max 1 --blocks 1 --eval-every 1
```

The batch collector uses synchronous complete-episode waves. Finished lanes are
masked out of replay until the next wave; reported valid transitions exclude
inactive lane ticks. PD is recomputed at every 5 ms substep, four times per control
step, with native joint-name mapping, torque limits and six zero root motors.
Canonical evaluation remains native MuJoCo. Backend, batch size and backend
versions enter resume identity; native replay cannot be resumed as Warp. Missing
CUDA or dependencies raises an error. Native evaluation and old compact weights
remain supported.

Stock Warp uses float32 physics and is not numerically identical to native
float64 MuJoCo. Its MULTICCD warning applies to this scene: cylinder/cylinder and
cylinder/mesh pairs produce at most one contact. Contact capacity and state
finiteness are checked; the XML is unchanged.

`tools/compare_physics_speed.py --output /tmp/gpu-vs-cpu-speed.png` measures cached
physics and real collection separately (three repeats, explicit synchronization).
On the tested RTX 4060 Ti, native collection reached 638 valid transitions/s;
Warp batches 64/128/256 reached 806/1372/2157, respectively. These production
paths can have different episode lengths. Pure physics was slower at batch 1
and 64, approximately equal at 128, and reached 1.80 times aggregate native
throughput at 1024. Aggregate throughput is distinct from per-world latency;
these measurements do not establish whole-training speedup. Initial cold CCD
compilation took approximately 60 seconds and is excluded from cached timings.

## Paper-size networks and parallel capacity

`--architecture-profile paper-1` sets world hidden layers to 2048/2048/2048,
encoder and decoder to 2048/1024/512/256, and critic to 256/128. Latent32,
history/future10, K8 and H32 remain unchanged. The local input representations
produce 17,806,724 trainable parameters. Smaller paper-ratio profiles and the
schema3-compatible `legacy --width 128` profile remain explicit options. Saved
configuration records every hidden shape; evaluation and world-model rendering
use the same factory. Architecture changes cannot resume a different architecture.

The tested safe nightly configuration is `configs/nightly-paper-warp.json`:
paper-1, 2048 physics environments/episodes per collection wave, training batch256,
W100/actor10/critic200, replay2,097,152 transitions, seed7. The replay budget holds
two full498-step waves. Warp checkpoint history retains only its latest block;
complete update and episode history remains in streaming `logs/episodes.jsonl`.
Whole-wave FIFO trimming preserves the same complete-episode suffix.

Measured standard two-block probes retained minimum free memory of 4.36 GiB at
2048 environments versus only 2.04 GiB at 2560. These measurements included two
synthetic full498-step trace copies and maximum-resident normalization stress;
the synthetic allocation does not claim a completed physical498-step rollout.
4096 was not run because the projected pressure leaves insufficient reserve.
The discrete measured point is not an absolute hardware maximum. Final numeric
results are in `configs/gpu-capacity-summary.json`. Larger environment counts
collect more data per update: the fixed W100/B256/K8 budget sampled approximately
2.37–2.54 state steps per new real transition at2048, versus8.83–9.57 at512.
This changes update-to-data exposure and does not establish better imitation.

For a fresh10-hour run, append a new Shanghai datetime output directory to:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/hm/.conda/envs/supermimic-warp/bin/python -m supermimic \
  --model /home/hm/workspace/skate-sim/assets/robot/g1/scene_29dof_freebase.xml \
  --motions dataset/processed/lafan_29dof_10s-clipped.pkl \
  --controller configs/g1_controller.json --output logs/YYYY-MM-DD_HH-MM-SS \
  --device cuda --physics-backend warp --num-envs 2048 --episodes 2048 \
  --architecture-profile paper-1 --batch 256 --world-updates 100 --actor-updates 10 \
  --replay-capacity 2097152 --warmup-blocks 5 --warmup-max 10 --warmup-gate \
  --blocks 5000 --max-hours 10 --eval-every 25 --checkpoint-every 10 \
  --threads 2 --seed 7
```

The warmup gate and numerical failure protection remain enabled; behavior-based
training gates are omitted. The5000-block value is a loop ceiling, not a promise
of5000 blocks within10 hours. Longer survival makes collection slower. The wall
budget retains the existing120-second evaluation/save reserve.

A separate serial CPU-prediction/EGL monitor generates matching loss summaries,
latest two-axis curves, and same-scene actual/WM/reference videos at resets0/200:

```bash
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYTHONPATH=src \
/home/hm/.conda/envs/supermimic-warp/bin/python tools/watch_eval_videos.py \
  --run logs/YYYY-MM-DD_HH-MM-SS --from-block 25 --poll-seconds 30
```

It follows actual saved evaluations, including25-block intervals and the final
stage, and records summaries in `logs/consol.jsonl`. It never creates new physics
rollouts. Curves stream the block log and load full final checkpoints safely.

## Complete LAFAN1 multi-recording run

The current full-dataset entry point is `python -m supermimic.multi_training`.
`configs/nightly-full40-warp.json` fixes paper-1/E2048/B256/W100/A10 and10hours.
Its source is all40 public G1 CSV recordings at revision
`07ff8845a94d59d4195a8ac04fe918d4e6338239`, not the previous single10s clip.
`dataset/prepare_lafan_full.py` preserves264705 original30Hz rows and prepares
44114050Hz states/441100 transitions with per-file hashes, native FK, velocity
reconstruction and compiled-model identity. The processed manifest is under
`dataset/processed/lafan1_full_g1/`. Full recordings last approximately102–273s.
No10s cuts, root recentering, artificial SMPL fields, or new domain randomization
are introduced. This is a traceable public retargeted dataset, not a claim of
identical unpublished author retargeting. All40 recordings are training data;
first-frame evaluation across them measures in-distribution tracking.

GPU collection persists2048 lanes and emits64control-step chunks per block.
Only real hard falls or the true end of a recording reset a lane. Previous
executed action, noisy collection history, clean replay history and cumulative
true-episode returns persist across chunks. Each replay segment carries clean
history context and up to7 prior transitions so K8 windows can cross chunks,
while H32+future10 seeds cannot cross recording ends. Segment boundaries are
not failures or successful episode completion. Replay is bounded by unique
transitions. New transitions per block are131072; the WM budget samples204800
state steps per block, an exposure ratio1.5625 before replay reuse.

The1s start-bin sampler mixes50%uniform,40%softmax difficulty,10%historical
softmax distribution. Temperature25 and gamma.99 are retained. Locally specified
conventions are: infinite-horizon upper return Gmax10, completed-episode return
credited only to its reset bin, running per-bin mean, unseen bins initialized
from the known-bin mean (all unknown→uniform), and a running mean of earlier
softmax PMFs. A chunk's PMF is frozen; partial returns never update difficulty.
The exact conventions beyond the published mixture are implementation choices.

Warmup is5–20WM-only blocks, with unchanged actor/critic. Its gate compares root
and joint K1..8 physical RMSE against persistence and analytic one-step reward
RMSE against a fixed first-collection reward mean. Fixed validation uses distinct
initial-policy episodes across these same40 recordings, saved once outside
training replay; it is not heldout-motion generalization. A failed maximum
warmup gate stops honestly. True full40 primary evaluation is deterministic,
noiseless, fixed-physics and starts each recording at frame0, running to hardfall
or its actual end. Macro return averages each recording's undiscounted return
normalized by its full planned0.1×(T−1) maximum. A separate real reset200 rollout
of `walk1_subject1` supports monitoring and is excluded from the40-recording score.

Multi-run checkpoint resume mode is explicitly
`reset_live_lanes_on_resume`: model/Adam/scales/replay/curriculum/RNG/counters are
restored, but physical solver/contact hidden state is not serialized. All live
lanes receive fresh curriculum resets with new episode IDs, and interrupted
partial-episode counts/steps are recorded without labeling failure or success.
This is statistical physical continuation, not bitexact simulation resume. The
single-recording legacy path retains its earlier recovery contract.

Use the dedicated Warp environment and a fresh datetime output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/hm/.conda/envs/supermimic-warp/bin/python -m supermimic.multi_training \
  --output logs/YYYY-MM-DD_HH-MM-SS --num-envs 2048 --chunk-steps 64 \
  --architecture-profile paper-1 --batch 256 --world-updates 100 --actor-updates 10 \
  --replay-capacity 2097152 --warmup-blocks 5 --warmup-max 20 --warmup-gate \
  --blocks 5000 --max-hours 10 --eval-every 25 --checkpoint-every 10 \
  --threads 2 --seed 7
```

The final-evaluation reserve is at least900seconds, increasing to
1.25×441100/the slowest measured native evaluation step rate+120seconds.
Weights freeze before final evaluation; a same-weight periodic result is reused.
An unexpected host slowdown can still overrun the estimate and is reported,
without dropping recordings or calling truncated evaluation complete. Monitoring
uses the existing watcher command above, now selecting `walk1_subject1` from the
saved config. It produces same-scene white actual/orange WM/purple reference
windows at reset0/200, full40 macro-return/failure curves, and matching loss logs.
Independent full40 evaluation uses the saved checkpoint through
`python -m supermimic.eval --checkpoint ... --output /tmp/full40-eval --device cuda`
in the dedicated environment, preserving the source optimization provenance.
