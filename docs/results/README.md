# Recorded results

These are local reproduction results, not paper-level performance. The run used
paper-sized hidden layers, 2,048 stock MuJoCo Warp environments, fixed physics,
and all 40 complete public G1 LAFAN1 recordings. The same 40 recordings were
used for deterministic first-frame evaluation; this is not heldout generalization.
Motion data, robot assets, checkpoints and raw training logs are not included.

## Run summary

- Best saved checkpoint: joint block 500, macro full-reference normalized return
  **0.3849349**, 23 recordings reached their reference end and 17 terminated.
- Final checkpoint: joint block 536, macro normalized return **0.2311093**,
  6 reached the reference end and 34 terminated.
- Reaching the end is not proof of accurate tracking. The termination rule checks
  reference-relative root height and gravity-z errors; horizontal drift alone
  does not terminate an episode.

The maximum return denominator is the full planned `0.1 × (T−1)`, including
unexecuted steps after early termination. Global/root-relative MPJPE and XY
errors below are additional diagnostics, not claimed paper benchmark metrics.

## Figures

[Evaluation and world-model validation curves](figures/training-curves.png)
show the complete run, not only checkpoint 500. The main score is the all-40
macro normalized return; failure fraction is shown separately.

[Training return/reward and optimizer losses](figures/training-params-curves.png)
show real episodes that ended in each block, including failures and reference
ends. They may span chunks and policy updates, with noisy random resets and a
changing curriculum. Light lines are raw block aggregates; bold lines use
EWMA alpha 0.10 with phase resets. Per-step reward is not an episode return.
WM/actor/critic losses average every update in the block; warmup has no
actor/critic update. These curves are not deterministic evaluation scores.

## Three representative videos

All videos use shared world coordinates: actual robot in white, learned world
model in orange, reference goal in purple. WM predictions reanchor at the true
state every 32 controls (0.64 s), using the same recorded actions within each
window. This is explicitly rolling prediction, not one full-episode open loop.
Playback is 1x at 25 fps, sampled from the 50 Hz recording with the true endpoint
retained and no added hold. MP4 duration can differ by at most one output frame.

- [BEST500: unseen BONES walking](videos/bones-unseen-walk-arc-best500.mp4).
  One pre-selected nonmirror `walk_arc_cw_stop_R_003__A255` recording,
  304/304 controls (6.08 s), normalized return **0.6946904**, global MPJPE
  **0.28084 m**, terminal XY error **0.309895 m**.
- [BEST500: unseen BONES dance](videos/bones-unseen-dance-retro-best500.mp4).
  One pre-selected nonmirror `dance_retro_up_down_step_R_003__A311` recording,
  366/366 controls (7.32 s), normalized return **0.5867358**, global MPJPE
  **0.40875 m**, terminal XY error **1.198857 m**. The visible drift remains
  substantial despite reaching the reference end.
- [FINAL536: training-recording jump failure](videos/lafan-jumps1-subject1-final536-failure.mp4).
  `jumps1_subject1`, 449 controls (8.98 s), terminated when reference-relative
  root-height error exceeded 0.25 m. This is an in-training recording and a
  different checkpoint from the two BEST500 examples.

The BONES recordings were not used to train this run. Each was selected before
evaluation and tested once per checkpoint, with no finetuning, root adjustment
or seed/reset search. Two short cross-dataset examples are not evidence of broad
generalization. Their retargeting distribution differs from the LAFAN1 source.

## Historical notes and implementation limits

[notes.md](../../notes.md) preserves the author's dated observations and hypotheses,
with language and Markdown edits only. They should be distinguished from measured
results. In particular, the earlier 3-hour single-clip run used **one serial native
CPU environment**; 128 was its shared network width, not a parallel-environment
count. Claims of alignment/convergence in personal notes do not establish complete
paper equivalence or paper-level performance.

Domain randomization and several undisclosed author implementation details are
not reproduced. The controller constants are a documented mjlab-derived local
choice, not confirmed exact SuperMimic gains. Stock Warp float32 collection differs
numerically from native float64 evaluation. Multi-run physical resume resets live
lanes and explicitly does not promise bitexact solver-state continuation.

Source revisions and dataset terms: [public source references](../../dataset/public-sources.json).
Third-party code notices: [third_party](../../third_party/README.md).
