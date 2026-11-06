# Local dataset workspace

The opening section records historical single-clip inputs. Current full40
preparation is documented under **Complete40-recording LAFAN1 simulator cache**
below. Data payloads, local inventories/access records and generated per-motion
metadata are excluded from the repository; public source facts are in
[public-sources.json](public-sources.json).

`raw/lafan_29dof.pkl` is an already-retargeted G1 motion library, not raw human
mocap. `processed/lafan_29dof_10s-clipped.pkl` is the existing 10-second-window
library copied byte-for-byte from the previous source. File sizes, hashes and
copy provenance are recorded in `provenance.json`. The historical single-clip training run used
that verified processed copy; no new slicing or retargeting was performed.

`prepare.py` is an independent offline utility: validate finite 29-joint states,
metre root translations and unit XYZW quaternions, then retain complete,
non-overlapping windows at the source fps. It writes a new joblib file and
manifest and refuses overwrites. Its naming/window policy is explicit and is
not a claim to reconstruct the historical processed library byte-for-byte.

```bash
/home/hm/.conda/envs/skatebfm/bin/python dataset/prepare.py \
  --input dataset/raw/lafan_29dof.pkl \
  --output dataset/processed/new-windows.pkl --seconds 10
```

Do not replace a historical run's frozen processed input. Runtime 30-to-50Hz
SLERP, SI/WXYZ state conversion, velocities and model-dependent FK remain in
`src/supermimic/motion.py`; this directory is not a generic Python package.

Verified copy identities (also machine-readable in provenance.json):

- Raw:209,659,488 bytes; SHA256 `f3a0c2810363f5c50bf4146fa2db33c1ff5b90d00cb7c0bc2aa4622696375e11`.
- Processed:205,117,479 bytes; SHA256 `7f5aa36957808ee2e972472b18add8510533742710ba312d8b8c6e6014f1c010`.

These identify copied inputs, not freshly generated output from prepare.py.

## Artifact catalogs and subset scopes

`validate_catalog.py` is an independent, read-only artifact/manifest checker.
It does not alter the fixed dataset used by the running trainer or convert an
unknown external format. Root `catalog.json` records source/license evidence,
local availability, byte count/SHA256, known parent IDs and explicit schema facts.
Unavailable or unknown fields stay marked unknown; metadata does not imply data
have been acquired. A gated BONES source is not a fabricated local clip set.

```bash
/home/hm/.conda/envs/skatebfm/bin/python dataset/validate_catalog.py \
  --catalog dataset/catalog.json --report dataset/catalog.validation.json
```

Artifact and clip-manifest paths are project-root relative (`dataset/...`).
For a file, `fingerprint(path)` returns plain SHA256 and bytes. For a directory,
it excludes only `.git`, includes license and other hidden files, sorts relative
POSIX paths and hashes compact UTF-8 JSONL rows `[path,file_sha256,file_bytes]`;
reported bytes are the file-byte sum. The helper is exported for catalog writers.

Each subset has a distinct `scope_id`, with JSONL clip rows containing clip_id,
artifact_id, original recording_id, subject, skill, source_start/end_exclusive
(half-open source-frame interval), fps, num_frames, split and asset_hash. The
asset hash is robot/asset identity, not the container hash; it may be unknown for
historical retargeting. Schema units, rotation representation/order/frame and
velocity reconstruction are recorded only when known.

Within one scope every original recording belongs to one split, so overlapping
and adjacent windows cannot leak across train/dev/test. Scopes are independent:
a single-clip fit, walking-only subset and full-library coverage are different
experiments. Training full-library scope can include recordings used as holdout
in walking scope, so it must not borrow that scope's blind-test claim. Subject2
has already been inspected repeatedly and cannot be called blind. Without
author-provided filter IDs and evidence, no subset is labeled paper_filtered_exact.

## Rebuild the derived catalog

The active training process is unchanged and still reads the frozen historical
processed G1 library. The new data below are inventories for later experiments,
not a silent replacement of its input.

`build_catalog.py` reads the two downloader-owned `acquisition.json` files and
existing payloads, generates `catalog.json`, `manifests/index.json`, scope JSONL
files and schema/slice validation reports, then runs `validate_catalog.py`.
Rerunning it refreshes acquisition status; it never edits the acquisition JSON,
binary datasets or trainer source. Optional dependencies are declared as the
`data` extra (`huggingface_hub`, `pyarrow`); install those in the intended data
workspace when needed, not by modifying the running training environment.

```bash
cd /home/hm/workspace/vibed_mujoco_supermimic
# With optional data dependencies available:
PYTHONPATH=src /home/hm/.conda/envs/skatebfm/bin/python dataset/build_catalog.py
```

LAFAN scopes are deliberately different:

- `lafan1_single_walk10s`: one train clip and one repeatedly inspected dev clip,
  no test. This is the current fit task.
- `lafan1_walking_family`: only recording names matching `^walk[1-4]_subject`;
  subjects1/3/4 give183 train cuts from8 recordings, subject2 gives50 dev cuts
  from2 recordings, and subject5 gives50 `reserved_unscored` test cuts from2
  recordings. Schema and slice metadata have been loaded; no pristine-unseen
  or blind-policy-evaluation claim is made.
- `lafan1_full40_coverage`:40 historical retargeted recordings, nominal2.450972 h,
  diagnostic inventory only. `lafan1_training_pool862` is a separate unassigned
  cut pool, nominal2.394444 h. They include walking holdouts, so a model trained
  on them cannot claim the walking scope's holdout separation.

All862 historical cuts were checked against their raw G1 parent for exact
root-translation, joint-position and quaternion slices. This does not recover
the unknown original retargeter or robot asset revision. The independently
acquired official77 BVH files and public40 G1 CSV files remain separate artifacts.
Public CSVs passed36-column, finite-value and unit-quaternion checks; their card
at revision `07ff8845a94d59d4195a8ac04fe918d4e6338239` declares30fps, XYZW and
29-joint order. Undeclared units/frame remain unknown. Public CSVs are declared
LAFAN1 derivatives, but their exact original archive revision is not established;
only `related_sources`, not an exact local-ZIP parent link, is recorded.

BONES metadata at revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66` has142220
unique G1 paths:71132 originals and71088 mirrors, grouped into71132 parent IDs
by removing only the terminal `_M` from mirrored basenames. There are71088 pairs
and44 original-only groups;7653 pairs have different metadata frame counts.
Do not require equal frame counts or treat these as validated G1 CSV lengths.

The `full_metadata`, `original`, `mirrored` and `LocomotionWalking` pools are all
`metadata_indexed` and split `unassigned`. Walking means exactly package
`Locomotion` and lower-trimmed movement string `walking`:15946 rows/7980 groups.
Every BONES row explicitly labels `source_frame_domain=source_motion_120hz`;
120fps and `move_duration_frames` describe source metadata, not G1 CSV duration,
fps, row count or physical compatibility. Data payload availability is separate
in the catalog and follows the downloader's verified acquisition status.

Three strictly nested local samples select32/512/4096 parent groups, including
all available mirror/original metadata in each group (64/1023/8191 rows). They
are labeled `local_package_balanced_seed7`: within each of8 packages rank parent
IDs by SHA256 of `revision|7|original_full_move_g1_path`, then take4/64/512 groups
per package. These are equal-package samples, not natural-frequency datasets,
held-out splits or author-filtered paper subsets. Complex movement labels remain
whole labels. Group-selection hashes and the exact rule are saved in catalog.
No CSV is duplicated to materialize a pool; manifests reference source paths.

The paper's129514 motions/264.56 h and CuboiLeo's129491 count are separately
`published_count_only`. Without author filter IDs/version mapping, no generated
scope claims to be that exact filtered set. All nominal hours in manifests are
source `num_frames/fps`, not measured policy control time.

The BONES download uses the official HF client and default local credentials;
no token/account contents enter catalog or logs. A durable version of the active
downloader is `acquire_bones_seed.py`; only an explicit authorization guard and
portable dataset-root path were added, and provenance is saved in
`manifests/bones_seed/download_code_provenance.json`. It was not executed again
while the existing downloader was active. Future authorized acquisition/resume
can use `python dataset/acquire_bones_seed.py --authorized`; `--help` is safe
without contacting the service. Fixed acquisition source revisions, per-file
hashes, license evidence and access history remain in downloader-owned records.

`.gitignore` excludes logs, raw/original/processed/retargeted payloads, download
caches and large per-action BONES JSONL files, while preserving tools, documentation
and aggregate catalog/provenance. No dataset is published or committed here.

### SOMA acquisition queue

Current catalog also lists the pinned SOMA Uniform and Proportional BVH archives.
A queued or downloading artifact is not locally validated BVH. No full extraction
is scheduled. The independent queue waits for verified G1 completion, preserves
at least 50 decimal GB of free space, and writes `manifests/bones_seed/soma-acquisition.json`:

```bash
PYTHONPATH=/tmp/supermimic-hf-tools python dataset/acquire_bones_seed.py --authorized --representations soma_uniform soma_proportional --wait-for-g1 --reserve-gb 50
```

Use an environment with the optional data dependencies installed; the temporary
PYTHONPATH above refers to this machine's isolated official-client installation,
not a required source-code dependency. Authorization must already be granted.
The active G1 downloader and its evidence are independent and must not be duplicated.

### Motion-content QC (legacy G1 only)

`manifests/lafan1/motion_content_qc.jsonl` indexes all 862 historical cuts.
It uses float64 source-frame arrays: summed horizontal root path, net displacement,
all valid 0.5-second sliding-window translation-speed p10/p50/p90 and fraction below
0.2 m/s, 12-leg-joint finite-difference velocity RMS, and each knee's peak-to-peak angle.
`candidate_continuous_walking` contains 128 of the 283 walking-family cuts
(train63/dev31/reserved-unscored-test34), inheriting recording-level splits.
Local thresholds are p10>=0.2, 0.3<=p50<=1.8, p90<=2.2 m/s and each knee amplitude
>=0.35 rad. This is a candidate pool, not proof every frame walks or foot-contact validation.
No current training input is changed; these thresholds are not applied to unverified BONES units.

BONES metadata manifests preserve six official descriptive fields (`content_props`,
`content_body_position`, `content_complex_action`, `content_repeated_action`,
`content_horizontal_move`, `content_vertical_move`) without normalizing original strings.
Their aggregate counts are in `bones_seed/content_classification_summary.json`.
They do not provide object SE(3) trajectories, geometry, contact forces or contact labels
and are not an inferred author filtering list.

### Verified BONES archive and v3 format evidence

The G1 archive is now `available_verified`: all 142220 CSV member paths exactly
match the official metadata (no missing, extra or duplicate CSV paths). This is
archive availability, not full extraction or full numerical parsing. The 64-file
g32 smoke subset alone is `parsed_verified`; its actual row counts, source hashes
and extracted paths are recorded. Other motions remain numerically unparsed.

`manifests/bones_seed/schema-evidence-v3.json` archives fixed first-party
ProtoMotions documentation and SONIC conversion source, with revisions, URLs and
content hashes. Their documented format is 120fps, root translation in cm,
extrinsic XYZ Euler root angles in degrees and named joint angles in degrees.
These are `doc_declared` facts, separate from directly `observed` 64-file headers,
finite values and row counts. Frame counters alone are not independent timing
measurements. Root global-axis semantics remain unestablished. Earlier v2 reports
retain the unknowns known at that earlier evidence cutoff; they were not rewritten.

`convert_bones_g1.py` converts only those 64 verified samples to
`processed/bones_seed_smoke_g32/motion_source/`: root XYZ divided by100,
SciPy lowercase `xyz` extrinsic Euler degrees to XYZW quaternion, named joint
degrees to radians. It preserves every120Hz row with no axis swap, root alignment,
height correction, clipping or fake SMPL fields. A separate JSONL maps each key
to its parent full CSV path/hash, recording/mirror identity and source Frame range.
The report checks explicit Rz@Ry@Rx composition and original/mirrored samples
through the existing50Hz motion loader and native MuJoCo FK. This conversion is
not used by the active LAFAN training run and makes no dynamic feasibility claim.

### Complete40-recording LAFAN1 simulator cache

`prepare_lafan_full.py` reads all40 unchanged public G1 CSV recordings from
`retargeted/lafan1_public_g1/` at revision
`07ff8845a94d59d4195a8ac04fe918d4e6338239`. It produces
`processed/lafan1_full_g1/motions-50hz.joblib` and `manifest.json`:264705 source
30Hz rows become441140 states/441100 transitions at50Hz. Recordings remain
complete (approximately102–273s); no10s segmentation or root offsets are applied.
The cache stores SI-interpreted G1 state71, native body FK and velocity fields,
with unit root quaternion WXYZ; original CSV rotation is XYZW. Velocities use
native `mj_differentiatePos`, respecting world root linear/body angular velocity.
Per-record hashes, row/frame counts, legacy consistency and exact compiled-model
identity are recorded. Runtime rejects a different compiled-model cache.
This public retargeted library is not asserted to equal the author's unpublished
retargeting. Source/license evidence remains in `manifests/lafan1/acquisition.json`.
All40 recordings enter the new training scope and its in-distribution benchmark;
subject2 or5 are not claimed as heldout generalization in this scope.

Reproduction from the acquired CSV files, in a fresh output folder:

```bash
PYTHONPATH=src /home/hm/.conda/envs/supermimic-warp/bin/python \
  dataset/prepare_lafan_full.py --output /tmp/lafan1-full-rebuild
```

The script refuses to overwrite an existing completed manifest. Old raw/clipped
artifacts and their earlier experimental scopes remain unchanged.
