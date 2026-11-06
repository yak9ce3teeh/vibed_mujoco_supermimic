"""Rebuild artifact/subset metadata without changing dataset payloads or training.

Requires joblib/numpy plus pyarrow for the acquired BONES metadata. Acquisition
JSON files are read-only inputs owned by the download process. All paths in the
catalog are project-relative; no policy evaluation or retarget conversion runs.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path, PurePosixPath
import re

import joblib
import numpy as np
import pyarrow.parquet as pq

from supermimic.motion import JOINT_NAMES
from validate_catalog import fingerprint, validate_catalog


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def g1_schema(fps):
    return dict(
        representation="retargeted_g1",
        robot="Unitree G1",
        dof=29,
        joint_names=list(JOINT_NAMES),
        root_translation_units="m",
        rotation_representation="quaternion",
        quaternion_order="XYZW",
        rotation_frame="unknown",
        joint_position_units="rad",
        fps=fps,
        velocity_reconstruction="runtime: SLERP then MuJoCo mj_differentiatePos; no stored joint/root velocities",
        runtime_interpretation="root applied as world pose in SI; exporter coordinate frame/robot-asset revision not independently established",
    )


def recording_fields(name):
    match = re.fullmatch(r"(.+)_subject([0-9]+)", name)
    if not match:
        raise ValueError(f"cannot derive declared recording subject: {name}")
    return match.group(1), "subject" + match.group(2)


def build(project):
    project = Path(project).resolve()
    data = project / "dataset"
    manifests = data / "manifests"
    lafan = json.loads((manifests / "lafan1/acquisition.json").read_text())
    bones = json.loads((manifests / "bones_seed/acquisition.json").read_text())
    legacy = json.loads((data / "provenance.json").read_text())
    artifacts = []
    subsets = []
    counts = {}

    def rel(path):
        return str(Path(path).relative_to(project))

    def artifact(
        identifier,
        family,
        stage,
        path,
        provenance,
        schema,
        parents=(),
        available=True,
        status=None,
        expected=None,
    ):
        path = Path(path) if path else None
        item = dict(
            id=identifier,
            family=family,
            stage=stage,
            path=rel(path) if path else None,
            status=status or ("local_available" if available else "not_acquired"),
            local_available=available,
            parent_ids=list(parents),
            provenance=provenance,
            schema=schema,
        )
        if available:
            item.update({k: v for k, v in fingerprint(path).items() if k in ("bytes", "sha256")})
            if path.is_dir():
                item["schema"]["hash_method"] = "tree-v1-jsonl-path-sha256-bytes"
        else:
            item.update(bytes=None, sha256=None)
        if expected:
            item["expected_download"] = expected
        artifacts.append(item)
        return item

    def subset(scope, family, name, rows, description, status="local_available", **extra):
        path = manifests / family / name
        write_rows(path, rows)
        item = dict(
            scope_id=scope,
            family=family,
            clip_manifest=rel(path),
            description=description,
            status=status,
            **extra,
        )
        subsets.append(item)
        counts[scope] = dict(
            rows=len(rows),
            recordings=len({r["recording_id"] for r in rows}),
            splits=dict(Counter(r["split"] for r in rows)),
            source_hours=sum(r["num_frames"] / r["fps"] for r in rows) / 3600,
        )
        return item

    raw = joblib.load(data / "raw/lafan_29dof.pkl")
    cuts = joblib.load(data / "processed/lafan_29dof_10s-clipped.pkl")
    if len(raw) != 40 or len(cuts) != 862:
        raise ValueError(
            "legacy source changed from verified40/862; review scopes before rebuilding"
        )
    legacy_provenance = dict(
        source_url=legacy["files"][0]["source"],
        source_revision="unknown",
        license="unknown for historical retargeted conversion; no redistribution rights inferred",
        evidence="dataset/provenance.json; original retargeter/assets not recovered",
    )
    artifact(
        "lafan_legacy_g1_full",
        "LAFAN1",
        "retargeted_g1_full",
        data / "raw/lafan_29dof.pkl",
        legacy_provenance,
        g1_schema(30),
    )
    slice_evidence = []
    cut_rows = {}
    for key in sorted(cuts):
        match = re.fullmatch(r"(.+)_clip([0-9]+)", key)
        if not match:
            raise ValueError(f"unknown historical clip naming: {key}")
        parent, index = match.group(1), int(match.group(2))
        clip = cuts[key]
        source = raw[parent]
        window = round(10 * float(source["fps"]))
        start = index * window
        end = start + len(clip["dof"])
        equal = {
            field: bool(
                np.array_equal(np.asarray(clip[field]), np.asarray(source[field])[start:end])
            )
            for field in ("root_trans_offset", "dof", "root_rot")
        }
        if not all(equal.values()):
            raise ValueError(f"historical slice mismatch: {key}")
        skill, subject = recording_fields(parent)
        cut_rows[key] = dict(
            clip_id=key,
            artifact_id="lafan_legacy_g1_10s",
            recording_id=parent,
            subject=subject,
            skill=skill,
            source_start=start,
            end_exclusive=end,
            fps=float(clip["fps"]),
            num_frames=len(clip["dof"]),
            split="unassigned",
            asset_hash="unknown",
            source_frame_domain="legacy_retargeted_g1_source_frames",
            visibility="metadata/schema inspected; no blind-policy evaluation claimed",
        )
        slice_evidence.append(
            dict(
                clip_id=key, recording_id=parent, source_start=start, end_exclusive=end, equal=equal
            )
        )
    write_json(
        manifests / "lafan1/legacy_slice_validation.json",
        dict(
            checked=len(slice_evidence),
            fields=["root_trans_offset", "dof", "root_rot"],
            results=slice_evidence,
        ),
    )
    artifact(
        "lafan_legacy_g1_10s",
        "LAFAN1",
        "historical_10s_cuts",
        data / "processed/lafan_29dof_10s-clipped.pkl",
        {
            **legacy_provenance,
            "evidence": "manifests/lafan1/legacy_slice_validation.json; all862 slices verified exactly",
        },
        g1_schema(30),
        parents=["lafan_legacy_g1_full"],
    )
    single = []
    for key, split in [("walk1_subject1_clip0", "train"), ("walk1_subject2_clip0", "dev")]:
        row = dict(cut_rows[key], split=split, blind=False)
        single.append(row)
    subset(
        "lafan1_single_walk10s",
        "lafan1",
        "single_walk10s.jsonl",
        single,
        "Current one-clip fit: train1/dev1, no test; subject2 repeatedly inspected.",
    )
    walking = []
    for key, row in cut_rows.items():
        if not re.fullmatch(r"walk[1-4]_subject[1-5]", row["recording_id"]):
            continue
        subject = row["subject"]
        split = (
            "train"
            if subject in ("subject1", "subject3", "subject4")
            else "dev"
            if subject == "subject2"
            else "test"
        )
        walking.append(
            dict(
                row,
                split=split,
                blind=False,
                benchmark_status="reserved_unscored"
                if split == "test"
                else "seen_development"
                if split == "dev"
                else "training_pool",
            )
        )
    split_counts = Counter(r["split"] for r in walking)
    if split_counts != dict(train=183, dev=50, test=50):
        raise ValueError(f"walking scope cardinality changed: {split_counts}")
    subset(
        "lafan1_walking_family",
        "lafan1",
        "walking_family.jsonl",
        walking,
        "Only ^walk[1-4]_subject; train subjects1/3/4=183cuts, dev2=50, reserved_unscored test5=50. Schema/slice metadata read; no pristine-blind claim.",
    )
    # Content QC uses original 30Hz legacy G1 arrays, not labels or renderer poses.
    qc_rows = []
    candidate_ids = set()
    for key, clip in sorted(cuts.items()):
        fps = float(clip["fps"])
        xy = np.asarray(clip["root_trans_offset"], dtype=np.float64)[:, :2]
        joints = np.asarray(clip["dof"], dtype=np.float64)
        window = round(.5 * fps)
        speed = np.linalg.norm(xy[window:] - xy[:-window], axis=1) / (window / fps)
        p10, p50, p90 = np.quantile(speed, [.1, .5, .9])
        knees = np.ptp(joints[:, [3, 9]], axis=0)
        candidate = bool(p10 >= .2 and .3 <= p50 <= 1.8 and p90 <= 2.2 and np.all(knees >= .35))
        if candidate:
            candidate_ids.add(key)
        qc_rows.append(dict(
            clip_id=key, skill=cut_rows[key]["skill"], family="LAFAN1 legacy G1",
            root_xy_path_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            net_xy_m=float(np.linalg.norm(xy[-1] - xy[0])),
            halfsecond_window_frames=window,
            horizontal_speed_m_s=dict(p10=float(p10), p50=float(p50), p90=float(p90)),
            low_translation_fraction=float(np.mean(speed < .2)),
            leg12_joint_velocity_rms_rad_s=float(np.sqrt(np.mean((np.diff(joints[:, :12], axis=0) * fps)**2))),
            left_knee_peak_to_peak_rad=float(knees[0]), right_knee_peak_to_peak_rad=float(knees[1]),
            candidate_thresholds_pass=candidate,
        ))
    write_rows(manifests / "lafan1/motion_content_qc.jsonl", qc_rows)
    candidates = [dict(row, qc_clip_index="dataset/manifests/lafan1/motion_content_qc.jsonl")
                  for row in walking if row["clip_id"] in candidate_ids]
    if Counter(row["split"] for row in candidates) != dict(train=63, dev=31, test=34):
        raise ValueError("Continuous walking candidate QC differs from independent 128 golden")
    subset("lafan1_candidate_continuous_walking", "lafan1", "candidate_continuous_walking.jsonl",
           candidates, "Local kinematic QC candidates from walking_family only; not pure-walk guarantee or contact validation.",
           qc_definition=dict(window_seconds=.5, horizontal_speed_p10_min=.2,
                              horizontal_speed_p50_range=[.3,1.8], horizontal_speed_p90_max=2.2,
                              each_knee_peak_to_peak_min_rad=.35),
           qc_index="dataset/manifests/lafan1/motion_content_qc.jsonl")
    coverage = []
    for key in sorted(raw):
        skill, subject = recording_fields(key)
        clip = raw[key]
        coverage.append(
            dict(
                clip_id=key,
                artifact_id="lafan_legacy_g1_full",
                recording_id=key,
                subject=subject,
                skill=skill,
                source_start=0,
                end_exclusive=len(clip["dof"]),
                fps=float(clip["fps"]),
                num_frames=len(clip["dof"]),
                split="diagnostic",
                asset_hash="unknown",
                source_frame_domain="legacy_retargeted_g1_source_frames",
            )
        )
    subset(
        "lafan1_full40_coverage",
        "lafan1",
        "full40_coverage.jsonl",
        coverage,
        "Inventory only, not a train/dev/test benchmark. Includes walking dev/test recordings.",
    )
    subset(
        "lafan1_training_pool862",
        "lafan1",
        "training_pool862.jsonl",
        list(cut_rows.values()),
        "All historical cuts, unassigned. Separate scope from walking split; full coverage includes its holdout.",
    )
    official = lafan["lafan1_original"]
    revision = official["revision"]
    original_schema = dict(
        representation="native_bvh",
        root_translation_units="unknown",
        rotation_representation="BVH Euler channels",
        quaternion_order="not_applicable",
        rotation_frame="unknown",
        joint_position_units="unknown",
        joint_names="unknown",
        fps="per-file Frame Time",
        velocity_reconstruction="not performed",
    )
    original_provenance = dict(
        source_url=official["artifact"]["source_url"],
        source_revision=revision,
        license="Archived upstream license.txt; CC BY-NC-ND4.0 terms, no expanded rights inferred",
        evidence="dataset/manifests/lafan1/acquisition.json and dataset/original/lafan1/license.txt",
    )
    artifact(
        "lafan1_original_zip",
        "LAFAN1",
        "original_archive",
        data / official["artifact"]["path"],
        original_provenance,
        dict(original_schema),
    )
    artifact(
        "lafan1_original_bvh77",
        "LAFAN1",
        "original_bvh_inventory",
        data / "original/lafan1/bvh",
        original_provenance,
        dict(original_schema),
        parents=["lafan1_original_zip"],
    )
    if len(list((data / "original/lafan1/bvh").glob("*.bvh"))) != 77:
        raise ValueError("expected77 original BVH files")
    public = lafan["public_g1"]
    public_checks = []
    for path in sorted((data / "retargeted/lafan1_public_g1").glob("*.csv")):
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape[1] != 36 or not np.isfinite(values).all():
            raise ValueError(f"invalid public G1 CSV: {path}")
        quaternion_error = float(np.max(np.abs(np.linalg.norm(values[:, 3:7], axis=1) - 1)))
        if quaternion_error > 1e-4:
            raise ValueError(f"nonunit public quaternion: {path}")
        public_checks.append(
            dict(
                path=rel(path),
                rows=len(values),
                columns=36,
                all_finite=True,
                max_quaternion_norm_error=quaternion_error,
                sha256=fingerprint(path)["sha256"],
                legacy_basename_match=path.stem in raw,
                conversion_equivalence_to_legacy="not established",
            )
        )
    if len(public_checks) != 40:
        raise ValueError("expected40 public G1 CSVs")
    write_json(
        manifests / "lafan1/public_g1_schema_validation.json",
        dict(
            files=public_checks,
            count=40,
            observed="36columns, finite values, unit columns3:7; no policy benchmark",
        ),
    )
    card_names = tuple(
        name + "_joint"
        for name in [
            side + "_" + joint
            for side in ("left", "right")
            for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ]
        + ["waist_yaw", "waist_roll", "waist_pitch"]
        + [
            side + "_" + joint
            for side in ("left", "right")
            for joint in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ]
    )
    if card_names != JOINT_NAMES:
        raise ValueError("source-card29 joint order differs from runtime order")
    public_schema = g1_schema(30)
    public_schema.update(
        root_translation_units="unknown",
        joint_position_units="unknown",
        velocity_reconstruction="not performed",
        declared_by_source_card=dict(fps=30, quaternion_order="XYZW", joint_names=list(card_names)),
        observed_check="manifests/lafan1/public_g1_schema_validation.json",
    )
    artifact(
        "lafan1_public_g1_csv40",
        "LAFAN1",
        "public_retargeted_g1",
        data / "retargeted/lafan1_public_g1",
        dict(
            source_url="https://huggingface.co/datasets/" + public["repository"],
            source_revision=public["revision"],
            license=public["license"],
            evidence=public.get("license_evidence", []),
            related_sources=["lafan1_original_zip"],
            relation="LAFAN1 family derivative according to source card; exact input archive revision/exporter asset unknown",
        ),
        public_schema,
        parents=[],
    )
    # BONES rows index source metadata only;120Hz never asserts G1 CSV sample rate.
    metadata = data / "original/bones_seed/metadata/seed_metadata_v004.parquet"
    metadata_file = next(f for f in bones["files"] if f["filename"].endswith(".parquet"))
    bone_provenance = dict(
        source_url=metadata_file["source_url"],
        source_revision=bones["revision"],
        license="BONES-SEED license: https://bones.studio/info/seed-license",
        evidence="dataset/original/bones_seed/README.md and manifests/bones_seed/acquisition.json",
    )
    artifact(
        "bones_seed_metadata_v004",
        "BONES-SEED",
        "source_metadata",
        metadata,
        bone_provenance,
        dict(
            representation="official_parquet_metadata",
            root_translation_units="unknown",
            rotation_representation="unknown",
            quaternion_order="unknown",
            rotation_frame="unknown",
            joint_position_units="unknown",
            joint_names="unknown",
            fps="120 source-motion domain only; G1 CSV unknown",
            velocity_reconstruction="not performed",
        ),
    )
    archive_record = next(f for f in bones["files"] if f["filename"] == "g1.tar.gz")
    archive_path = data / "retargeted/bones_seed_g1/g1.tar.gz"
    archive_available = archive_record["status"] == "available_verified" and archive_path.is_file()
    archive_audit_path = manifests / "bones_seed/archive-validation.json"
    archive_audit = json.loads(archive_audit_path.read_text()) if archive_audit_path.exists() else None
    archive_paths = set()
    parsed_csv = {}
    if archive_available and archive_audit:
        if not archive_audit["passed"] or archive_audit["archive_sha256"] != archive_record["sha256"]:
            raise ValueError("Archive coverage audit disagrees with verified acquisition")
        for line in (manifests / "bones_seed/archive-index.jsonl").open():
            member = json.loads(line)
            if member["type"] == "0" and member["path"].endswith(".csv"):
                archive_paths.add(member["path"])
        if len(archive_paths) != 142220:
            raise ValueError("Verified archive path cardinality changed")
        parsed_csv = {row["path"]: row for row in archive_audit["selected_csv"]}
        if len(parsed_csv) != 64:
            raise ValueError("Expected exactly 64 parsed smoke CSVs")
    artifact(
        "bones_seed_g1_archive",
        "BONES-SEED",
        "g1_payload_archive",
        archive_path,
        {**bone_provenance, "source_url": archive_record["source_url"]},
        dict(
            representation="verified archive of G1 CSVs; all142220 member paths indexed, only64 parsed" if archive_paths else "unknown until archive/CSV inspection",
            root_translation_units="unknown",
            rotation_representation="unknown",
            quaternion_order="unknown",
            rotation_frame="unknown",
            joint_position_units="unknown",
            joint_names="unknown",
            fps="unknown",
            velocity_reconstruction="not performed",
        ),
        available=archive_available,
        status=archive_record["status"],
        expected=dict(
            bytes=archive_record["expected_bytes"], sha256=archive_record["expected_sha256"]
        ),
    )
    format_evidence_path = manifests / "bones_seed/schema-evidence-v3.json"
    format_evidence = json.loads(format_evidence_path.read_text()) if format_evidence_path.exists() else None
    if format_evidence:
        artifacts[-1]["schema"]["doc_declared"] = format_evidence["doc_declared"]
        artifacts[-1]["schema"]["observed"] = format_evidence["observed"]
        artifacts[-1]["schema"]["format_evidence"] = rel(format_evidence_path)
    converted = data / "processed/bones_seed_smoke_g32/motion_source/bones_seed_g32_motion_source.pkl"
    if converted.exists():
        artifact("bones_seed_g32_motion_source", "BONES-SEED", "local_64_motion_conversion",
                 converted,
                 dict(source_url=bone_provenance["source_url"], source_revision=bones["revision"],
                      license=bone_provenance["license"],
                      evidence="dataset/processed/bones_seed_smoke_g32/motion_source/conversion-validation.json and dataset/manifests/bones_seed/schema-evidence-v3.json"),
                 dict(representation="local Motion source 64 clips; root_trans_offset,dof,root_rot,fps only",
                      root_translation_units="meters", rotation_representation="unit quaternion", quaternion_order="XYZW",
                      rotation_frame="source global axes preserved; physical axis convention unspecified",
                      joint_position_units="radians", joint_names=list(JOINT_NAMES), fps=120,
                      velocity_reconstruction="not stored; runtime load_motions reconstructs at 50Hz"),
                 parents=["bones_seed_g1_archive"])
    # SOMA payloads are independent queued acquisitions, not locally usable BVH.
    soma_evidence = manifests / "bones_seed/soma-acquisition.json"
    soma = json.loads(soma_evidence.read_text()) if soma_evidence.exists() else {}
    soma_records = {row["filename"]: row for row in soma.get("files", [])}
    for variant, expected_bytes in (("uniform", 45188088297), ("proportional", 45468083856)):
        filename = f"soma_{variant}.tar.gz"
        record = soma_records.get(filename, {})
        path = data / "original/bones_seed" / filename
        available = record.get("status") == "available_verified" and path.is_file()
        artifact(
            f"source_soma_{variant}_bvh_archive",
            "BONES-SEED",
            "source_soma_bvh_archive",
            path,
            dict(
                source_url=f"https://huggingface.co/datasets/bones-studio/seed/resolve/{bones['revision']}/{filename}",
                source_revision=bones["revision"],
                license=bone_provenance["license"],
                evidence="dataset/manifests/bones_seed/soma-acquisition.json (independent download queue)",
            ),
            dict(
                representation=f"SOMA {variant} BVH archive; members not inspected",
                root_translation_units="unknown",
                rotation_representation="unknown",
                quaternion_order="not applicable or unknown until BVH inspection",
                rotation_frame="unknown",
                joint_position_units="unknown",
                joint_names="unknown",
                fps="unknown until actual BVH Frames/Frame Time inspection",
                velocity_reconstruction="not performed",
            ),
            available=available,
            status=record.get("status", "queued_not_acquired"),
            expected=dict(bytes=expected_bytes, sha256=record.get("expected_sha256")),
        )
    columns = [
        "move_g1_path",
        "move_name",
        "move_duration_frames",
        "package",
        "category",
        "content_type_of_movement",
        "is_mirror",
        "actor_uid",
    ]
    content_columns = ["content_props", "content_body_position", "content_complex_action",
                       "content_repeated_action", "content_horizontal_move", "content_vertical_move"]
    columns += content_columns
    metadata_rows = pq.read_table(metadata, columns=columns).to_pylist()
    bone_rows = []
    groups = defaultdict(list)
    seen = set()
    for source in metadata_rows:
        key = str(PurePosixPath(source["move_g1_path"]))
        mirror = bool(source["is_mirror"])
        if key in seen:
            raise ValueError(f"duplicate BONES G1 path: {key}")
        seen.add(key)
        parent = key[:-6] + ".csv" if key.endswith("_M.csv") else key
        if key.endswith("_M.csv") != mirror:
            raise ValueError(f"mirror flag/path disagreement: {key}")
        frames = int(source["move_duration_frames"])
        movement = str(source["content_type_of_movement"] or "unknown").strip()
        row = dict(
            clip_id=key,
            move_name=source["move_name"],
            move_g1_path=key,
            artifact_id="bones_seed_metadata_v004",
            recording_id=parent,
            subject=str(source["actor_uid"]),
            skill=movement.lower(),
            source_start=0,
            end_exclusive=frames,
            fps=120,
            num_frames=frames,
            split="unassigned",
            asset_hash="unknown",
            source_frame_domain="source_motion_120hz",
            data_status="metadata_indexed",
            g1_csv_local_status="not_validated",
            g1_fps="unknown",
            g1_num_frames="unknown",
            g1_data_artifact_id="bones_seed_g1_archive",
            package=source["package"],
            category=source["category"],
            content_type_of_movement=source["content_type_of_movement"],
            is_mirror=mirror,
            actor_uid=source["actor_uid"],
        )
        if archive_paths:
            if key not in archive_paths:
                raise ValueError(f"Metadata path absent from verified archive: {key}")
            row.update(
                g1_documented_fps=120 if format_evidence else "unknown",
                g1_doc_schema_evidence="dataset/manifests/bones_seed/schema-evidence-v3.json" if format_evidence else "unknown",
                g1_storage_status="available_in_verified_archive",
                archive_artifact_id="bones_seed_g1_archive",
                g1_csv_local_status="verified_in_archive",
                g1_schema_status="unknown_unparsed",
            )
            if key in parsed_csv:
                parsed = parsed_csv[key]
                row.update(
                    g1_csv_local_status="parsed_verified",
                    g1_schema_status="observed_36_named_columns_finite",
                    g1_num_frames=parsed["rows"],
                    g1_extracted_path="dataset/processed/bones_seed_smoke_g32/csv/" + key,
                    g1_csv_sha256=parsed["sha256"],
                    g1_schema_evidence="dataset/manifests/bones_seed/smoke-schema-validation.json",
                )
        row.update({name: source[name] for name in content_columns})
        bone_rows.append(row)
        groups[parent].append(row)
    write_json(manifests / "bones_seed/content_classification_summary.json", dict(
        counts={name: dict(Counter(str(row[name]) for row in bone_rows)) for name in content_columns},
        meaning="Official descriptive metadata only; no object SE(3), geometry, force, or contact annotation inferred.",
    ))
    originals = [r for r in bone_rows if not r["is_mirror"]]
    mirrored = [r for r in bone_rows if r["is_mirror"]]
    walking_bones = [
        r
        for r in bone_rows
        if r["package"] == "Locomotion"
        and str(r["content_type_of_movement"]).strip().lower() == "walking"
    ]
    expected = (
        len(bone_rows),
        len(originals),
        len(mirrored),
        len(groups),
        len(walking_bones),
        len({r["recording_id"] for r in walking_bones}),
    )
    if expected != (142220, 71132, 71088, 71132, 15946, 7980):
        raise ValueError(f"BONES metadata cardinality/version changed: {expected}")
    for name, rows in [
        ("full_metadata", bone_rows),
        ("original", originals),
        ("mirrored", mirrored),
        ("LocomotionWalking", walking_bones),
    ]:
        subset(
            "bones_seed_" + name,
            "bones_seed",
            name + ".jsonl",
            rows,
            "Metadata indexed only; fps/frame ranges describe source motion120Hz, not retargeted G1 duration/rows.",
            status="metadata_indexed",
        )
    package_groups = defaultdict(list)
    for group_id, group_rows in groups.items():
        original = next(r for r in group_rows if not r["is_mirror"])
        if any(r["package"] != original["package"] for r in group_rows):
            raise ValueError("mirror group crosses package labels")
        package_groups[original["package"]].append(group_id)
    golden = {
        4: "df4860b7ff49d56951a4ee3d630f3f6227d0277f24a07fcd272801c63a813d2e",
        64: "7db60351480330d788e0a397bbae359fd34279305abbcee06d972783d5cb7a6f",
        512: "096635027f69513909a917fd0bfdd1a02528ed0d6a1c8a88a0306f308cae120f",
    }
    for quota in (4, 64, 512):
        selected = set()
        for package, identifiers in package_groups.items():
            ordered = sorted(
                identifiers,
                key=lambda key: (
                    hashlib.sha256(f"{bones['revision']}|7|{key}".encode("utf8")).digest(),
                    key,
                ),
            )
            selected.update(ordered[:quota])
        selected_hash = hashlib.sha256(("\n".join(sorted(selected)) + "\n").encode()).hexdigest()
        if len(package_groups) != 8 or selected_hash != golden[quota]:
            raise ValueError("balanced subset changed from independently reviewed metadata/version")
        selected_rows = [row for row in bone_rows if row["recording_id"] in selected]
        name = f"local_package_balanced_seed7_g{len(selected)}"
        subset(
            "bones_seed_" + name,
            "bones_seed",
            name + ".jsonl",
            selected_rows,
            "Local equal-package metadata sample, not natural-frequency, author-filtered or holdout; includes whole original/mirror groups; split unassigned.",
            status="metadata_indexed",
            selection=dict(
                seed=7,
                groups_per_package=quota,
                packages=8,
                group_ids_sha256=selected_hash,
                ranking="sha256(revision|7|original_full_move_g1_path), then path",
                mirror_policy="include all acquired metadata rows in selected parent group",
                distinct_full_movement_labels=len(
                    {
                        str(row["content_type_of_movement"]).strip().casefold()
                        for row in selected_rows
                    }
                ),
            ),
        )
    unequal = sum(
        len(rows) == 2 and rows[0]["num_frames"] != rows[1]["num_frames"]
        for rows in groups.values()
    )
    write_json(
        manifests / "bones_seed/group_summary.json",
        dict(
            metadata_rows=len(bone_rows),
            parent_groups=len(groups),
            group_sizes=dict(Counter(map(len, groups.values()))),
            unequal_frame_count_pairs=unequal,
            note="Mirrors share parent recording group, but source frame counts may differ; no G1 frame-rate inference.",
        ),
    )
    subsets.extend(
        [
            dict(
                scope_id="bones_seed_paper_filtered_reported",
                family="bones_seed",
                clip_manifest=None,
                status="published_count_only",
                description="Paper reports129514 motions/264.56h; author filter IDs not supplied, not an exact local subset.",
                published_count=129514,
                published_hours=264.56,
                evidence="https://arxiv.org/abs/2610.28480",
            ),
            dict(
                scope_id="bones_seed_cuboi_reported",
                family="bones_seed",
                clip_manifest=None,
                status="published_count_only",
                description="Reproduction reports129491; no matching ID list locally established.",
                published_count=129491,
                evidence="https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2",
            ),
        ]
    )
    catalog = dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        artifacts=artifacts,
        subsets=subsets,
        note="Independent scopes; no cross-scope blind benchmark claim. Current trainer input/source is unchanged. Metadata-only BONES pools are not G1 CSV availability claims.",
    )
    write_json(data / "catalog.json", catalog)
    write_json(
        manifests / "index.json",
        dict(
            schema_version=1,
            scopes=counts,
            source_frame_semantics="source ranges are half-open. BONES metadata fps120 is not G1 data fps.",
            unassigned_is_not_training=True,
        ),
    )
    validation = validate_catalog(data / "catalog.json", project)
    write_json(data / "catalog.validation.json", validation)
    if not validation["valid"]:
        raise ValueError(validation["errors"])
    print(
        json.dumps(
            dict(
                artifacts=len(artifacts),
                scopes=counts,
                validation_errors=validation["errors"],
                warnings=len(validation["warnings"]),
                bones_archive_available=archive_available,
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    build(args.project)


if __name__ == "__main__":
    main()
