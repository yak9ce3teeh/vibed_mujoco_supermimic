"""Validate local artifact integrity and scope-specific clip split provenance.

Paths in catalog and subsets are relative to the project root. This tool never
converts unknown formats, invents clip IDs, or changes dataset/training files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


CLIP_FIELDS = (
    "clip_id",
    "artifact_id",
    "recording_id",
    "subject",
    "skill",
    "source_start",
    "end_exclusive",
    "fps",
    "num_frames",
    "split",
    "asset_hash",
)


def _file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    """Directory tree-v1 excludes only .git; license/hidden data files are included.

    Tree SHA hashes UTF-8 JSONL [relative POSIX path,file SHA256,file bytes], with
    compact separators and lexical relative-path ordering. File bytes are summed.
    Regular files use their plain SHA256. No credential/account access is used.
    """
    path = Path(path)
    if path.is_file():
        return dict(
            bytes=path.stat().st_size, sha256=_file_sha(path), kind="file", hash_method="sha256"
        )
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        (p for p in path.rglob("*") if p.is_file() and ".git" not in p.relative_to(path).parts),
        key=lambda p: p.relative_to(path).as_posix(),
    )
    digest = hashlib.sha256()
    size = 0
    for file in files:
        count = file.stat().st_size
        size += count
        row = [file.relative_to(path).as_posix(), _file_sha(file), count]
        digest.update(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        )
    return dict(
        bytes=size,
        sha256=digest.hexdigest(),
        kind="directory",
        file_count=len(files),
        hash_method="tree-v1-jsonl-path-sha256-bytes",
        excluded=[".git"],
    )


def _known_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value.lower())
    )


def validate_catalog(path, project_root=None):
    path = Path(path).resolve()
    root = Path(project_root).resolve() if project_root else path.parent.parent
    catalog = json.loads(path.read_text())
    errors = []
    warnings = []
    fingerprints = {}
    artifacts = {}
    scopes = {}

    def error(message):
        errors.append(message)

    if catalog.get("schema_version") != 1:
        error("catalog schema_version must be 1")
    for artifact in catalog.get("artifacts", []):
        name = artifact.get("id")
        if not name or name in artifacts:
            error(f"duplicate/missing artifact id: {name}")
        artifacts[name] = artifact
        for key in (
            "family",
            "stage",
            "status",
            "local_available",
            "parent_ids",
            "provenance",
            "schema",
        ):
            if key not in artifact:
                error(f"{name}: missing {key}")
        if not isinstance(artifact.get("local_available"), bool):
            error(f"{name}: local_available must be boolean")
        if artifact.get("local_available"):
            local = root / str(artifact.get("path", ""))
            if not artifact.get("path") or not local.exists():
                error(f"{name}: local artifact missing: {local}")
                continue
            try:
                if str(local.resolve()) not in fingerprints:
                    fingerprints[str(local.resolve())] = fingerprint(local)
                found = fingerprints[str(local.resolve())]
                if artifact.get("bytes") != found["bytes"]:
                    error(
                        f"{name}: byte count mismatch expected={artifact.get('bytes')} actual={found['bytes']}"
                    )
                if str(artifact.get("sha256", "")).lower() != found["sha256"]:
                    error(f"{name}: SHA256 mismatch")
                if found["kind"] == "directory" and artifact.get("schema", {}).get(
                    "hash_method"
                ) not in (None, found["hash_method"]):
                    error(f"{name}: incompatible tree hash method")
            except OSError as exc:
                error(f"{name}: unreadable local artifact: {exc}")
        if not isinstance(artifact.get("schema"), dict):
            error(f"{name}: schema must be an object")
            continue
        schema = artifact["schema"]
        joints = schema.get("joint_names")
        if isinstance(joints, list):
            if not all(isinstance(x, str) for x in joints) or len(joints) != len(set(joints)):
                error(f"{name}: invalid/duplicate joint names")
            if schema.get("dof") == 29 and len(joints) != 29:
                error(f"{name}: 29DoF declaration needs 29 joint names")
        elif joints not in (None, "unknown", "not_applicable"):
            error(f"{name}: joint_names must be a list or explicit unknown/not_applicable")
        for key in (
            "root_translation_units",
            "rotation_representation",
            "quaternion_order",
            "rotation_frame",
            "joint_position_units",
            "joint_names",
            "fps",
            "velocity_reconstruction",
        ):
            if schema.get(key) in (None, "unknown"):
                warnings.append(f"{name}: schema.{key} unknown/not recorded; no value inferred")
        if not artifact.get("provenance", {}).get("license"):
            warnings.append(f"{name}: source license unknown/not recorded")
    for name, artifact in artifacts.items():
        for parent in artifact.get("parent_ids", []):
            if parent not in artifacts:
                error(f"{name}: unknown parent artifact {parent}")
    for subset in catalog.get("subsets", []):
        scope = subset.get("scope_id")
        if not scope or scope in scopes:
            error(f"duplicate/missing scope_id: {scope}")
        scopes[scope] = subset
        manifest = subset.get("clip_manifest")
        if not manifest:
            continue  # Explicitly gated/unacquired or planned scopes need no fabricated rows.
        local = root / manifest
        if not local.is_file():
            error(f"{scope}: clip manifest missing: {local}")
            continue
        seen = set()
        recording_splits = {}
        count = 0
        for line_number, line in enumerate(local.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                error(f"{scope}:{line_number}: invalid JSON: {exc}")
                continue
            prefix = f"{scope}:{line_number}"
            missing = [key for key in CLIP_FIELDS if key not in row]
            if missing:
                error(f"{prefix}: missing fields {missing}")
                continue
            count += 1
            identifier = row["clip_id"]
            if identifier in seen:
                error(f"{prefix}: duplicate clip_id {identifier}")
            seen.add(identifier)
            artifact = artifacts.get(row["artifact_id"])
            if artifact is None:
                error(f"{prefix}: unknown artifact_id {row['artifact_id']}")
            elif not artifact.get("local_available"):
                error(f"{prefix}: clip references unacquired data artifact")
            start, end, frames = row["source_start"], row["end_exclusive"], row["num_frames"]
            if (
                not all(
                    isinstance(x, int) and not isinstance(x, bool) for x in (start, end, frames)
                )
                or start < 0
                or end <= start
                or frames != end - start
            ):
                error(f"{prefix}: invalid source-frame half-open interval or num_frames")
            if (
                not isinstance(row["fps"], (int, float))
                or not math.isfinite(row["fps"])
                or row["fps"] <= 0
            ):
                error(f"{prefix}: fps must be positive finite")
            if row["asset_hash"] != "unknown" and not _known_hash(row["asset_hash"]):
                error(f"{prefix}: asset_hash must be SHA256 or explicit unknown")
            subject = str(row["subject"]).lower().replace("_", "").replace("-", "")
            if subject in ("subject2", "2") and (
                row.get("blind", False)
                or subset.get("blind", False)
                or subset.get("blind_test", False)
            ):
                error(f"{prefix}: previously inspected subject2 cannot be blind test")
            if row["split"] not in (
                "train",
                "dev",
                "validation",
                "test",
                "diagnostic",
                "unassigned",
            ):
                error(f"{prefix}: unsupported split {row['split']}")
            key = str(row["recording_id"])
            prior = recording_splits.setdefault(key, row["split"])
            if prior != row["split"]:
                error(
                    f"{prefix}: same recording crosses splits ({prior}/{row['split']}); scope leakage"
                )
        if "paper_filtered_exact" in str(scope).lower() and not subset.get("exact_filter_evidence"):
            error(f"{scope}: cannot claim exact author-filtered IDs without evidence")
        if not count and subset.get("status") == "local_available":
            warnings.append(f"{scope}: available subset contains no clip rows")
    return dict(
        valid=not errors,
        catalog=str(path),
        project_root=str(root),
        artifacts=len(artifacts),
        scopes=len(scopes),
        local_fingerprints=fingerprints,
        errors=errors,
        warnings=warnings,
        split_policy="Each scope independently assigns one split per original recording, stronger than only overlap/adjacency checks. No cross-scope blind-test claim is inferred.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_catalog(args.catalog, args.project_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 2)


if __name__ == "__main__":
    main()
