import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "catalog_validator", Path(__file__).resolve().parents[1] / "dataset/validate_catalog.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_catalog(tmp_path, rows):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    file = dataset / "data.bin"
    file.write_bytes(b"actual archive bytes")
    fp = module.fingerprint(file)
    catalog = dict(
        schema_version=1,
        created_utc="test",
        artifacts=[
            dict(
                id="source",
                family="LAFAN1",
                stage="source",
                path="dataset/data.bin",
                status="local_available",
                local_available=True,
                bytes=fp["bytes"],
                sha256=fp["sha256"],
                parent_ids=[],
                provenance=dict(license="recorded separately"),
                schema=dict(joint_names="unknown"),
            )
        ],
        subsets=[
            dict(
                scope_id="walking_scope",
                family="LAFAN1",
                clip_manifest="dataset/clips.jsonl",
                description="test",
                status="local_available",
            )
        ],
    )
    (dataset / "clips.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = dataset / "catalog.json"
    path.write_text(json.dumps(catalog))
    return path


def row(identifier, start, split="train", subject="subject1"):
    return dict(
        clip_id=identifier,
        artifact_id="source",
        recording_id="one_original_recording",
        subject=subject,
        skill="unknown",
        source_start=start,
        end_exclusive=start + 10,
        fps=30,
        num_frames=10,
        split=split,
        asset_hash="unknown",
    )


def test_fingerprint_keeps_license_and_excludes_only_git(tmp_path):
    data = tmp_path / "source"
    data.mkdir()
    (data / "LICENSE").write_text("real license")
    (data / ".git").mkdir()
    (data / ".git/metadata").write_text("first")
    first = module.fingerprint(data)
    (data / ".git/metadata").write_text("changed")
    assert module.fingerprint(data) == first
    (data / "LICENSE").write_text("changed license")
    assert module.fingerprint(data)["sha256"] != first["sha256"]
    assert first["file_count"] == 1


def test_catalog_allows_unknown_format_and_same_recording_same_split(tmp_path):
    path = fixture_catalog(tmp_path, [row("first", 0), row("second", 10)])
    result = module.validate_catalog(path)
    assert result["valid"] and not result["errors"] and result["warnings"]


def test_catalog_rejects_adjacent_recording_split_leakage(tmp_path):
    path = fixture_catalog(tmp_path, [row("first", 0), row("second", 10, "dev")])
    result = module.validate_catalog(path)
    assert not result["valid"] and any("crosses splits" in x for x in result["errors"])


def test_catalog_rejects_seen_subject2_blind_label_and_wrong_hash(tmp_path):
    item = row("seen", 0, "test", "subject2")
    item["blind"] = True
    path = fixture_catalog(tmp_path, [item])
    catalog = json.loads(path.read_text())
    catalog["artifacts"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(catalog))
    result = module.validate_catalog(path)
    assert not result["valid"]
    assert any("SHA256 mismatch" in x for x in result["errors"])
    assert any("cannot be blind" in x for x in result["errors"])
