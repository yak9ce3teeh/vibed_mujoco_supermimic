"""One-pass verified BONES G1 archive indexing and bounded smoke extraction.

Only exact metadata paths (after removing leading ./) are accepted. No retargeting
or frame-rate conversion is performed; source-motion metadata is not CSV timing.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import time

import numpy as np
import pyarrow.parquet as pq

EXPECTED_BYTES = 23499973647
EXPECTED_SHA = '52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a'


def normalized_path(name):
    while name.startswith('./'):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    return path.as_posix()


def inspect_csv(payload):
    text = payload.decode('utf-8-sig')
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError('Empty CSV')
    try:
        [float(v) for v in rows[0]]
        header = None
    except ValueError:
        header = rows.pop(0)
    widths = sorted({len(row) for row in rows})
    if len(widths) != 1 or (header is not None and widths != [len(header)]):
        raise ValueError(f'Inconsistent column widths: {widths}')
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Nonfinite or empty numeric CSV')
    # A quaternion is tested only when explicitly identified by column names.
    quaternion = {'status': 'not_identified_in_header'}
    if header:
        lower = [s.strip().casefold() for s in header]
        for names in [('qx', 'qy', 'qz', 'qw'),
                      ('root_quatx', 'root_quaty', 'root_quatz', 'root_quatw')]:
            if all(s in lower for s in names):
                q = values[:, [lower.index(s) for s in names]]
                error = float(np.abs(np.linalg.norm(q, axis=1) - 1).max())
                quaternion = {'columns': names, 'max_unit_norm_error': error,
                              'unit_norm_pass': error <= 1e-4}
                if error > 1e-4:
                    raise ValueError(f'Nonunit quaternion: {error}')
                break
    return {'header': header, 'columns': values.shape[1], 'rows': values.shape[0],
            'finite': True, 'quaternion_check': quaternion, 'g1_fps': 'unknown',
            'units': 'unknown', 'joint_order': 'unknown',
            'timing_note': 'No frame-rate inferred from source-motion metadata or Frame index.'}


def scan(root):
    acquisition = json.loads((root / 'dataset/manifests/bones_seed/acquisition.json').read_text())
    record = next(r for r in acquisition['files'] if r['filename'] == 'g1.tar.gz')
    archive = root / 'dataset/retargeted/bones_seed_g1/g1.tar.gz'
    if (record.get('status') != 'available_verified'
            or record.get('sha256') != EXPECTED_SHA
            or record.get('bytes') != EXPECTED_BYTES
            or archive.stat().st_size != EXPECTED_BYTES):
        raise ValueError('Archive is not available_verified with expected official bytes/SHA; refusing scan')
    recorded_path = root / 'dataset' / record.get('path', '')
    if recorded_path.resolve() != archive.resolve():
        raise ValueError('Acquisition verified path does not match archive')
    metadata = pq.read_table(root / 'dataset/original/bones_seed/metadata/seed_metadata_v004.parquet',
                             columns=['move_g1_path', 'move_duration_frames', 'is_mirror']).to_pylist()
    expected = {r['move_g1_path']: r for r in metadata}
    if len(expected) != len(metadata):
        raise ValueError('Duplicate metadata paths')
    manifest = root / 'dataset/manifests/bones_seed/local_package_balanced_seed7_g32.jsonl'
    selected = {r['move_g1_path']: r for r in map(json.loads, manifest.read_text().splitlines())}
    if len(selected) != 64 or not selected.keys() <= expected.keys():
        raise ValueError('Expected 64 real nested-g32 IDs contained in metadata')
    output = root / 'dataset/manifests/bones_seed'
    index = output / 'archive-index.jsonl'
    report_path = output / 'archive-validation.json'
    smoke = root / 'dataset/processed/bones_seed_smoke_g32/csv'
    if index.exists() or report_path.exists() or smoke.exists():
        raise FileExistsError('Refusing to overwrite existing archive audit or smoke extraction')
    smoke.mkdir(parents=True)
    seen, duplicates, extra, non_csv, samples = set(), [], [], [], []
    started = time.monotonic()
    with archive.open('rb') as compressed, index.open('x') as stream:
        if os.fstat(compressed.fileno()).st_size != EXPECTED_BYTES:
            raise ValueError('Opened archive size differs from verified size')
        tar = tarfile.open(fileobj=compressed, mode='r|gz')
        for member in tar:
            name = normalized_path(member.name)
            row = {'path': name, 'bytes': member.size, 'type': member.type.decode('ascii', 'replace')}
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            if not member.isfile():
                non_csv.append(row)
                continue
            if not name.lower().endswith('.csv'):
                non_csv.append(row)
                continue
            if name in seen:
                duplicates.append(name)
                continue
            seen.add(name)
            if name not in expected:
                extra.append(name)
            if name not in selected:
                continue
            if member.size > 128 * 1024 * 1024:
                raise ValueError(f'Oversized selected CSV: {name}')
            payload = tar.extractfile(member).read()
            details = inspect_csv(payload)
            destination = smoke / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            details.update(path=name, sha256=hashlib.sha256(payload).hexdigest(),
                           bytes=len(payload), is_mirror=expected[name]['is_mirror'],
                           metadata_source_frames=expected[name]['move_duration_frames'])
            samples.append(details)
    missing = sorted(expected.keys() - seen)
    report = {'archive_sha256': EXPECTED_SHA, 'archive_bytes': EXPECTED_BYTES,
              'hash_evidence': 'Official acquisition verified SHA; no second full hash scan.',
              'path_mapping': 'Remove safe leading ./ only; exact full metadata move_g1_path.',
              'metadata_paths': len(expected), 'csv_paths': len(seen), 'missing': missing,
              'extra': sorted(extra), 'duplicates': sorted(duplicates), 'non_csv': non_csv,
              'selected_csv': samples, 'selected_count': len(samples),
              'elapsed_seconds': time.monotonic() - started,
              'passed': not (missing or extra or duplicates) and len(samples) == 64,
              'g1_fps': 'unknown', 'source_motion_120hz_is_not_g1_csv_fps': True,
              'scan_mode': 'single sequential r|gz; only selected 64 CSV parsed/extracted'}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'metadata_paths', 'csv_paths', 'selected_count', 'elapsed_seconds']}))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    scan(args.root.resolve())
