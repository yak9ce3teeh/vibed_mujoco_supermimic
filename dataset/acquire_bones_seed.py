"""Authorized official-SDK BONES acquisition; credentials remain inside SDK.

Adapted from the original project downloader; supports sequential representation
queues with separate SOMA evidence. Never writes G1 evidence for a SOMA queue.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

REVISION = '2f59b2077b9da34dd4e43618e705c7cb962c9a66'
REPOSITORY = 'bones-studio/seed'
SIZES = {'g1': 23499973647, 'soma_uniform': 45188088297,
         'soma_proportional': 45468083856}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized', action='store_true')
    parser.add_argument('--representations', nargs='+', choices=list(SIZES), default=['g1'])
    parser.add_argument('--wait-for-g1', action='store_true')
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--reserve-gb', type=float, default=50)
    args = parser.parse_args()
    if not args.authorized:
        parser.error('Explicit user authorization is required')
    if args.reserve_gb < 50:
        parser.error('At least 50 decimal GB free-space reserve is required')
    if len(set(args.representations)) != len(args.representations):
        parser.error('Duplicate representation')
    soma = any(s.startswith('soma_') for s in args.representations)
    if soma and 'g1' in args.representations:
        parser.error('Use independent G1 and SOMA queues/evidence')
    root = Path(__file__).resolve().parent
    main_evidence = root / 'manifests/bones_seed/acquisition.json'
    evidence = args.evidence or (root / 'manifests/bones_seed/soma-acquisition.json' if soma else main_evidence)
    evidence = evidence.resolve()
    if soma and evidence == main_evidence.resolve():
        parser.error('SOMA cannot write active G1 acquisition evidence')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    lock = evidence.with_suffix('.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another queue owns this evidence')
    if args.wait_for_g1:
        if not soma:
            parser.error('--wait-for-g1 is only for SOMA queues')
        print(json.dumps({'status': 'waiting_for_verified_g1'}), flush=True)
        while True:
            g1 = json.loads(main_evidence.read_text()) if main_evidence.exists() else {}
            records = g1.get('files', [])
            if g1.get('status') == 'complete' and any(r.get('filename') == 'g1.tar.gz' and r.get('status') == 'available_verified' for r in records):
                break
            if g1.get('status') == 'failed':
                raise RuntimeError('G1 acquisition failed; SOMA queue not started')
            time.sleep(30)
    from huggingface_hub import HfApi, hf_hub_download
    logging.getLogger('huggingface_hub').setLevel(logging.CRITICAL)
    info = HfApi().dataset_info(REPOSITORY, revision=REVISION, files_metadata=True, token=True)
    siblings = {item.rfilename: item for item in info.siblings}
    state = dict(repository=REPOSITORY, revision=REVISION, authorized_by_user=True,
                 started_utc=datetime.now(timezone.utc).isoformat(), status='running', files=[])

    def save():
        temporary = evidence.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(evidence)

    try:
        for representation in args.representations:
            filename = representation + '.tar.gz'
            item = siblings[filename]
            if item.size != SIZES[representation] or not item.lfs:
                raise RuntimeError('Unexpected official artifact identity')
            folder = 'retargeted/bones_seed_g1' if representation == 'g1' else 'original/bones_seed'
            destination = root / folder / filename
            record = dict(filename=filename, status='queued_not_acquired', expected_bytes=item.size,
                          expected_sha256=item.lfs.sha256,
                          source_url=f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{filename}')
            state['files'].append(record)
            save()
            # SDK local-dir resumes incomplete downloads; conservatively budget full
            # archive even if a partial cache already occupies space.
            if not destination.is_file() and shutil.disk_usage(root).free < item.size + args.reserve_gb * 1e9:
                raise RuntimeError('Insufficient space for full artifact plus reserve')
            record['status'] = 'downloading'
            save()
            local = Path(hf_hub_download(REPOSITORY, filename, repo_type='dataset',
                         revision=REVISION, local_dir=root / folder, token=True))
            digest = sha(local)
            if local.stat().st_size != item.size or digest != item.lfs.sha256:
                raise RuntimeError('Downloaded artifact checksum mismatch')
            record.update(status='available_verified', path=str(local.relative_to(root)),
                          bytes=local.stat().st_size, sha256=digest)
            save()
            print(json.dumps(dict(filename=filename, status=record['status'], bytes=record['bytes'])), flush=True)
        state['status'] = 'complete'
        save()
    except Exception as error:
        state.update(status='failed', error_class=type(error).__name__,
                     http_status=getattr(getattr(error, 'response', None), 'status_code', None))
        save()
        print(json.dumps({k: state[k] for k in ('status', 'error_class', 'http_status')}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
