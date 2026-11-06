"""Refresh catalog once after the independent SOMA acquisition finishes.

No downloading, extraction or archive-member scanning. build_catalog performs
its existing artifact identity checks and exactly one catalog validation pass.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--poll-seconds',type=float,default=45)
    args=parser.parse_args()
    if not 30 <= args.poll_seconds <= 60:
        parser.error('Polling interval must be 30–60 seconds')
    root=args.project.resolve();folder=root/'dataset/manifests/bones_seed'
    acquisition=folder/'soma-acquisition.json'
    report_path=folder/'catalog-finalization.json'
    lock=(folder/'catalog-finalization.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('A catalog finalizer already owns this queue')
    state=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='waiting',
               acquisition=str(acquisition),script_sha256=digest(Path(__file__)),
               operation='Wait for SOMA completion, then build_catalog including its one validation pass; no member scan/extraction')

    def save():
        temporary=report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(report_path)

    save()
    while True:
        try:
            current=json.loads(acquisition.read_text()) if acquisition.exists() else {}
        except json.JSONDecodeError:
            # Evidence writer uses atomic replace; tolerate a transient unreadable
            # observation without interpreting it as a successful completion.
            time.sleep(args.poll_seconds);continue
        if current.get('status')=='failed':
            state.update(status='acquisition_failed',finished_utc=datetime.now(timezone.utc).isoformat(),
                         acquisition_sha256=digest(acquisition),error_class=current.get('error_class'),
                         http_status=current.get('http_status'))
            save();print(json.dumps(state),flush=True);return 1
        if current.get('status')=='complete':
            expected={'soma_uniform.tar.gz','soma_proportional.tar.gz'}
            records={r['filename']:r for r in current.get('files',[])}
            if not expected <= records.keys() or not all(records[n].get('status')=='available_verified' for n in expected):
                state.update(status='invalid_completion_evidence',finished_utc=datetime.now(timezone.utc).isoformat())
                save();print(json.dumps(state),flush=True);return 1
            break
        time.sleep(args.poll_seconds)
    state.update(status='rebuilding',acquisition_sha256=digest(acquisition),
                 builder_sha256=digest(root/'dataset/build_catalog.py'))
    save()
    command=[sys.executable,str(root/'dataset/build_catalog.py'),'--project',str(root)]
    state['command']=command
    logfile=folder/'catalog-finalization.build.log'
    environment=os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE']='1'
    environment['OMP_NUM_THREADS']='1'
    environment['OPENBLAS_NUM_THREADS']='1'
    # Preserve the caller's isolated optional data dependencies while making
    # the project's motion-name module importable from any working directory.
    environment['PYTHONPATH']=str(root/'src')+os.pathsep+environment.get('PYTHONPATH','')
    started=time.monotonic()
    with logfile.open('w') as log:
        result=subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
    state.update(status='complete' if result.returncode==0 else 'catalog_build_failed',
                 returncode=result.returncode,build_seconds=time.monotonic()-started,
                 finished_utc=datetime.now(timezone.utc).isoformat(),build_log=str(logfile))
    if result.returncode==0:
        validation_path=root/'dataset/catalog.validation.json'
        validation=json.loads(validation_path.read_text())
        catalog_path=root/'dataset/catalog.json';catalog=json.loads(catalog_path.read_text())
        soma=[x for x in catalog['artifacts'] if x['id'].startswith('source_soma_')]
        state.update(catalog_sha256=digest(catalog_path),validation_sha256=digest(validation_path),
                     validation_valid=validation['valid'],soma_artifacts=[{k:x[k] for k in ('id','status','local_available','bytes','sha256')} for x in soma],
                     scope='Verified compressed archives only; SOMA BVH members/schema remain unparsed and unknown')
        if not validation['valid'] or len(soma)!=2 or not all(x['local_available'] and x['status']=='available_verified' for x in soma):
            state['status']='catalog_postcondition_failed'
    save();print(json.dumps(state),flush=True)
    return 0 if state['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
