"""Serial new-evaluation three-layer video watcher; no initial comparison videos."""
import argparse
from datetime import datetime,timezone
import json
import os
import io
import contextlib
import subprocess
import sys
from pathlib import Path
import time
import torch
from supermimic.run_layout import evaluation_folders,stage_iteration,stage_key
from eval_video import render_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--from-block',type=int,default=0)
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args();run=args.run.resolve();torch.set_num_threads(1);failed=set()
    monitored=set()
    if (run/'logs/consol.jsonl').exists():
        for line in (run/'logs/consol.jsonl').open():
            event=json.loads(line)
            if event.get('record_type')=='evaluation_monitor' and event.get('status')=='generated':monitored.add(event['stage'])
    while True:
        status=json.loads((run/'report.json').read_text())
        ended=status.get('reason') not in (None,'running','signal')
        folders=[]
        for folder in evaluation_folders(run):
            key=stage_key(folder.name)
            if not (key.isdigit() or (key=='final' and ended)):continue
            report=json.loads((folder/'report.json').read_text())
            if stage_iteration(key,report)>=args.from_block:folders.append(folder)
        folders.sort(key=lambda p:stage_iteration(p.name,json.loads((p/'report.json').read_text())),reverse=True)
        for folder in folders:
            if folder.name in failed:continue
            try:
                config=json.loads((folder/'report.json').read_text())['config']
                monitor_clip=config.get('monitor_clip',config.get('train_clip','walk1_subject1_clip0'))
                videos=[]
                for start in (0,200):
                    output=run/'visuals'/stage_key(folder.name)/f'start{start:04d}'
                    if (output/'manifest.json').exists() and any(output.glob('*.mp4')):continue
                    videos.append(render_stage(folder,output,start,clip=monitor_clip))
                if folder.name not in monitored:
                    from eval_log_summary import summarize
                    captured=io.StringIO()
                    with contextlib.redirect_stdout(captured):summarize(run,stage=folder.name)
                    command=[sys.executable,str(Path(__file__).with_name('plot_training_curves.py')),
                             '--run',str(run),'--through-block',str(stage_iteration(folder.name,json.loads((folder/'report.json').read_text())))]
                    subprocess.run(command,check=True)
                    event=dict(record_type='evaluation_monitor',status='generated',stage=folder.name,videos=videos,summary=captured.getvalue())
                    monitored.add(folder.name)
                elif videos:
                    event=dict(record_type='video',status='generated',stage=folder.name,videos=videos)
                else:continue
            except Exception as error:
                failed.add(folder.name);event=dict(record_type='video',status='failed',stage=folder.name,error_class=type(error).__name__,error=str(error))
            event['utc']=datetime.now(timezone.utc).isoformat()
            with (run/'logs/consol.jsonl').open('a') as log:
                log.write(json.dumps(event)+'\n');log.flush();os.fsync(log.fileno())
            print(json.dumps(event),flush=True)
        if args.once or ended:return
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
