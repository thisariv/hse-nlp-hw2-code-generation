#!/usr/bin/env python3
"""Pinned read-only status polling; no restart or push; bounded wait."""
import argparse, datetime, json, subprocess, time
from pathlib import Path
from kaggle_tools import kaggle_python
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('ref');p.add_argument('--key',required=True);p.add_argument('--interval',type=int,default=240)
    p.add_argument('--wait-seconds',type=int,default=9000);p.add_argument('--initial-delay',type=int,default=0);a=p.parse_args()
    start=time.monotonic();time.sleep(a.initial_delay)
    while True:
        result=subprocess.run([kaggle_python(),str(ROOT/'kaggle_versioned.py'),'status',a.ref],capture_output=True,text=True,timeout=90)
        state=result.stdout.strip().upper() if result.returncode==0 else 'STATUS_CHECK_FAILED'
        if state not in ('QUEUED','RUNNING','COMPLETE','ERROR','CANCEL_REQUESTED','CANCEL_ACKNOWLEDGED','STATUS_CHECK_FAILED'):
            state='UNKNOWN'
        checked=datetime.datetime.now(datetime.timezone.utc).isoformat()
        runs=ROOT/'runs.json';data=json.loads(runs.read_text());row=data[a.key]
        row.update(status=state,checked_at=checked);row.setdefault('status_history',[]).append({'status':state,'checked_at':checked})
        temp=runs.with_suffix('.json.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(runs)
        print(checked,a.ref,state,flush=True)
        if state in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'):break
        if time.monotonic()-start>=a.wait_seconds:
            print('WAIT_LIMIT_REACHED; no repush/restart',flush=True);break
        time.sleep(min(a.interval,max(0,a.wait_seconds-(time.monotonic()-start))))

if __name__=='__main__':main()
