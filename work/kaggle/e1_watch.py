"""Read-only latest-version monitor, maximum75min from the sole push request."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
launch=json.loads((WORK/'checks/e1-launch.json').read_text())
deadline=datetime.fromisoformat(launch['requested_at_utc']).timestamp()+75*60
while datetime.now(timezone.utc).timestamp()<deadline:
    started=time.monotonic()
    try:
        run=subprocess.run([sys.executable,str(ROOT/'e1_api.py'),'observe'],capture_output=True,text=True,timeout=90)
        # e1_api emits only sanitized JSON; never print arbitrary subprocess logs.
        row=json.loads(run.stdout.strip().splitlines()[-1])
        print(json.dumps({k:row[k]for k in ('at_utc','version','status','output_visible','wait_threshold_crossed','error_type','reason')if k in row}),flush=True)
        if run.returncode==0 and row.get('status')in('COMPLETE','ERROR'):break
    except (subprocess.TimeoutExpired,json.JSONDecodeError,IndexError):
        print(json.dumps({'at_utc':datetime.now(timezone.utc).isoformat(),'observation':'temporary read failure; no mutation'}),flush=True)
    time.sleep(max(0.,min(55.-(time.monotonic()-started),deadline-datetime.now(timezone.utc).timestamp())))
else:
    path=WORK/'checks/e1-wait-limit.json'
    path.write_text(json.dumps(dict(at_utc=datetime.now(timezone.utc).isoformat(),wait_limit_minutes=75,
        own_push_calls=launch['own_push_calls'],restart_performed=False),indent=2)+'\n')
    print(json.dumps({'wait_limit_minutes':75,'restart_performed':False}),flush=True)
