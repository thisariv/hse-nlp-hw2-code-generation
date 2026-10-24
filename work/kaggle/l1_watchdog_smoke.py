#!/usr/bin/env python3
"""Real CPU process-group TERM/KILL and hard-exit fuses for the long session."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from l1_session import Watchdog
from kaggle_io import write

def main():
    tmp=Path(tempfile.mkdtemp(prefix='nlp-l1-watchdog-'));start=time.monotonic()
    long={'working_root':str(tmp),'session_start_monotonic':start,'deadline_monotonic':start+40680,
          'hard_deadline_monotonic':start+42300,'budget_seconds':40680,'terminate_grace_seconds':.1,'session_kind':'l1'}
    w=Watchdog(long);assert abs(w.hard_deadline-start-42300)<.001
    # A SIGTERM-resistant descendant is actually killed; no unbounded wait.
    child=subprocess.Popen([sys.executable,'-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"],start_new_session=True)
    w.register(child);time.sleep(.1);w.terminate();time.sleep(.1);w.kill();child.wait(timeout=2);assert child.returncode!=0
    script=tmp/'fuse.py'
    script.write_text("import sys,time\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(ROOT/'src'))+")\nfrom l1_session import Watchdog\ns=time.monotonic()\nc="+repr(dict(long,working_root=str(tmp/'fuse')))+"\nc.update(session_start_monotonic=s,deadline_monotonic=s+.1,hard_deadline_monotonic=s+.4)\nWatchdog(c).start()\ntime.sleep(30)\n")
    started=time.monotonic();p=subprocess.run([sys.executable,str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=3)
    assert p.returncode==0 and time.monotonic()-started<2
    receipt=json.loads((tmp/'fuse/session-summary.json').read_text());assert receipt['watchdog_forced_exit'] is True
    result={'passed':True,'checks':['L1 hard deadline42300s is not clamped to old S2 13200s',
          'real SIGTERM-resistant child group killed','real notebook hard fuse exits within deadline and writes partial receipt'],
          'real_CUDA_D_state_verified':False}
    write(ROOT.parent/'checks/kaggle-l1-watchdog-smoke.json',result);print(json.dumps(result))

if __name__=='__main__':main()
