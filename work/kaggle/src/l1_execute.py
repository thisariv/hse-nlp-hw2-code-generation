"""Torch/NumPy-free spawn entrypoint; its parent kills this process group at deadline."""
import ctypes, json, os, signal, sys, threading, time
from pathlib import Path
from exp26_fast_exec import run_chunks
from kaggle_io import write

def guard_parent():
    parent=os.getppid()
    def kill_group(*unused):os.killpg(os.getpgrp(),signal.SIGKILL)
    if parent==1:kill_group()
    if sys.platform.startswith('linux'):
        signal.signal(signal.SIGTERM,kill_group)
        if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError('Parent death guard failed')
        if os.getppid()!=parent:kill_group()
    def watch():
        while os.getppid()==parent:time.sleep(.1)
        kill_group()
    threading.Thread(target=watch,daemon=True,name='executor-parent-guard').start()

if __name__=='__main__':
    guard_parent()
    codes=json.loads(Path(sys.argv[1]).read_text())
    write(sys.argv[2],run_chunks(codes,workers=int(sys.argv[3]),wall_limit=3.,cpu_limit=2.))
