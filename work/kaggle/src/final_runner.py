"""Torch-free process supervisor. Hard stop and greedy publication share the honest timer."""
import time
import os
START = float(os.environ.get('FINAL_PROCESS_START', time.monotonic()))
import json
import signal
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__' and sys.platform.startswith('linux'):
    import ctypes
    _parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
    if _parent == 1 or os.getppid() != _parent:
        os.killpg(os.getpgrp(), signal.SIGKILL)


def supervise(config_file, start=None):
    start = START if start is None else start
    config = json.loads(Path(config_file).read_text())
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    old = [name for name in ('answers.jsonl', 'greedy-fallback.jsonl', 'greedy-ready.json',
                             'timing.json', 'summary.json', 'worker.log') if (out/name).exists()]
    if old:
        raise ValueError('Fresh cold run requires a new output directory; existing artifacts: '+','.join(old))
    budget = float(config.get('budget_seconds', 180.))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', FINAL_PROCESS_START=str(start),
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    child = None
    stopped = False
    with (out/'worker.log').open('w') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('final_inference.py')), str(config_file)],
                                 env=env, start_new_session=True, stdout=log, stderr=log)
        try:
            while child.poll() is None:
                if time.monotonic() >= start+budget-min(8., budget*.045):
                    stopped = True
                    break
                time.sleep(.01)
        finally:
            # Executor is a separate session; its Linux parent-death guard kills the tree.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                pass
    timer_path = out/'timing.json'
    if stopped or child.returncode or not timer_path.exists():
        # Fatal GPU/driver/init failure before greedy completion cannot produce real greedy.
        # Fail explicitly, never fabricate code or silently claim the time gate passed.
        ready = out/'greedy-ready.json'
        if not ready.exists():
            report = {'status': 'failed_before_greedy', 'elapsed_seconds': time.monotonic()-start,
                      'exit_code': child.returncode, 'hard_stop': stopped}
            (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            raise RuntimeError('Cold pipeline failed before greedy; inspect worker.log')
        data = (out/'greedy-fallback.jsonl').read_bytes()
        with (out/'answers.jsonl').open('wb') as stream:
            stream.write(data)
        closed = time.monotonic()
        timing = {'full_pipeline_seconds': closed-start, 'rows': json.loads(ready.read_text())['rows'],
                  'deadline_guard': {'triggered': True, 'supervisor': True, 'fallback_rows': 'all',
                                     'fallback_count': json.loads(ready.read_text())['rows']},
                  'phases_seconds': {}, 'worker_exit_code': child.returncode, 'hard_stop': stopped}
        timer_path.write_text(json.dumps(timing, indent=2)+'\n')
    else:
        timing = json.loads(timer_path.read_text())
    timing['status'] = 'guarded' if timing['deadline_guard']['triggered'] else 'complete'
    timing['under_180_seconds'] = timing['full_pipeline_seconds'] < 180.
    timing['target_150_seconds'] = timing['full_pipeline_seconds'] <= 150.
    timing['target_120_seconds'] = timing['full_pipeline_seconds'] <= 120.
    (out/'summary.json').write_text(json.dumps(timing, indent=2)+'\n')
    assert timing['full_pipeline_seconds'] < budget, 'Budget exceeded: reject this run'
    return timing


if __name__ == '__main__':
    report = supervise(sys.argv[1])
    print(json.dumps({k: report[k] for k in ['status', 'rows', 'full_pipeline_seconds']}, separators=(',', ':')))
