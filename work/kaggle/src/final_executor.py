"""Incremental global execution cache, usable while the GPU generates next batch."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class Executor:
    def __init__(self, workers=None, service_path=None):
        self.workers = workers if workers is not None else (os.cpu_count() or 1)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONHASHSEED='0')
        self.child = subprocess.Popen([sys.executable, str(service_path or Path(__file__).with_name('final_execute.py')),
                                      str(self.workers)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
                                      start_new_session=True)
        self.codes = {}
        self.results = {}
        self.requests = 0
        self.started = time.monotonic()
        self.error = None
        self.task_wall_seconds = self.task_cpu_seconds = 0.
        self.first_submit = self.last_result = None
        def read():
            try:
                for line in self.child.stdout:
                    index, value, *metadata = json.loads(line)
                    if metadata and metadata[0]:
                        self.task_wall_seconds += metadata[0]['task_wall_seconds']
                        self.task_cpu_seconds += metadata[0]['task_cpu_seconds']
                    self.last_result = time.monotonic()
                    self.results[index] = value
            except Exception as error:
                self.error = type(error).__name__
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def submit(self, candidates):
        jobs = []
        for candidate in candidates:
            self.requests += 1
            code = candidate['code']
            if code not in self.codes:
                index = len(self.codes)
                self.codes[code] = index
                jobs.append([index, code])
        if jobs:
            if self.first_submit is None:
                self.first_submit = time.monotonic()
            self.child.stdin.write(json.dumps(jobs) + '\n')
            self.child.stdin.flush()

    def attach(self, candidates):
        for candidate in candidates:
            index = self.codes[candidate['code']]
            candidate['execution'] = self.results.get(index, {'status': 'budget_not_evaluated'})

    def drain(self, deadline):
        while len(self.results) < len(self.codes) and time.monotonic() < deadline:
            exit_code = self.child.poll()
            if exit_code is not None:
                self.error = self.error or 'executor_exit_'+str(exit_code)
                break
            if self.error:
                break
            time.sleep(.002)
        return len(self.results) == len(self.codes)

    def summary(self):
        return {'workers': self.workers, 'requested_candidates': self.requests,
                'unique_programs': len(self.codes), 'deduplicated': self.requests-len(self.codes),
                'completed_unique': len(self.results), 'service_wall_seconds': time.monotonic()-self.started,
                'sum_worker_task_wall_seconds': self.task_wall_seconds,
                'sum_worker_task_cpu_seconds': self.task_cpu_seconds,
                'first_submit_to_last_result_seconds': (self.last_result-self.first_submit
                    if self.last_result is not None and self.first_submit is not None else None),
                'service_wall_includes_GPU_wait': True,
                'wall_limit': 3., 'cpu_limit': 2., 'memory_bytes': 768*1024**2,
                'overlap': True, 'isolation': 'fresh numeric namespaces / pristine SymPy fork', 'error': self.error}

    def close(self):
        # Never pool.join() unbounded in the CUDA process.
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.child.wait(timeout=.1)
        except subprocess.TimeoutExpired:
            pass
        self.child.stdin.close()
        self.reader.join(timeout=.1)
        self.child.stdout.close()
