"""V6 execution cache with nice10 service and lossless 2 -> up to4 expansion."""
import math
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
        self.workers = workers if workers is not None else 2
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or not 1 <= self.workers <= 4:
            raise ValueError('Require one to four execution workers')
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONHASHSEED='0')
        self.child = subprocess.Popen([sys.executable, str(service_path or Path(__file__).with_name('final_execute_v6.py')),
                                      str(self.workers)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
                                      start_new_session=True)
        self.service_state = {}
        self.worker_history = [self.workers]
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
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self.service_state = message
                        continue
                    index, value, *metadata = message
                    if metadata and metadata[0]:
                        self.task_wall_seconds += metadata[0]['task_wall_seconds']
                        self.task_cpu_seconds += metadata[0]['task_cpu_seconds']
                    self.last_result = time.monotonic()
                    self.results[index] = value
            except Exception as error:
                self.error = type(error).__name__
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def resize(self, workers):
        if not isinstance(workers, int) or not self.workers <= workers <= 4:
            raise ValueError('Only expansion up to four workers is supported')
        self.child.stdin.write(json.dumps({'workers': workers})+'\n')
        self.child.stdin.flush()
        self.workers = workers
        self.worker_history.append(workers)

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
        while (len(self.results) < len(self.codes) or self.service_state.get('workers') != self.workers) and time.monotonic() < deadline:
            exit_code = self.child.poll()
            if exit_code is not None:
                self.error = self.error or 'executor_exit_'+str(exit_code)
                break
            if self.error:
                break
            time.sleep(.002)
        return len(self.results) == len(self.codes) and self.service_state.get('workers') == self.workers

    def summary(self):
        return {'workers': self.workers, 'worker_history': self.worker_history,
                'service_state': dict(self.service_state), 'requested_candidates': self.requests,
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


def verify_winners(winners, deadline, workers=None):
    """Fresh service: compare the actual chosen CODE against its voted execution."""
    executor = Executor(workers)
    eligible = [i for i, w in enumerate(winners) if w['execution']['status'] == 'numeric']
    items = [{'code': winners[i]['code']} for i in eligible]
    try:
        executor.submit(items)
        complete = executor.drain(deadline)
        executor.attach(items)
        mismatches = [i for i, item in zip(eligible, items) if
                      item['execution']['status'] != 'numeric' or not math.isclose(
                          item['execution']['value'], winners[i]['execution']['value'],
                          rel_tol=1e-6, abs_tol=1e-6)]
        return {'checked_rows': len(eligible), 'complete': complete,
                'mismatch_rows': mismatches, 'execution': executor.summary()}
    finally:
        executor.close()
