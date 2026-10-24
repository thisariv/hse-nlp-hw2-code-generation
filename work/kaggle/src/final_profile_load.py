"""Torch-free bounded workload driver, using the real frozen final executor."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from final_executor import Executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--nice', type=int, default=0)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    # Service and spawn workers inherit priority; GPU Python priority is untouched.
    nice_error = None
    try:
        effective_nice = os.nice(args.nice) if args.nice else os.getpriority(os.PRIO_PROCESS, 0)
    except PermissionError as error:
        # macOS sandbox can deny even LOWERING priority. Record, never bypass it.
        effective_nice, nice_error = os.getpriority(os.PRIO_PROCESS, 0), str(error)
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    if sys.platform.startswith('linux'):
        import ctypes
        parent = os.getppid()
        assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
        if parent == 1 or os.getppid() != parent:
            return
    executor = Executor(args.workers)
    started = time.monotonic()
    batches, completed = 0, 0
    try:
        executor.submit([{'code': 'def solve():\n    return 1\n'}])
        assert executor.drain(time.monotonic()+20.), 'Load worker startup failed'
        (args.directory/'ready.json').write_text(json.dumps({'workers': args.workers,
            'nice': effective_nice, 'nice_requested_increment': args.nice,
            'nice_applied': nice_error is None, 'nice_error': nice_error}))
        while not stop and time.monotonic()-started < 660:
            # Same jobs and rate across variants. Numeric AST/execution path, no training/gold.
            # ~0.02 CPU sec/task on T4 host, sustained rather than bursty pipeline load.
            jobs = [{'code': f'def solve():\n    x = {batches*32+i}\n    for j in range(250000):\n        x = (x+j)%1000003\n    return x\n'}
                    for i in range(32)]
            executor.submit(jobs)
            while not stop and not executor.drain(time.monotonic()+.05):
                pass
            batches += 1
            completed = len(executor.results)
    finally:
        report = executor.summary()
        report.update(batches=batches, completed=completed, nice=effective_nice, nice_error=nice_error,
                      workload='synthetic sustained numeric programs through frozen executor')
        (args.directory/'load-summary.json').write_text(json.dumps(report, indent=2))
        executor.close()


if __name__ == '__main__':
    main()
