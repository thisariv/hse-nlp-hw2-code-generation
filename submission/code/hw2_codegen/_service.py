"""Torch-free очередь с заменой зависших workers и расширением 2 → 4."""

import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import wait

from ._worker import WALL, worker


def lower_priority():
    error = None
    try:
        effective = os.nice(10)
    except PermissionError as exc:
        effective, error = os.getpriority(os.PRIO_PROCESS, 0), str(exc)
    return {
        "nice_requested_increment": 10,
        "nice": effective,
        "nice_applied": error is None,
        "nice_error": error,
    }


def serve(workers, priority):
    inbox = queue.Queue()

    def read():
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put(None)

    threading.Thread(target=read, daemon=True).start()
    context = mp.get_context("spawn")
    slots = []
    pending = deque()
    finished = False

    def start():
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(child,), daemon=True)
        process.start()
        child.close()
        return [process, parent, None, 0.0]

    def stop(slot):
        process, pipe, _, _ = slot
        if process.is_alive():
            process.kill()
        process.join(timeout=0.05)
        pipe.close()

    def emit(index, result, metrics=None):
        print(json.dumps([index, result, metrics]), flush=True)

    try:
        slots = [start() for _ in range(workers)]

        def state():
            print(json.dumps({"workers": len(slots), **priority}), flush=True)

        state()
        while not finished or pending or any(s[2] is not None for s in slots):
            while True:
                try:
                    batch = inbox.get_nowait()
                except queue.Empty:
                    break
                if batch is None:
                    finished = True
                elif isinstance(batch, dict):
                    target = batch["workers"]
                    if not isinstance(target, int) or not len(slots) <= target <= 4:
                        raise ValueError(
                            "Only expansion up to four workers is supported"
                        )
                    slots.extend(start() for _ in range(target - len(slots)))
                    state()
                else:
                    pending.extend(batch)
            for number, slot in enumerate(slots):
                process, pipe, active, submitted = slot
                pipe_failed = False
                # Worker enforces wall+2; allow a bounded IPC/reap margin here.
                try:
                    if active is not None and pipe.poll():
                        index, result, *metadata = pipe.recv()
                        assert index == active
                        emit(index, result, metadata[0] if metadata else None)
                        slot[2] = None
                except (EOFError, OSError):
                    pipe_failed = True
                if (
                    pipe_failed
                    or not process.is_alive()
                    or (
                        slot[2] is not None
                        and time.monotonic() - submitted >= WALL + 2.5
                    )
                ):
                    if slot[2] is not None:
                        emit(slot[2], {"status": "timeout_or_worker_exit"})
                    stop(slot)
                    slots[number] = slot = start()
                    process, pipe = slot[:2]
                if slot[2] is None and pending:
                    index, code = pending.popleft()
                    try:
                        pipe.send((index, code))
                        slot[2:] = [index, time.monotonic()]
                    except (EOFError, OSError):
                        # Death can race is_alive(): this job was not accepted.
                        pending.appendleft((index, code))
                        stop(slot)
                        slots[number] = start()
            active_pipes = [slot[1] for slot in slots if slot[2] is not None]
            if active_pipes:
                wait(
                    active_pipes, timeout=0.002
                )  # Wake immediately, no per-program 1ms polling tax.
            else:
                time.sleep(0.002)
    finally:
        for slot in slots:
            stop(slot)


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        import ctypes

        parent = os.getppid()
        signal.signal(
            signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL)
        )
        assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
        if parent == 1 or os.getppid() != parent:
            os.killpg(os.getpgrp(), signal.SIGKILL)
    serve(int(sys.argv[1]), lower_priority())
