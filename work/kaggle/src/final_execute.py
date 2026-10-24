"""Persistent Torch-free service: fresh numeric namespaces, pristine SymPy forks.

stdin: JSON lists of [id, code], or null to finish. stdout: [id, execution, metrics].
The launcher can kill the complete service process group, including its workers.
"""
import json
import multiprocessing as mp
import os
import queue
import resource
import select
import signal
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import wait
import final_numeric as numeric

MEMORY = 768 * 1024**2
WALL = 3.
CPU = 2.


def kill_program(pid):
    # Each program owns a group; kill descendants even after a successful result.
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)  # Child may not yet have called setsid().
        except ProcessLookupError:
            pass
    until = time.monotonic()+.1
    while time.monotonic() < until:
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:
                return
        except ChildProcessError:
            return
        time.sleep(.001)
    raise RuntimeError('Program could not be reaped; replace pristine worker')


def isolated_task(code, tree, connection):
    reader, writer = os.pipe()
    parent = os.getpid()
    pid = os.fork()  # This parent has no Torch/CUDA and no reader/BLAS threads.
    if pid == 0:
        try:
            os.close(reader)
            connection.close()
            os.setsid()
            if sys.platform.startswith('linux'):
                import ctypes
                signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
                if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
                    os.killpg(os.getpgrp(), signal.SIGKILL)
            resource.setrlimit(resource.RLIMIT_CPU, (int(CPU)+1, int(CPU)+1))
            result = numeric.execute_task((code, WALL, CPU), tree=tree, isolated=True)
            os.write(writer, json.dumps(result, separators=(',', ':')).encode())
        finally:
            os._exit(0)
    os.close(writer)
    try:
        # Same outer limit as v2; catches code that repeatedly catches timer exceptions.
        if select.select([reader], [], [], WALL+2.)[0]:
            data = os.read(reader, 4096)
            if data:
                return json.loads(data)
        return {'status': 'timeout_or_worker_exit'}
    finally:
        os.close(reader)
        kill_program(pid)


def worker(connection):
    # Library diagnostics and user output cannot corrupt the JSON protocol.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    numeric.worker_init(MEMORY)
    pristine = None
    while True:
        item = connection.recv()
        if item is None:
            return
        index, code = item
        started = time.monotonic()
        used = time.process_time()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        tree, status = numeric.filter_program(code)
        if status:
            result = {'status': status}
        elif any((isinstance(node, numeric.ast.Import) and any(a.name == 'sympy' for a in node.names)) or
                 (isinstance(node, numeric.ast.ImportFrom) and node.module == 'sympy')
                 for node in numeric.ast.walk(tree)):
            if pristine is None:
                numeric.prepare_pristine_worker()
                pristine = dict(numeric._MODULES)
            numeric._MODULES.clear()
            numeric._MODULES.update(pristine)
            result = isolated_task(code, tree, connection)
        else:
            # Only builtins and a fresh math namespace are reachable on this path.
            # No user code can access SymPy objects/caches; env and timers are per task.
            result = numeric.execute_task((code, WALL, CPU), tree=tree)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        connection.send((index, result, {'task_wall_seconds': time.monotonic()-started,
            'task_cpu_seconds': time.process_time()-used + after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime}))


def serve(workers):
    inbox = queue.Queue()
    def read():
        for line in sys.stdin:
            inbox.put(json.loads(line))
        inbox.put(None)
    threading.Thread(target=read, daemon=True).start()
    context = mp.get_context('spawn')
    slots = []
    pending = deque()
    finished = False
    def start():
        parent, child = context.Pipe()
        process = context.Process(target=worker, args=(child,), daemon=True)
        process.start()
        child.close()
        return [process, parent, None, 0.]
    def stop(slot):
        process, pipe, _, _ = slot
        if process.is_alive():
            process.kill()
        process.join(timeout=.05)
        pipe.close()
    def emit(index, result, metrics=None):
        print(json.dumps([index, result, metrics]), flush=True)
    try:
        slots = [start() for _ in range(workers)]
        while not finished or pending or any(s[2] is not None for s in slots):
            while True:
                try:
                    batch = inbox.get_nowait()
                except queue.Empty:
                    break
                if batch is None:
                    finished = True
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
                if pipe_failed or not process.is_alive() or (slot[2] is not None and time.monotonic()-submitted >= WALL+2.5):
                    if slot[2] is not None:
                        emit(slot[2], {'status': 'timeout_or_worker_exit'})
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
                wait(active_pipes, timeout=.002)  # Wake immediately, no per-program 1ms polling tax.
            else:
                time.sleep(.002)
    finally:
        for slot in slots:
            stop(slot)


if __name__ == '__main__':
    # Independent supervisor kills the process group on parent death (Linux).
    if sys.platform.startswith('linux'):
        import ctypes
        parent = os.getppid()
        signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
        assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
        if parent == 1 or os.getppid() != parent:
            os.killpg(os.getpgrp(), signal.SIGKILL)
    serve(int(sys.argv[1]))
