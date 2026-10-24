"""Изолированное исполнение: fresh env для math, pristine fork для SymPy."""

import json
import os
import resource
import select
import signal
import sys
import time

from . import _numeric as numeric

MEMORY = 768 * 1024**2
WALL, CPU = 3.0, 2.0


def kill_program(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    until = time.monotonic() + 0.1
    while time.monotonic() < until:
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:
                return
        except ChildProcessError:
            return
        time.sleep(0.001)
    raise RuntimeError("Program could not be reaped; replace pristine worker")


def isolated_task(code, tree, connection):
    reader, writer = os.pipe()
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(reader)
            connection.close()
            os.setsid()
            if sys.platform.startswith("linux"):
                import ctypes

                signal.signal(
                    signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL)
                )
                if (
                    ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0
                    or os.getppid() != parent
                ):
                    os.killpg(os.getpgrp(), signal.SIGKILL)
            resource.setrlimit(resource.RLIMIT_CPU, (int(CPU) + 1, int(CPU) + 1))
            result = numeric.execute_task((code, WALL, CPU), tree=tree, isolated=True)
            os.write(writer, json.dumps(result, separators=(",", ":")).encode())
        finally:
            os._exit(0)
    os.close(writer)
    try:
        if select.select([reader], [], [], WALL + 2.0)[0]:
            data = os.read(reader, 4096)
            if data:
                return json.loads(data)
        return {"status": "timeout_or_worker_exit"}
    finally:
        os.close(reader)
        kill_program(pid)


def worker(connection):
    with open(os.devnull, "w") as sink:
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
            result = {"status": status}
        elif any(
            isinstance(node, numeric.ast.Import)
            and any(a.name == "sympy" for a in node.names)
            or (isinstance(node, numeric.ast.ImportFrom) and node.module == "sympy")
            for node in numeric.ast.walk(tree)
        ):
            if pristine is None:
                numeric.prepare_pristine_worker()
                pristine = dict(numeric._MODULES)
            numeric._MODULES.clear()
            numeric._MODULES.update(pristine)
            result = isolated_task(code, tree, connection)
        else:
            result = numeric.execute_task((code, WALL, CPU), tree=tree)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        connection.send(
            (
                index,
                result,
                {
                    "task_wall_seconds": time.monotonic() - started,
                    "task_cpu_seconds": time.process_time()
                    - used
                    + after.ru_utime
                    + after.ru_stime
                    - before.ru_utime
                    - before.ru_stime,
                },
            )
        )
