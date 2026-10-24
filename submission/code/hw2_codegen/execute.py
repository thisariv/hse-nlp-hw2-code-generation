"""Асинхронный CPU executor: дедупликация исполнения без потери голосов.

AST-фильтр, wall 3s / CPU 2s / Linux AS 768 MiB, внешнее kill/reap и
новые namespaces ограничивают программы датасета. Это не OS sandbox для
произвольного враждебного Python; сеть/файлы недоступны через allowed API.
"""

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class Executor:
    """Фоновый сервис; submit не блокирует GPU, drain ограничен deadline."""

    def __init__(self, workers=None):
        self.workers = workers if workers is not None else 2
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or (not 1 <= self.workers <= 4)
        ):
            raise ValueError("Require one to four execution workers")
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parent.parent)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONHASHSEED="0",
        )
        self.child = subprocess.Popen(
            [sys.executable, "-m", "hw2_codegen._service", str(self.workers)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self.service_state = {}
        self.codes = {}
        self.results = {}
        self.error = None

        def read():
            try:
                for line in self.child.stdout:
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self.service_state = message
                        continue
                    index, value, *_ = message
                    self.results[index] = value
            except Exception as error:
                self.error = type(error).__name__

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def resize(self, workers):
        """Добавить workers, сохранив очередь и кеш результатов."""
        if not isinstance(workers, int) or not self.workers <= workers <= 4:
            raise ValueError("Only expansion up to four workers is supported")
        self.child.stdin.write(json.dumps({"workers": workers}) + "\n")
        self.child.stdin.flush()
        self.workers = workers

    def submit(self, candidates):
        """Послать только новые программы, сохранив отдельные голоса кандидатов."""
        jobs = []
        for candidate in candidates:
            code = candidate["code"]
            if code not in self.codes:
                index = len(self.codes)
                self.codes[code] = index
                jobs.append([index, code])
        if jobs:
            self.child.stdin.write(json.dumps(jobs) + "\n")
            self.child.stdin.flush()

    def attach(self, candidates):
        """Подставить исполнение либо явный budget_not_evaluated."""
        for candidate in candidates:
            index = self.codes[candidate["code"]]
            candidate["execution"] = self.results.get(
                index, {"status": "budget_not_evaluated"}
            )

    def drain(self, deadline):
        """Дождаться результатов и запуска workers до заданного срока."""
        while (
            len(self.results) < len(self.codes)
            or self.service_state.get("workers") != self.workers
        ) and time.monotonic() < deadline:
            exit_code = self.child.poll()
            if exit_code is not None:
                self.error = self.error or "executor_exit_" + str(exit_code)
                break
            if self.error:
                break
            time.sleep(0.002)
        return (
            len(self.results) == len(self.codes)
            and self.service_state.get("workers") == self.workers
        )

    def close(self):
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.child.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
        self.child.stdin.close()
        self.reader.join(timeout=0.1)
        self.child.stdout.close()


def verify_winners(winners, deadline, workers=None):
    """Fresh service: compare the actual chosen CODE against its voted execution."""
    executor = Executor(workers)
    eligible = [
        i for i, w in enumerate(winners) if w["execution"]["status"] == "numeric"
    ]
    items = [{"code": winners[i]["code"]} for i in eligible]
    try:
        executor.submit(items)
        complete = executor.drain(deadline)
        executor.attach(items)
        mismatches = [
            i
            for i, item in zip(eligible, items)
            if item["execution"]["status"] != "numeric"
            or not math.isclose(
                item["execution"]["value"],
                winners[i]["execution"]["value"],
                rel_tol=1e-06,
                abs_tol=1e-06,
            )
        ]
        return {
            "checked_rows": len(eligible),
            "complete": complete,
            "mismatch_rows": mismatches,
            "completed_unique": len(executor.results),
        }
    finally:
        executor.close()
