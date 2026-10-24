"""Cold worker: timestamp перед Torch import; один split на одном процессе/GPU."""

import os
import signal
import sys
import time

STARTED = float(os.environ.get("HW2_PROCESS_START", time.monotonic()))
if sys.platform.startswith("linux"):
    import ctypes

    parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
    if parent == 1 or os.getppid() != parent:
        os.killpg(os.getpgrp(), signal.SIGKILL)

import json
from pathlib import Path
from .pipeline import run_inference

if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    fixture = {}
    if config.get("dry_run"):
        from .dry_run import fixtures

        rows, models, tok, table, specs = fixtures()
        if config.get("variant"):
            # Exercise the exact variant policy with cheap CPU fixtures.
            tiny = list(models.values())
            models = {m["name"]: tiny[i % len(tiny)] for i, m in enumerate(config["models"])}
        else:
            config["models"] = specs
        config.update(cap=32, sequence_budget=16, greedy_sequence_budget=16)
        fixture = dict(rows=rows, models=models, tok=tok, table=table)
    run_inference(config, started=STARTED, **fixture)
