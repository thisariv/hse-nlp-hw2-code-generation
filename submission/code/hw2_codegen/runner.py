"""Torch-free supervisor для одного cold pipeline, без Kaggle API и credentials."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class DeviceMemory:
    """NVML device-wide used memory, sampled every supervisor poll (20 ms).

    Includes CUDA context and allocator cache. Torch peak allocated/reserved
    are recorded separately by the worker and are exact allocator counters.
    """

    def __init__(self, enabled):
        self.peak = None
        self.lib = self.handle = None
        if not enabled:
            return
        try:
            import ctypes

            class Memory(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in ("total", "free", "used")]

            lib = ctypes.CDLL("libnvidia-ml.so.1")
            assert lib.nvmlInit_v2() == 0
            handle = ctypes.c_void_p()
            assert lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0
            self.lib, self.handle, self.memory = lib, handle, Memory()
            self.ctypes = ctypes
        except (OSError, AssertionError):
            return

    def sample(self):
        if (
            self.lib is not None
            and self.lib.nvmlDeviceGetMemoryInfo(
                self.handle, self.ctypes.byref(self.memory)
            )
            == 0
        ):
            self.peak = max(self.peak or 0, self.memory.used)


def run_cold(config):
    """Запустить свежий процесс, сохранить timer и готовый greedy при hard stop.

    Нельзя повторно использовать каталог предыдущего запуска. Timer начинается
    до создания процесса и включает imports/load/warmup/decode/exec/vote/write.
    Диагностика и оценка tune идут после закрытия answers.jsonl.
    """
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / n).exists() for n in ["timing.json", "answers.jsonl", "worker.log"]
    ):
        raise ValueError("Cold run requires a fresh output directory")
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    start = time.monotonic()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        HW2_PROCESS_START=str(start),
        PYTHONHASHSEED="0",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTORCH_ALLOC_CONF=config["cuda_allocator"],
    )
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent.parent)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    hard_stop = config.get("guards", {}).get("supervisor", 172.0)
    stopped = False
    memory = DeviceMemory(not config.get("dry_run", False))
    with (output / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-m", "hw2_codegen.worker", str(config_path.resolve())],
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            while child.poll() is None:
                memory.sample()
                if time.monotonic() - start >= hard_stop:
                    stopped = True
                    break
                time.sleep(0.02)
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=2.0)
    timer = output / "timing.json"
    if child.returncode or stopped or not timer.is_file():
        if not (output / "greedy-ready.json").is_file():
            raise RuntimeError(
                "Cold worker failed before greedy completion; inspect worker.log"
            )
        (output / "answers.jsonl").write_bytes(
            (output / "greedy-fallback.jsonl").read_bytes()
        )
        report = dict(
            elapsed_seconds=time.monotonic() - start,
            guards=["supervisor"],
            rows=json.loads((output / "greedy-ready.json").read_text())["rows"],
            path=str(output / "answers.jsonl"),
            hard_stop=stopped,
            exit_code=child.returncode,
        )
        report["full_pipeline_seconds"] = report["elapsed_seconds"]
        timer.write_text(json.dumps(report, indent=2) + "\n")
    else:
        report = json.loads(timer.read_text())
    report["peak_device_used_bytes"] = memory.peak
    report["device_memory_sampling_seconds"] = 0.02
    report["status"] = "guarded" if report["guards"] else "complete"
    report["within_budget"] = report["elapsed_seconds"] <= config.get(
        "budget_seconds", 180.0
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
