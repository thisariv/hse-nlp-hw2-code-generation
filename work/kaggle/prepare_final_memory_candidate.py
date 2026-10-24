"""Prepare an isolated memory candidate and reviewable patch; no API or GPU.

Requires a separately authorized GPU run; does not overwrite submission.
"""

from pathlib import Path
import json, shutil, difflib, hashlib

base = Path(__file__).resolve().parents[2]
submission = base / "submission"
target = base / "work/kaggle/staging/final-package-memory-candidate"
for dirname in ["code", "tests"]:
    shutil.copytree(
        submission / dirname,
        target / dirname,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
shutil.copyfile(submission / "inference.ipynb", target / "inference.ipynb")
package = target / "code/hw2_codegen"
p = package / "config.json"
cfg = json.loads(p.read_text())
cfg.update(
    sequence_budget=640,
    greedy_sequence_budget=640,
    cuda_allocator="expandable_segments:True,garbage_collection_threshold:0.8",
    release_cache_between_batches=True,
)
p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
p = package / "runner.py"
s = p.read_text().replace(
    'CUBLAS_WORKSPACE_CONFIG=":4096:8",',
    'CUBLAS_WORKSPACE_CONFIG=":4096:8",\n        PYTORCH_ALLOC_CONF=config["cuda_allocator"],',
)
s = s.replace(
    '    env["PYTHONPATH"] = (',
    '    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)\n    env["PYTHONPATH"] = (',
)
p.write_text(s)
p = package / "pipeline.py"
s = p.read_text().replace(
    '                    if device.type == "cuda":\n                        torch.cuda.empty_cache()',
    '                if device.type == "cuda" and (oom or config.get("release_cache_between_batches", False)):\n                    torch.cuda.empty_cache()',
)
p.write_text(s)
p = target / "tests/test_v11.py"
p.write_text(
    p.read_text().replace(
        'self.assertEqual(c["sequence_budget"], 768)',
        'self.assertEqual(c["sequence_budget"], 640)',
    )
)
p = target / "inference.ipynb"
nb = json.loads(p.read_text())
for c in nb["cells"]:
    text = "".join(c["source"])
    if c["cell_type"] == "code":
        c.update(
            source=text.replace("14 * 1024**3", "14_000_000_000"),
            outputs=[],
            execution_count=None,
        )
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
patch = []
for f in [
    "code/hw2_codegen/config.json",
    "code/hw2_codegen/runner.py",
    "code/hw2_codegen/pipeline.py",
    "inference.ipynb",
]:
    patch.extend(
        difflib.unified_diff(
            (submission / f).read_text().splitlines(True),
            (target / f).read_text().splitlines(True),
            fromfile="a/" + f,
            tofile="b/" + f,
        )
    )
p = base / "work/checks/final-package-v11-memory-candidate.patch"
p.write_text("".join(patch))
print(target.resolve())
print("Patch SHA:", hashlib.sha256(p.read_bytes()).hexdigest())
