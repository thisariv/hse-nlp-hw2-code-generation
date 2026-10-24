"""Optional CPU parity using all three real FP16 exports (promoted to FP32).

Three original test questions, cap64, identical budget14; no GPU.
Requires downloaded final weights in work/kaggle/staging/final-package-v2 (or HW2_WEIGHTS_STAGE).
The generic mixed-model test remains v11_reference_check.py.
"""

import ast
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch

SUBMISSION = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("HW2_WORK", SUBMISSION.parent / "work"))
sys.path.insert(0, str(SUBMISSION / "code"))
from hw2_codegen import model, pipeline
from hw2_codegen.dry_run import fixtures


def main():
    torch.set_num_threads(1)
    notebook = json.loads(
        (WORK / "kaggle/notebooks/nlp-hw02-ensemble-inference-v11.ipynb").read_text()
    )
    code = next(c["source"] for c in notebook["cells"] if c["cell_type"] == "code")
    sources = ast.literal_eval(
        next(
            n.value
            for n in ast.parse(code).body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "SOURCES"
        )
    )
    with tempfile.TemporaryDirectory(prefix="hw2-v11-parity-") as tmp:
        root = Path(tmp)
        for name, source in sources.items():
            (root / name).write_text(source)
        sys.path.insert(0, str(root))
        import final_inference as baseline
        import final_core as original

        config = json.loads((SUBMISSION / "code/hw2_codegen/config.json").read_text())
        stage = Path(os.environ.get("HW2_WEIGHTS_STAGE", WORK / "kaggle/staging/final-package-v2"))
        rows = json.loads((stage / "test.json").read_text())[:3]
        tok, table = pipeline.tokenizer(stage / "tokenizer.json")
        specs = [dict(m, path="unused") for m in config["models"]]
        clean, old = {}, {}
        for spec in specs:
            path = stage / spec["filename"]
            assert pipeline.sha256(path) == spec["sha256"]
            clean[spec["name"]] = pipeline.load_model(path, torch.device("cpu"))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            previous = original.SimpleTransformer(
                **payload["model_spec"], dropout=0.0
            ).eval()
            previous.load_state_dict(payload["model"], strict=True)
            old[spec["name"]] = previous
        rows_path = root / "rows.json"
        rows_path.write_text(json.dumps(rows))
        for lanes in [14]:
            cfg = dict(
                models=specs,
                rows=str(rows_path),
                tokenizer="unused",
                verify_weights=False,
                device="cpu",
                cap=64,
                sequence_budget=lanes,
                greedy_sequence_budget=lanes,
            )
            groups_old, groups_new = [], []

            def observe(store, fn):
                def run(*args, **kwargs):
                    value = fn(*args, **kwargs)
                    store.append(copy.deepcopy(value))
                    return value

                return run

            iterator = iter(old.values())
            with patch.object(
                baseline, "load", side_effect=lambda *a: next(iterator)
            ), patch.object(
                baseline, "tokenizer", return_value=(tok, table)
            ), patch.object(
                torch, "set_num_interop_threads"
            ), patch.object(
                baseline, "PROCESS_START", time.monotonic()
            ), patch.object(
                baseline, "pack", side_effect=observe(groups_old, baseline.pack)
            ):
                ref = baseline.run(dict(cfg, output=str(root / f"old-{lanes}")))
            with patch.object(
                pipeline, "pack", side_effect=observe(groups_new, pipeline.pack)
            ):
                got = pipeline.run_inference(
                    dict(cfg, output=str(root / f"new-{lanes}")),
                    rows=rows,
                    models=clean,
                    tok=tok,
                    table=table,
                )
            assert (
                groups_old == groups_new
            ), "tokens / log probabilities / restoration differ"
            assert sorted(got["stopped"]) == ref["adaptive"]["stopped_rows"]
            assert sorted(got["escalated"]) == ref["adaptive"]["escalated_rows"]
            assert (root / f"old-{lanes}/answers.jsonl").read_bytes() == (
                root / f"new-{lanes}/answers.jsonl"
            ).read_bytes()
            assert len(got["batches"]) == len(ref["generation_batches"])
            print(
                f"Frozen v11 decoder, final ordered models: 3 real test questions, cap64, budget {lanes}, 1/1/1 -> 1/8/8; candidates/logprobs/answers exact"
            )


if __name__ == "__main__":
    main()
