"""CPU parity с исполняемыми SOURCES замороженного notebook v11 (без GPU)."""

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

        rows, _, tok, table, _ = fixtures()
        torch.manual_seed(100)
        tiny = dict(
            d_model=16,
            heads=2,
            encoder_layers=1,
            decoder_layers=1,
            swiglu_dim=24,
            vocab=len(table),
            max_positions=64,
        )
        specs = [
            dict(name=n, K1=1, K2=k, filename=n + ".pt", path="unused", sha256="0" * 64)
            for n, k in [("C", 1), ("P1-B", 8), ("W", 8)]
        ]
        clean = {
            m["name"]: model.from_spec(
                dict(
                    tiny,
                    **(
                        dict(d_model=24, heads=3, swiglu_dim=32, qk_norm="layernorm")
                        if m["name"] == "W"
                        else {}
                    ),
                )
            ).eval()
            for m in specs
        }
        old = {}
        for name, m in clean.items():
            spec = dict(
                tiny,
                **(
                    dict(d_model=24, heads=3, swiglu_dim=32, qk_norm="layernorm")
                    if name == "W"
                    else {}
                ),
            )
            old[name] = original.SimpleTransformer(**spec, dropout=0.0).eval()
            old[name].load_state_dict(m.state_dict(), strict=True)
        rows_path = root / "rows.json"
        rows_path.write_text(json.dumps(rows))
        for lanes in [14, 21]:
            cfg = dict(
                models=specs,
                rows=str(rows_path),
                tokenizer="unused",
                verify_weights=False,
                device="cpu",
                cap=32,
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
                f"Frozen v11 mixed C/P1/W-QK: 8 rows, budget {lanes}, 1/1/1 -> 1/8/8; candidates/logprobs/answers exact"
            )


if __name__ == "__main__":
    main()
