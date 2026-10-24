"""Регрессии финальной политики, SHA и frozen v11 на нескольких строках."""

import json
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hw2_codegen.config import default_config
from hw2_codegen.pipeline import load_model, sha256
from hw2_codegen import pipeline
from hw2_codegen.dry_run import fixtures
from hw2_codegen.decode import BudgetExpired
from unittest.mock import patch
import torch


class FinalV11(unittest.TestCase):
    def test_oom_retry_restores_rng_and_preserves_candidates(self):
        """An OOM may consume RNG; retry must equal a run at the smaller batch."""
        rows, models, tok, table, specs = fixtures()
        specs = [dict(m, K1=1, K2=1) for m in specs]
        packed, expected = [], []
        real_pack, real_decode = pipeline.pack, pipeline.decode

        def capture(store):
            def call(*args, **kwargs):
                result = real_pack(*args, **kwargs)
                store.extend(copy.deepcopy(result))
                return result

            return call

        def out_of_memory(context, k=1, temperature=None, *args, **kwargs):
            if temperature is not None and len(context.src) > 2:
                torch.rand(13)  # A failed sampling allocation already advanced RNG.
                raise torch.cuda.OutOfMemoryError("synthetic allocation failure")
            return real_decode(context, k, temperature, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(models=specs, device="cpu", cap=32, greedy_sequence_budget=4)
            with patch.object(pipeline, "pack", side_effect=capture(expected)):
                reference = pipeline.run_inference(
                    dict(cfg, sequence_budget=2, output=str(root / "reference")),
                    rows=rows,
                    models=models,
                    tok=tok,
                    table=table,
                )
            with patch.object(
                pipeline, "decode", side_effect=out_of_memory
            ), patch.object(pipeline, "pack", side_effect=capture(packed)):
                result = pipeline.run_inference(
                    dict(cfg, sequence_budget=4, output=str(root / "retry")),
                    rows=rows,
                    models=models,
                    tok=tok,
                    table=table,
                )
            self.assertEqual(packed, expected)
            self.assertEqual(
                (root / "reference/answers.jsonl").read_bytes(),
                (root / "retry/answers.jsonl").read_bytes(),
            )
            self.assertEqual(len(result["oom_attempts"]), len(specs))
            self.assertFalse(
                result["guards"] or result["partial_rows"] or result["fallback_rows"]
            )

    def test_missing_greedy_is_an_error(self):
        rows, models, tok, table, specs = fixtures()
        original = pipeline.decode

        def expire(context, *args, **kwargs):
            if context.capacity == 32:  # Warmup uses capacity 16.
                raise BudgetExpired("greedy deadline")
            return original(context, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, patch.object(
            pipeline, "decode", side_effect=expire
        ):
            with self.assertRaises(BudgetExpired):
                pipeline.run_inference(
                    dict(
                        models=specs,
                        device="cpu",
                        cap=32,
                        sequence_budget=14,
                        output=temp,
                    ),
                    rows=rows,
                    models=models,
                    tok=tok,
                    table=table,
                )
            self.assertFalse((Path(temp) / "answers.jsonl").exists())

    def test_frozen_v11_mixed_pipeline(self):
        script = Path(__file__).with_name("v11_reference_check.py")
        subprocess.run([sys.executable, str(script)], check=True, timeout=90)

    def test_policy_and_checkpoint_rejection(self):
        c = default_config()
        self.assertEqual([m["K2"] for m in c["models"]], [1, 8, 8])
        self.assertEqual(c["sequence_budget"], 640)
        c["models"][0]["K2"] = 100
        self.assertEqual(default_config()["models"][0]["K2"], 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.pt"
            torch.save(dict(model_spec={}, model={}), path)
            self.assertEqual(len(sha256(path)), 64)
            with self.assertRaises(ValueError):
                load_model(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
