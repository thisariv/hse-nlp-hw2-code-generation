"""CPU-регрессии относительно work/kaggle/src, без GPU/обученных больших моделей.

Запуск из submission/: PYTHONPATH=code python -m unittest discover -s tests -v
Реальные журналы остаются в work/results (ignored), не копируются в сдачу.
"""

import copy
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SUBMISSION = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("HW2_WORK", SUBMISSION.parent / "work"))
sys.path[:0] = [
    str(SUBMISSION / "code"),
    str(WORK / "kaggle/src"),
    str(WORK / "kaggle"),
]
import final_core as original_model
import final_fast as original_fast
import final_speed as original_speed
import final_representation as original_repr
import s3_representation as original_compact
import final_ensemble as original_vote
import l1_runtime as original_train
import p1_math as original_truncation
import exp26_order as original_data
from hw2_codegen import (
    model,
    decode,
    tokenizer_repr as representation,
    vote,
    data,
    train,
)
from hw2_codegen.execute import Executor
from hw2_codegen.pipeline import pack, run_inference
from hw2_codegen.dry_run import fixtures

torch.set_num_threads(1)
DEVICE = torch.device("cpu")
SPEC = dict(
    d_model=16,
    heads=2,
    encoder_layers=1,
    decoder_layers=2,
    swiglu_dim=24,
    vocab=31,
    max_positions=256,
)


def equal_state(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(
            torch.equal(a, b), f"Tensor mismatch, max={float((a - b).abs().max())}"
        )
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            equal_state(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            equal_state(test, x, y)
    else:
        test.assertEqual(a, b)


class TinyCache:
    def __init__(self):
        rng = np.random.default_rng(42)
        self.rows = [
            (np.array([1, *rng.integers(3, 31, 3 + i % 4), 2]), rng.integers(3, 31, n))
            for i, n in enumerate([10, 15, 52, 53, 54, 78, 79, 80, 104, 105, 106, 110])
        ]
        self.source_lengths = np.array([len(s) for s, t in self.rows], dtype=np.uint16)
        self.target_lengths = np.array([len(t) for s, t in self.rows], dtype=np.uint16)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[int(i)]


class Equivalence(unittest.TestCase):
    def test_model_initialization_forward_and_checkpoint(self):
        old = original_train.new_model(SPEC, DEVICE).eval()
        train.seed_all()
        new = model.from_spec(SPEC).eval()
        equal_state(self, old.state_dict(), new.state_dict())
        src = torch.tensor([[1, 4, 5, 2], [1, 9, 2, 0]])
        tgt = torch.tensor([[1, 6, 4], [1, 8, 0]])
        with torch.no_grad():
            self.assertTrue(torch.equal(old(src, tgt), new(src, tgt)))

    def test_decode_tokens_logprobs_and_cache_compaction(self):
        train.seed_all()
        new = model.from_spec(SPEC).eval()
        old = original_model.SimpleTransformer(**SPEC, dropout=0.0).eval()
        old.load_state_dict(new.state_dict())
        src = torch.tensor([[1, 4, 5, 2], [1, 9, 2, 0], [1, 2, 0, 0]])
        options = original_speed.Options(
            share_cross=True,
            async_eos=True,
            eos_every=4,
            compact_done=True,
            race_sampler=True,
            lean_greedy=True,
        )
        for k, temperature in [(1, None), (1, 0.7), (4, 0.7)]:
            torch.manual_seed(270042)
            ref = original_speed.decode(
                original_fast.Context(old, src, 32),
                k,
                temperature,
                options=options,
                compute_logprob=temperature is not None,
            )
            torch.manual_seed(270042)
            got = decode.decode(
                decode.Context(new, src, 32),
                k,
                temperature,
                options=decode.Options(eos_every=4),
                compute_logprob=temperature is not None,
            )
            self.assertTrue(torch.equal(ref[0], got[0]))
            self.assertTrue(torch.equal(ref[1], got[1]))
        old_decoder = original_speed.Decoder(original_fast.Context(old, src, 16), 2)
        new_decoder = decode.Decoder(decode.Context(new, src, 16), 2)
        for token in [
            torch.ones(6, dtype=torch.long),
            torch.tensor([3, 4, 5, 6, 7, 8]),
        ]:
            self.assertTrue(
                torch.equal(old_decoder.step(token), new_decoder.step(token))
            )
        index = torch.tensor([1, 4])
        old_decoder.select(index)
        new_decoder.select(index)
        self.assertEqual(new_decoder.t, 2)
        self.assertTrue(
            torch.equal(
                old_decoder.step(torch.tensor([6, 7])),
                new_decoder.step(torch.tensor([6, 7])),
            )
        )

    def test_collate_order_schedule_truncation(self):
        cache = TinyCache()
        ids = np.array([8, 1, 3, 3, 5])
        equal_state(
            self,
            original_data.collate(cache, ids, DEVICE),
            data.collate(cache, ids, DEVICE),
        )
        self.assertTrue(
            np.array_equal(
                original_data.make_order(cache, seed=123, batch=4, chunk=8),
                data.make_order(cache, seed=123, batch=4, chunk=8),
            )
        )
        for seed in range(20):
            scores = np.random.default_rng(seed).integers(0, 5, 128).astype(float)
            lengths = np.random.default_rng(seed + 1).integers(1, 130, 128)
            for q in [0.0, 0.1, 0.5, 0.99]:
                ref, _ = original_truncation.truncation_mask(scores, lengths, q)
                self.assertTrue(
                    np.array_equal(ref, train.truncation_mask(scores, lengths, q))
                )
        for update in range(1, 9):
            self.assertEqual(
                train.learning_rate(update, 0.0015, 4, 8, 10),
                original_train.lr_at(update, 0.0015, 4),
            )
        for step in range(1, 11):
            self.assertEqual(
                train.learning_rate(8 + step, 0.0015, 4, 8, 10),
                original_truncation.sqrt_lr(step, 10),
            )

    def test_adamw_updates_plain_and_truncated(self):
        cache = TinyCache()
        ids = np.arange(len(cache))
        for q in [0.0, 0.1]:
            old = original_train.new_model(SPEC, DEVICE).train()
            train.seed_all()
            new = model.from_spec(SPEC).train()
            old_opt = original_train.optimizer(old, 0.0015, True)
            new_opt = torch.optim.AdamW(new.parameters(), lr=0.0015, weight_decay=0.01)
            old_scaler = torch.amp.GradScaler("cpu", enabled=False)
            new_scaler = torch.amp.GradScaler("cpu", enabled=False)
            for step in range(1, 3):
                lr = 0.0015 * (1 - step / 3)
                if q:
                    ref, _, keep, _ = original_truncation.graph_truncated_update(
                        old, old_opt, old_scaler, cache, ids, DEVICE, 4, lr
                    )
                else:
                    ref = original_train.update(
                        old, old_opt, old_scaler, cache, ids, DEVICE, 4, lr
                    )
                got = train.update(
                    new, new_opt, new_scaler, cache, ids, DEVICE, 4, lr, q
                )
                self.assertEqual(ref["CE"], got["CE"])
                self.assertEqual(ref["tokens"], got["tokens"])
                if q:
                    self.assertEqual(keep.tolist(), got["keep"])
                equal_state(self, old.state_dict(), new.state_dict())
                equal_state(self, old_opt.state_dict(), new_opt.state_dict())

    def test_training_cache_export_and_exact_resume(self):
        cache = TinyCache()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "cache/shard-000"
            folder.mkdir(parents=True)
            for prefix, column in [("source", 0), ("target", 1)]:
                values = [row[column] for row in cache.rows]
                np.save(
                    folder / (prefix + ".npy"), np.concatenate(values).astype(np.int16)
                )
                np.save(
                    folder / (prefix + "-offsets.npy"),
                    np.array([0, *np.cumsum([len(v) for v in values])], dtype=np.int64),
                )
            (root / "cache/compact-manifest.json").write_text(
                json.dumps(
                    dict(
                        sealed=True,
                        representation=30,
                        rows=len(cache),
                        shards=[dict(folder="shard-000", rows=len(cache))],
                    )
                )
            )
            order = data.make_order(cache)
            np.save(root / "order.npy", order)
            loaded = data.CompactCache(root / "cache")
            for i in range(len(cache)):
                for a, b in zip(loaded[i], cache[i]):
                    self.assertTrue(np.array_equal(a, b))
            config = dict(
                cache=str(root / "cache"),
                order=str(root / "order.npy"),
                device="cpu",
                model_spec=SPEC,
                peak_lr=0.0015,
                warmup_updates=1,
                batch=12,
                micro_batch=4,
                main_updates=2,
                cooldown_updates=2,
                cooldown="sqrt",
                cooldown_truncation=0.1,
                checkpoint_every=1,
                output=str(root / "whole"),
            )
            train.train(config)
            train.train(dict(config, cooldown_updates=0, output=str(root / "resumed")))
            train.train(
                dict(
                    config,
                    output=str(root / "resumed"),
                    resume=str(root / "resumed/latest.pt"),
                )
            )
            whole = torch.load(root / "whole/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            for key in ["model", "optimizer", "scaler", "state", "torch_rng"]:
                equal_state(self, whole[key], resumed[key])
            from hw2_codegen.pipeline import load_model

            exported = load_model(root / "resumed/weights-fp16.pt", DEVICE)
            for k, v in resumed["model"].items():
                self.assertTrue(torch.equal(exported.state_dict()[k], v.half().float()))
            self.assertTrue((root / "resumed/stable.pt").is_file())

    def test_deadline_preserves_completed_samples(self):
        import hw2_codegen.pipeline as clean_pipeline

        rows, models, tok, table, specs = fixtures()
        calls = 0
        original_decode = clean_pipeline.decode

        def expire(*args, **kwargs):
            nonlocal calls
            calls += 1
            # Two warmups, greedy and tiny-a K1 complete; tiny-b gets no batch.
            if calls == 5:
                raise decode.BudgetExpired("CPU guard fixture")
            return original_decode(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(clean_pipeline, "decode", side_effect=expire):
                result = run_inference(
                    dict(
                        models=specs,
                        device="cpu",
                        cap=32,
                        sequence_budget=16,
                        output=directory,
                    ),
                    rows=rows,
                    models=models,
                    tok=tok,
                    table=table,
                )
            self.assertEqual(result["guards"], ["stage1_tiny-b"])
            self.assertEqual(
                [(b["target"], b["k"]) for b in result["batches"]],
                [("greedy", 1), ("tiny-a", 1)],
            )
            self.assertEqual(len(Path(result["path"]).read_text().splitlines()), 8)

    def test_executor_numeric_isolation_and_limits(self):
        programs = [
            "def solve():\n    import math\n    return math.sqrt(81)\n",
            "def solve():\n    from sympy import Rational\n    return Rational(1, 2)\n",
            "def solve():\n    return 9\n",
            "def solve():\n    import os\n    return 0\n",
            "def solve():\n    import math\n    math.pi = 0\n    return math.pi\n",
            "def solve():\n    return ().__class__\n",
            "def solve():\n    while True: pass\n",
            "def solve():\n    return 7\n",
        ]
        items = [dict(code=p) for p in programs]
        executor = Executor(2)
        try:
            executor.submit(items + items[:1])
            executor.resize(4)
            self.assertTrue(executor.drain(time.monotonic() + 20))
            executor.attach(items)
            self.assertEqual(len(executor.codes), len(programs))
            self.assertEqual(
                [c["execution"].get("value") for c in items[:3]], [9.0, 0.5, 9.0]
            )
            self.assertTrue(items[3]["execution"]["status"].startswith("blocked"))
            self.assertTrue(items[4]["execution"]["status"].startswith("blocked"))
            self.assertTrue(items[5]["execution"]["status"].startswith("blocked"))
            self.assertIn(
                items[6]["execution"]["status"], ("timeout", "timeout_or_worker_exit")
            )
            self.assertEqual(items[7]["execution"]["value"], 7.0)
        finally:
            executor.close()

    def test_vote_transitive_bridge_and_partial(self):
        def candidate(value, lp=-1.0):
            return dict(
                code=str(value),
                mean_logprob=lp,
                execution=dict(status="numeric", value=value),
            )

        g = candidate(1.0, 0.0)
        samples = [candidate(1.0000015), candidate(1.00000075), candidate(8.0)]
        specs = [dict(name="a", K1=1, K2=3), dict(name="b", K1=1, K2=1)]
        record = dict(greedy=g, samples=dict(a=samples, b=[candidate(9.0)]))
        self.assertIs(vote.select_available(record, specs)[0], g)
        self.assertIs(original_vote.select_available(record, specs)[0], g)
        record["samples"]["a"][1]["execution"] = dict(status="budget_not_evaluated")
        self.assertEqual(
            vote.select_available(record, specs),
            original_vote.select_available(record, specs),
        )

    def test_pipeline_matches_v7_on_tiny_actual_decode(self):
        # Run the real old orchestration, replacing only I/O-heavy loading and executor metrics.
        import final_inference_v7 as baseline

        rows, models, tok, table, specs = fixtures()
        specs = [dict(s, filename=s["name"] + ".pt", path="/unused") for s in specs]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path = root / "rows.json"
            rows_path.write_text(json.dumps(rows))
            base_config = dict(
                models=specs,
                tokenizer="unused",
                rows=str(rows_path),
                device="cpu",
                output=str(root / "old"),
                budget_seconds=180,
                cap=32,
                sequence_budget=16,
                greedy_sequence_budget=16,
                verify_weights=False,
            )
            tiny_spec = dict(
                d_model=16,
                heads=2,
                encoder_layers=1,
                decoder_layers=1,
                swiglu_dim=24,
                vocab=len(table),
                max_positions=64,
            )
            old_models = {}
            for name, clean_model in models.items():
                reference_model = original_model.SimpleTransformer(
                    **tiny_spec, dropout=0.0
                ).eval()
                reference_model.load_state_dict(clean_model.state_dict())
                old_models[name] = reference_model
            baseline_specs = [dict(s, sha256="0" * 64) for s in specs]
            base_config["models"] = baseline_specs
            iterator = iter(old_models.values())
            old_groups, new_groups = [], []
            import hw2_codegen.pipeline as clean_pipeline

            def observe(groups, function):
                def call(*args, **kwargs):
                    result = function(*args, **kwargs)
                    groups.append(copy.deepcopy(result))
                    return result

                return call

            with (
                patch.object(baseline, "load", side_effect=lambda p, d: next(iterator)),
                patch.object(baseline, "tokenizer", return_value=(tok, table)),
                patch.object(torch, "set_num_interop_threads"),
                patch.object(baseline, "PROCESS_START", time.monotonic()),
                patch.object(
                    baseline, "pack", side_effect=observe(old_groups, baseline.pack)
                ),
            ):
                timing = baseline.run(base_config)
            with patch.object(
                clean_pipeline,
                "pack",
                side_effect=observe(new_groups, clean_pipeline.pack),
            ):
                got = run_inference(
                    dict(base_config, models=specs, output=str(root / "new")),
                    rows=rows,
                    models=models,
                    tok=tok,
                    table=table,
                )
            self.assertEqual(old_groups, new_groups)
            self.assertEqual(timing["adaptive"]["stopped_rows"], got["stopped"])
            self.assertEqual(timing["adaptive"]["escalated_rows"], got["escalated"])
            self.assertEqual(
                (root / "old/answers.jsonl").read_bytes(),
                (root / "new/answers.jsonl").read_bytes(),
            )
            self.assertEqual(got["rows"], 8)

    def test_real_question_subset_tokenizer_decoder(self):
        inputs = WORK / "results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs"
        tok, table = representation.tokenizer(inputs / "tokenizer.json")
        ref_tok, ref_table = original_repr.tokenizer(inputs / "tokenizer.json", 30)
        self.assertEqual(table, ref_table)
        rows = json.loads((inputs / "heldout4k.json").read_text())[:8]
        encoded = [representation.encode_question(tok, r["question"]) for r in rows]
        expected = [
            [1] + original_model.encode(ref_tok, r["question"]) + [2] for r in rows
        ]
        self.assertEqual(encoded, expected)
        src = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in encoded], batch_first=True
        )
        train.seed_all()
        new = model.from_spec(dict(SPEC, vocab=8195)).eval()
        old = original_model.SimpleTransformer(
            **dict(SPEC, vocab=8195), dropout=0.0
        ).eval()
        old.load_state_dict(new.state_dict())
        for k, temp in [(1, None), (2, 0.7)]:
            torch.manual_seed(270042)
            ref = original_speed.decode(
                original_fast.Context(old, src, 32),
                k,
                temp,
                options=original_speed.Options(
                    share_cross=True,
                    async_eos=True,
                    eos_every=32,
                    compact_done=True,
                    race_sampler=True,
                    lean_greedy=True,
                ),
                compute_logprob=temp is not None,
            )
            torch.manual_seed(270042)
            got = decode.decode(
                decode.Context(new, src, 32), k, temp, compute_logprob=temp is not None
            )
            self.assertTrue(torch.equal(ref[0], got[0]))
            self.assertTrue(torch.equal(ref[1], got[1]))
            self.assertEqual(
                pack(ref[0], ref[1], ref_table), pack(got[0], got[1], table)
            )

    def test_l3_journals_restore_vote(self):
        from final_v7_analysis import load_all

        journals, _ = load_all()
        specs = [
            dict(name=n, K1=1, K2=k)
            for n, k in [("C-main", 3), ("C-long2x", 5), ("pilot-b", 4)]
        ]
        rows, restored, correct = 0, 0, 0
        for i in sorted(journals["C-main"]):
            greedy = journals["C-main"][i]["greedy"]
            samples = {
                s["name"]: journals[s["name"]][i]["samples"]["0.7"] for s in specs
            }
            ref = original_vote.replay_record(greedy, samples, specs)
            got = vote.replay_record(greedy, samples, specs)
            self.assertEqual(ref, got)
            for name in journals:
                row = journals[name][i]
                # Greedy journals store only full code; samples store the actual compact string.
                self.assertEqual(
                    original_compact.compact(row["greedy"]["code"]),
                    representation.compact(row["greedy"]["code"]),
                )
                for c in row["samples"]["0.7"]:
                    text = c["represented_code"]
                    try:
                        ref_code = original_repr.restore_compact(text)
                    except (ValueError, IndexError) as error:
                        with self.assertRaises(type(error)):
                            representation.restore_compact(text)
                    else:
                        self.assertEqual(ref_code, representation.restore_compact(text))
                    restored += 1
            ref_exec = journals["C-main"][i]["reference"]
            winner = got[0]
            correct += int(
                vote.numeric(winner)
                and ref_exec["status"] == "numeric"
                and vote.same(winner["execution"]["value"], ref_exec["value"])
            )
            rows += 1
        self.assertEqual(rows, 2048)
        self.assertEqual(correct, 1535)
        print(
            f"\nL3 replay: {rows} winners, {restored} restores, {correct} matches; all exact"
        )


if __name__ == "__main__":
    unittest.main()
