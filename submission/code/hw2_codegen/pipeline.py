"""Адаптивный v11: greedy → K1 всех моделей → priority waves до K2 → vote.

Порядок samples, per-model RNG и operation order совпадают с боевым v11.
SHA весов проверяется до загрузки; Kaggle API не нужен.
"""

import json
import hashlib
import gzip
import math
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .decode import BudgetExpired, Context, check_deadline, decode, question_batch
from .execute import Executor, verify_winners
from .model import PAD, from_spec
from .tokenizer_repr import (
    active_tokens,
    decode_bytes,
    encode_question,
    restore_compact,
    tokenizer,
)
from .vote import candidates, select_available, stage1_decision, stage1_leader_fraction, winner_fraction, slot_temperatures

from .config import default_config
from .decode import Options

DEFAULT_MODELS = default_config()["models"]


def sha256(path):
    """SHA256 файла без загрузки большого веса целиком в RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path, device):
    """Model-only checkpoint: model_spec + model; FP16 только на CUDA."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("format") != "nlp-hw02-model-only-v1"
        or payload.get("optimizer_included") is not False
    ):
        raise ValueError("Expected model-only checkpoint without optimizer")
    model = from_spec(payload["model_spec"]).eval()
    if device.type == "cuda":
        model.half()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def pack(ids, lps, table, greedy=False):
    """Батч token ids → восстановленный solve и средний untempered log p."""
    groups = []
    for sequences, means in zip(ids.tolist(), lps.tolist()):
        group = []
        for tokens, lp in zip(sequences, means):
            active = active_tokens(tokens)
            represented = decode_bytes(table, active).decode("utf8", errors="replace")
            try:
                code, error = restore_compact(represented), None
            except (ValueError, IndexError):
                code, error = "", "invalid representation"
            group.append(
                dict(
                    code=code,
                    mean_logprob=0.0 if greedy else (lp if math.isfinite(lp) else None),
                    represented_code=represented,
                    restore_error=error,
                    length=len(active),
                )
            )
        groups.append(group)
    return groups


def write_answers(path, codes):
    """Формат бота: одна строка {code: str} на вопрос, исходный порядок."""
    with Path(path).open("w", encoding="utf8") as stream:
        for code in codes:
            if not isinstance(code, str):
                raise TypeError("Answer code must be a string")
            stream.write(json.dumps({"code": code}, ensure_ascii=False) + "\n")


def configure_runtime():
    """Один CPU thread у Torch оставляет ядра CPU executor; одна CUDA T4."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_inference(
    config, *, started=None, rows=None, models=None, tok=None, table=None
):
    """Полный timer до записи answers.jsonl; started задаётся ПЕРЕД импортами.

    rows/models/tok/table используются только CPU dry-run и тестами.
    Deadline cooperative: завершение CUDA kernel и загрузка весов не прерываются.
    Поэтому ≤180s подтверждается полным cold T4 замером, а не самим guard.
    """
    started = time.monotonic() if started is None else started
    budget = float(config.get("budget_seconds", 180.0))
    if not math.isfinite(budget) or not 0 < budget <= 180:
        raise ValueError("Require 0 < budget_seconds <= 180")
    policy = default_config()
    guards_config = config.get("guards", {})
    options = Options(**config.get("decoder", policy["decoder"]))
    work_deadline = started + guards_config.get(
        "generation", budget - min(20.0, budget * 0.12)
    )
    execution_deadline = started + guards_config.get(
        "execution", budget - min(14.0, budget * 0.08)
    )
    verification_deadline = started + guards_config.get(
        "verification", budget - min(10.0, budget * 0.06)
    )
    greedy_deadline = started + guards_config.get(
        "greedy", budget - min(50.0, budget * 0.28)
    )
    specs = config["models"]
    if not 2 <= len(specs) <= 4 or len({m["name"] for m in specs}) != len(specs):
        raise ValueError("Require an ordered list of 2–4 distinct models")
    if any(
        not isinstance(m["K1"], int)
        or not isinstance(m["K2"], int)
        or not 1 <= m["K1"] <= m["K2"]
        for m in specs
    ):
        raise ValueError("Require integer 1 <= K1 <= K2")
    e2_primary = config.get("e2_primary", False)
    if e2_primary:
        if ([m["K1"] for m in specs] != [1, 1, 1] or
            [m["K2"] for m in specs] != [2, 8, 8] or
            [m.get("K3", m["K2"]) for m in specs] != [2, 16, 8] or
            [bool(m.get("mixed_temperature")) for m in specs] != [False, True, True]):
            raise ValueError("E2 primary requires C2/P8/W8 mixed, stage3 P16")
    threshold = config.get("unanimity_threshold", 1.0)
    if not 0 < threshold <= 1:
        raise ValueError("Require 0 < unanimity_threshold <= 1")
    device = torch.device(config.get("device", "cuda:0"))
    configure_runtime()
    if device.type == "cuda" and (
        torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Official timing requires one NVIDIA T4")
    if rows is None:
        rows = json.loads(Path(config["rows"]).read_text())
    if not rows or not all(isinstance(r["question"], str) for r in rows):
        raise ValueError("Expected a JSON list with question strings")
    model_hashes = {}
    if models is None:
        for spec in specs:
            path = Path(config["weights_root"]) / spec["filename"]
            model_hashes[spec["name"]] = sha256(path)
            if model_hashes[spec["name"]] != spec["sha256"]:
                raise ValueError("Weight SHA mismatch: " + spec["name"])
        models = {
            m["name"]: load_model(Path(config["weights_root"]) / m["filename"], device)
            for m in specs
        }
    if tok is None:
        if sha256(config["tokenizer"]) != config["tokenizer_sha256"]:
            raise ValueError("Tokenizer SHA mismatch")
        tok, table = tokenizer(config["tokenizer"])
    encoded = [encode_question(tok, row["question"]) for row in rows]
    # Same nonstochastic warmup as v7; model RNG is reset below.
    for model in models.values():
        decode(
            Context(model, torch.tensor([encoded[0]], device=device), 16),
            options=options,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    records = [dict(greedy=None, samples={m["name"]: [] for m in specs}) for _ in rows]
    lanes = config.get("sequence_budget", policy["sequence_budget"])
    greedy_batch = question_batch(config.get("greedy_sequence_budget", lanes), 1)
    extras = {m["name"]: m["K2"] - m["K1"] for m in specs}
    wave_size = (
        question_batch(lanes, max(extras.values())) if any(extras.values()) else 0
    )
    cap = config.get("cap", 512)
    rng, effective_batches = {}, {}
    guards, batches, oom_attempts = [], [], []
    executor = Executor(config.get("generation_workers", policy["generation_workers"]))

    def generate(name, order, batch, k, temperature, target, deadline):
        batch = min(batch, effective_batches.get((name, k), batch))
        offset = 0
        while offset < len(order):
            check_deadline(deadline)
            selected = order[offset : offset + batch]
            cpu_rng = torch.get_rng_state() if temperature is not None else None
            gpu_rng = (
                torch.cuda.get_rng_state(device)
                if temperature is not None and device.type == "cuda"
                else None
            )
            src = context = ids = lps = None
            oom = False
            try:
                src = pad_sequence(
                    [torch.tensor(encoded[i]) for i in selected],
                    batch_first=True,
                    padding_value=PAD,
                ).to(device)
                context = Context(models[name], src, cap)
                ids, lps, _ = decode(
                    context,
                    k,
                    temperature,
                    deadline,
                    compute_logprob=temperature is not None,
                    options=options,
                )
                groups = pack(ids, lps, table, greedy=temperature is None)
                for i, group in zip(selected, groups):
                    if target == "greedy":
                        records[i]["greedy"] = group[0]
                    else:
                        records[i]["samples"][target].extend(group)
                executor.submit([c for group in groups for c in group])
                batches.append(dict(model=name, target=target, ids=selected, k=k))
                offset += len(selected)
            except torch.cuda.OutOfMemoryError:
                oom = True
                oom_attempts.append(dict(model=name, offset=offset, batch=batch, k=k))
                if cpu_rng is not None:
                    torch.set_rng_state(cpu_rng)
                if gpu_rng is not None:
                    torch.cuda.set_rng_state(gpu_rng, device)
                batch //= 2
                if batch < 1:
                    raise
                effective_batches[name, k] = batch
            finally:
                del src, context, ids, lps
                if oom:
                    import gc

                    gc.collect()
                if device.type == "cuda" and (oom or config.get("release_cache_between_batches", False)):
                    torch.cuda.empty_cache()

    stopped, escalated, third_rows = [], [], []
    try:
        order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]), i))
        generate(
            specs[0]["name"], order, greedy_batch, 1, None, "greedy", greedy_deadline
        )
        output = Path(config.get("output", "."))
        output.mkdir(parents=True, exist_ok=True)
        write_answers(
            output / "greedy-fallback.jsonl", [r["greedy"]["code"] for r in records]
        )
        (output / "greedy-ready.json").write_text(json.dumps(dict(rows=len(rows))))
        # An unfinished greedy pass is an error: never invent a missing answer.
        for spec in specs:
            name = spec["name"]
            torch.manual_seed(config.get("seed", policy["seed"]))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(config.get("seed", policy["seed"]))
            try:
                generate(
                    name,
                    order,
                    question_batch(lanes, spec["K1"]),
                    spec["K1"],
                    config.get("temperature", policy["temperature"]),
                    name,
                    work_deadline,
                )
            except BudgetExpired:
                guards.append("stage1_" + name)
            rng[name] = (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            )
            if guards:
                break
        try:
            if not guards:
                if not executor.drain(work_deadline):
                    raise BudgetExpired("Stage-one execution incomplete")
                leaders = {}
                for i in order:
                    check_deadline(work_deadline)
                    executor.attach(candidates(records[i], specs, "K1"))
                    decision = stage1_decision(records[i], specs, threshold)
                    if decision is None:
                        raise BudgetExpired("Stage-one execution incomplete")
                    if decision:
                        stopped.append(i)
                    else:
                        escalated.append(i)
                        leaders[i] = stage1_leader_fraction(records[i], specs)
                priority = sorted(
                    escalated, key=lambda i: (leaders[i], len(encoded[i]), i)
                )
                for offset in range(0, len(priority), wave_size) if wave_size else []:
                    check_deadline(work_deadline)
                    ready = priority[offset : offset + wave_size]
                    for spec in specs:
                        name = spec["name"]
                        if not extras[name]:
                            continue
                        cpu_rng, gpu_rng = rng[name]
                        torch.set_rng_state(cpu_rng)
                        if gpu_rng is not None:
                            torch.cuda.set_rng_state(gpu_rng, device)
                        generate(
                            name,
                            ready,
                            wave_size,
                            extras[name],
                            slot_temperatures(spec, spec["K1"] + 1, spec["K2"], config.get("temperature", policy["temperature"]))
                            if e2_primary else config.get("temperature", policy["temperature"]),
                            name,
                            work_deadline,
                        )
                        rng[name] = (
                            torch.get_rng_state(),
                            (
                                torch.cuda.get_rng_state(device)
                                if device.type == "cuda"
                                else None
                            ),
                        )
                if e2_primary:
                    if not executor.drain(work_deadline):
                        raise BudgetExpired("Stage-two execution incomplete")
                    executor.attach([c for i in escalated for c in candidates(records[i], specs)])
                    # Decide only after all 19 stage-two votes have executed.
                    third_rows = [i for i in priority if winner_fraction(records[i], specs) < .5]
                    spec = specs[1]
                    name = spec["name"]
                    k = spec["K3"] - spec["K2"]
                    batch = question_batch(lanes, k)
                    cpu_rng, gpu_rng = rng[name]
                    torch.set_rng_state(cpu_rng)
                    if gpu_rng is not None:
                        torch.cuda.set_rng_state(gpu_rng, device)
                    generate(name, third_rows, batch, k,
                             slot_temperatures(spec, spec["K2"] + 1, spec["K3"], config.get("temperature", policy["temperature"])),
                             name, work_deadline)
        except BudgetExpired:
            guards.append("stage2_or_3" if e2_primary else "stage2")
        executor.resize(config.get("tail_workers", 4))
        if not executor.drain(execution_deadline):
            guards.append("execution")
        executor.attach([c for record in records for c in candidates(record, specs)])
    finally:
        executor.close()
    selected = [select_available(record, specs) for record in records]
    winners = [item[0] for item in selected]
    counts = [item[1] for item in selected]
    fallback = [i for i, count in enumerate(counts) if count == 0]
    stopped_set, third_set = set(stopped), set(third_rows)
    partial = [
        i
        for i, count in enumerate(counts)
        if 0 < count < sum(m["K1"] if i in stopped_set else m.get("K3", m["K2"]) if i in third_set else m["K2"] for m in specs)
    ]
    verification = verify_winners(
        winners, verification_deadline, config.get("tail_workers", 4)
    )
    for i in verification["mismatch_rows"]:
        winners[i] = records[i]["greedy"]
    output = Path(config.get("output", "."))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "answers.jsonl"
    write_answers(path, [w["code"] for w in winners])
    elapsed = time.monotonic() - started
    result = dict(
        path=str(path),
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                     device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        elapsed_seconds=elapsed,
        within_budget=elapsed <= budget,
        rows=len(rows),
        stopped=stopped,
        escalated=escalated,
        third_rows=third_rows,
        e2_primary=e2_primary,
        guards=guards,
        verification=verification,
        batches=batches,
        fallback_rows=fallback,
        partial_rows=partial,
        oom_attempts=oom_attempts,
        full_pipeline_seconds=elapsed,
        model_sha256=model_hashes,
        model_order=[m["name"] for m in specs],
        sequence_budget=lanes,
        answers_sha256=sha256(path),
        question_order_sha256=hashlib.sha256(
            json.dumps([r["question"] for r in rows], ensure_ascii=False).encode()
        ).hexdigest(),
        mean_sample_count=sum(len(v) for r in records for v in r["samples"].values())
        / len(rows),
        peak_reserved_bytes=(
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
        peak_allocated_bytes=(
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        total_device_memory_bytes=(
            torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda"
            else None
        ),
        no_numeric_winner_rows=[
            i for i, w in enumerate(winners) if w["execution"]["status"] != "numeric"
        ],
    )
    # Diagnostics and reference scoring happen after answers file close.
    (output / "timing.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    if config.get("reference"):
        refs = json.loads(Path(config["reference"]).read_text())
        matches = [
            i
            for i, (ref, w) in enumerate(zip(refs, winners))
            if ref["status"] == w["execution"]["status"] == "numeric"
            and math.isclose(
                ref["value"], w["execution"]["value"], rel_tol=1e-6, abs_tol=1e-6
            )
        ]
        assert len(refs) == len(winners)
        quality = dict(
            matches=len(matches),
            n=len(rows),
            accuracy_pp=100 * len(matches) / len(rows),
            match_indices=matches,
        )
        (output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")
        with gzip.open(
            output / "candidate-journal.json.gz", "wt", encoding="utf8"
        ) as stream:
            json.dump(records, stream, ensure_ascii=False)
    return result
