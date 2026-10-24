"""Audit the only private T4 run; replay tune, re-execute answers, retain outputs.

No API calls, training, sampling or policy selection. Run after final_package_api
output. Full downloaded artifacts stay ignored; compact evidence goes to Git.
"""

import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SUBMISSION = WORK.parent / "submission"
sys.path.insert(0, str(SUBMISSION / "code"))
from hw2_codegen.config import default_config
from hw2_codegen.execute import Executor
from hw2_codegen.vote import replay_record


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def codes(notebook):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def answers(path, count):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == count
    assert all(set(r) == {"code"} and isinstance(r["code"], str) for r in rows)
    return rows


def execute(rows):
    executor = Executor(4)
    candidates = [dict(code=r["code"]) for r in rows]
    try:
        executor.submit(candidates)
        assert executor.drain(
            time.monotonic() + 180
        ), "Independent execution incomplete"
        executor.attach(candidates)
        return [r["execution"] for r in candidates]
    finally:
        executor.close()


def same(a, b):
    return a["status"] == b["status"] and (
        a["status"] != "numeric"
        or math.isclose(a["value"], b["value"], rel_tol=1e-6, abs_tol=1e-6)
    )


def check():
    download = read(WORK / "checks/final-package-v11-download.json")
    launch = read(WORK / "checks/final-package-v11-launch.json")
    remote = read(WORK / "checks/final-package-v11-remote.json")
    assert download["ref"] == launch["ref"] == "thisarivson/nlp-hw02-final-package-v11"
    assert download["version"] == launch["version"] == remote["current_version"] == 1
    assert (
        launch["push_calls"] == 1
        and remote["private"]
        and remote["all_code_cells_exact"]
    )
    raw = WORK / download["output"]
    roots = list(raw.rglob("final-package-v11/summary.json"))
    assert len(roots) == 1
    root = roots[0].parent
    summary = read(roots[0])
    policy = default_config()
    expected_hashes = {m["name"]: m["sha256"] for m in policy["models"]}
    for split, n, rowsfile in [
        ("test-1", 1919, "test.json"),
        ("tune", 2048, "tune2048.json"),
    ]:
        report = read(root / split / "summary.json")
        assert report == summary["test" if split == "test-1" else "tune"]
        assert report["status"] == "complete" and report["rows"] == n
        assert report["within_budget"] and not report["guards"]
        assert not report["partial_rows"] and not report["fallback_rows"]
        assert not report["oom_attempts"] and not report["no_numeric_winner_rows"]
        assert (
            report["verification"]["complete"]
            and report["verification"]["checked_rows"] == n
        )
        assert not report["verification"]["mismatch_rows"]
        assert report["model_sha256"] == expected_hashes and report[
            "model_order"
        ] == list(expected_hashes)
        assert report["sequence_budget"] == policy["sequence_budget"] == 768
        rows = read(ROOT / "staging/final-package-v11" / rowsfile)
        order_sha = hashlib.sha256(
            json.dumps([r["question"] for r in rows], ensure_ascii=False).encode()
        ).hexdigest()
        assert report["question_order_sha256"] == order_sha
        assert sha(root / split / "answers.jsonl") == report["answers_sha256"]
        answers(root / split / "answers.jsonl", n)
        cfg = read(root / split / "config.json")
        assert all(cfg[k] == policy[k] for k in policy)
        assert ("reference" in cfg) == (split == "tune")
        assert all(len(b["ids"]) * b["k"] <= 768 for b in report["batches"])
    assert (root / "answers.jsonl").read_bytes() == (
        root / "test-1/answers.jsonl"
    ).read_bytes()
    assert summary["training"] is False
    assert summary["status"] in ("passed", "review_required")
    assert all(value for key, value in summary["gates"].items() if key != "memory")
    assert summary["test"]["elapsed_seconds"] <= 150
    memory_target_passed = (
        max(summary[s]["peak_device_used_bytes"] for s in ("test", "tune")) <= 14e9
    )
    refs = read(ROOT / "staging/final-package-v11/tune2048-reference.json")
    quality = read(root / "tune/quality.json")
    with gzip.open(root / "tune/candidate-journal.json.gz", "rt") as stream:
        journal = json.load(stream)
    tune = answers(root / "tune/answers.jsonl", 2048)
    assert len(journal) == len(refs) == 2048
    hits = []
    for i, (record, reference) in enumerate(zip(journal, refs)):
        winner, stopped, _ = replay_record(
            record["greedy"], record["samples"], policy["models"]
        )
        assert winner["code"] == tune[i]["code"]
        assert stopped == (i in summary["tune"]["stopped"])
        if reference["status"] == "numeric" and same(reference, winner["execution"]):
            hits.append(i)
    assert hits == quality["match_indices"] and len(hits) == quality["matches"]
    assert abs(len(hits) - 1547) <= 0.01 * 2048
    current = answers(root / "answers.jsonl", 1919)
    previous_path = WORK / "submissions/final-ensemble-v11.jsonl"
    previous = answers(previous_path, 1919)
    executions = execute(current + previous)
    assert all(x["status"] == "numeric" for x in executions)
    comparison = dict(
        rows=1919,
        previous_sha256=sha(previous_path),
        answers_sha256=sha(root / "answers.jsonl"),
        changed_code_rows=[
            i for i, (a, b) in enumerate(zip(current, previous)) if a != b
        ],
        changed_numeric_rows=[
            i
            for i, (a, b) in enumerate(zip(executions[:1919], executions[1919:]))
            if not same(a, b)
        ],
        independent_execution=True,
    )
    destination = WORK / "submissions/final-package-v11.jsonl"
    if destination.exists():
        assert destination.read_bytes() == (root / "answers.jsonl").read_bytes()
    else:
        destination.write_bytes((root / "answers.jsonl").read_bytes())
    notebooks = []
    for path in raw.rglob("*.ipynb"):
        nb = read(path)
        if codes(nb) == codes(read(SUBMISSION / "inference.ipynb")):
            if all(
                c.get("execution_count") is not None
                for c in nb["cells"]
                if c["cell_type"] == "code"
            ):
                assert not any(
                    o["output_type"] == "error"
                    for c in nb["cells"]
                    if c["cell_type"] == "code"
                    for o in c["outputs"]
                )
                notebooks.append((path, nb))
    assert notebooks, "Download the executed server notebook; do not synthesize outputs"
    nbpath, executed = notebooks[0]
    local = read(SUBMISSION / "inference.ipynb")
    for dest, source in zip(local["cells"], executed["cells"]):
        assert dest["cell_type"] == source["cell_type"]
        if dest["cell_type"] == "code":
            dest.update(
                outputs=source["outputs"], execution_count=source["execution_count"]
            )
    (SUBMISSION / "inference.ipynb").write_text(
        json.dumps(local, ensure_ascii=False, indent=1) + "\n"
    )
    result = dict(
        audit_passed=True,
        targets_passed=all(summary["gates"].values()) and memory_target_passed,
        memory_target_passed=memory_target_passed,
        ref=launch["ref"],
        version=1,
        private=True,
        own_push_calls=1,
        test_seconds=summary["test"]["elapsed_seconds"],
        tune_seconds=summary["tune"]["elapsed_seconds"],
        tune_matches=len(hits),
        tune_n=2048,
        tune_accuracy_pp=100 * len(hits) / 2048,
        peak_memory_bytes={
            s: {
                k: summary[s][k]
                for k in [
                    "peak_device_used_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                    "total_device_memory_bytes",
                ]
            }
            for s in ["test", "tune"]
        },
        changed_code=len(comparison["changed_code_rows"]),
        changed_numeric=len(comparison["changed_numeric_rows"]),
        answers_sha256=sha(destination),
        question_order_verified=True,
        model_sha256=expected_hashes,
        tune_journal_replay_exact=True,
        executed_notebook=str(nbpath.relative_to(WORK)),
        submission=str(destination.relative_to(WORK)),
        gates=summary["gates"],
        dataset_manifest_sha256=sha(
            ROOT / "staging/final-package-v11/package-manifest.json"
        ),
    )
    for name, value in [
        ("checks/final-package-v11-output.json", result),
        ("checks/final-package-v11-comparison.json", comparison),
        ("results/final-package-v11.json", result),
    ]:
        (WORK / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    check()
