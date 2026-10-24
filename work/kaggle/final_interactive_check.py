"""Validate v10/v11 interactive artifacts, including a partial v10 packet.

Missing tune files are reported explicitly, never reconstructed as GPU evidence.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from final_v10_output_check import execute, same
from final_vote import validate_answers
from final_ensemble import select_available
from final_split import fixed_split


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def codes(nb):
    return [''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in nb['cells'] if c['cell_type'] == 'code']


def find(root, *names):
    return next((root/n for n in names if (root/n).exists()), None)


def check(root, version):
    import nbformat
    root = Path(root)
    config = read(ROOT/f'final-v{version}-config.json')
    notebook_path = find(root, f'nlp-hw02-ensemble-inference-v{version}.ipynb')
    notebook_exact = False
    printed = None
    if notebook_path:
        nb = read(notebook_path)
        nbformat.validate(nbformat.from_dict(nb))
        local = read(ROOT/f'notebooks/nlp-hw02-ensemble-inference-v{version}.ipynb')
        bundle = read(ROOT/f'bundles/final_inference_v{version}_mine/nlp-hw02-ensemble-inference-v{version}.ipynb')
        assert codes(nb) == codes(local) == codes(bundle), 'Executed notebook code differs'
        assert [c['execution_count'] for c in nb['cells'] if c['cell_type'] == 'code'] == [1, 2]
        assert not any(o['output_type'] == 'error' for c in nb['cells'] for o in c.get('outputs', []))
        sources = ast.literal_eval(next(n.value for n in ast.parse(codes(nb)[0]).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
        assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()} == read(ROOT/f'final-v{version}-source-identity.json')['embedded_source_sha256']
        stream = ''.join(''.join(o.get('text', [])) for c in nb['cells'] for o in c.get('outputs', []) if o['output_type'] == 'stream')
        printed = json.loads(stream[stream.index('{\n'):])
        notebook_exact = True
    summary_path = find(root, 'summary.json')
    summary = read(summary_path) if summary_path else printed
    assert summary and summary['status'] in ('passed', 'review_required')
    if not notebook_exact:
        remote = read(WORK/f'checks/final-inference-v{version}-remote.json')
        assert remote['all_code_cells_exact'] and remote['private']
        if root.name != 'interactive':
            # Full batch packet: explicit version and exact runtime source hashes.
            download = read(WORK/f'checks/final-inference-v{version}-output-download.json')
            assert download['files'] > 0 and remote['version'] == download['version']
            assert root.resolve().is_relative_to((WORK/download['output']).resolve())
            identity = read(ROOT/f'final-v{version}-source-identity.json')
            assert {n: sha(root/'src'/n) for n in identity['embedded_source_sha256']} == identity['embedded_source_sha256']
    test = read(find(root, 'test-1-summary.json', 'test-1/summary.json'))
    timing = read(find(root, 'test-1-timing.json', 'test-1/timing.json'))
    assert all(test[k] == value for k, value in timing.items())
    if 'test' in summary:
        assert len(summary['test']) == 1 and summary['test'][0] == test
        assert summary['GPU_runs'] == 2 and not summary['training'] and summary['budget_seconds'] == 720
        assert summary['single_gpu'] and not summary['internet']
    else:
        assert summary['test_seconds'] == [test['full_pipeline_seconds']]
    hashes = {m['name']: m['sha256'] for m in config['models']}
    assert test['status'] == 'complete' and test['rows'] == 1919 and test['under_180_seconds']
    assert test['single_gpu'] and test['model_order'] == list(hashes) and test['model_sha256'] == hashes
    assert test['seed'] == 270042 and test['deterministic_algorithms']
    guard = test['deadline_guard']
    assert not guard['triggered'] and guard['generation_stop_seconds'] == 160
    assert guard['fallback_count'] == guard['partial_vote_count'] == 0
    assert test['winner_reexecution']['complete'] and test['winner_reexecution']['checked_rows'] == 1919
    assert not test['winner_reexecution']['mismatch_rows']
    assert test['adaptive']['K1'] == {m['name']: m['K1'] for m in config['models']}
    assert test['adaptive']['K2'] == {m['name']: m['K2'] for m in config['models']}
    assert not test['adaptive']['scorer_enabled'] and not test['adaptive']['undecided_rows']
    stopped = set(test['adaptive']['stopped_rows'])
    assert len(guard['executed_sample_counts']) == 1919
    assert all(c == sum(m['K1' if i in stopped else 'K2'] for m in config['models']) for i, c in enumerate(guard['executed_sample_counts']))
    assert test['execution']['completed_unique'] == test['execution']['unique_programs'] and test['execution']['error'] is None
    assert test['execution']['service_state']['nice'] >= 10
    assert all(b['sequences'] <= config['sequence_budget'] for b in test['generation_batches'])
    assert test['peak_reserved_bytes'] < test['total_device_memory_bytes']
    assert test['peak_allocated_bytes'] + 256*1024**2 < test['total_device_memory_bytes']
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    assert test['input_sha256'] == sha(inputs/'test.json')
    assert test['question_order_sha256'] == hashlib.sha256(json.dumps([r['question'] for r in read(inputs/'test.json')], ensure_ascii=False).encode()).hexdigest()
    answer_path = find(root, 'test-1-answers.jsonl', 'test-1/answers.jsonl')
    answers = validate_answers(answer_path, 1919)
    assert test['answers_sha256'] == sha(answer_path)
    quality = summary['quality']
    assert quality['n'] == 2048 and quality['expected_matches'] == config['expected_matches']
    assert quality['matches'] == len(set(quality['match_indices'])) == len(quality['match_indices'])
    assert all(0 <= i < 2048 for i in quality['match_indices'])
    assert quality['accuracy_pp'] == 100*quality['matches']/2048
    assert quality['delta_pp'] == 100*(quality['matches']-quality['expected_matches'])/2048
    assert all(value for key, value in summary['gates'].items() if key not in ('quality_within_1pp', 'test_with_40s_margin'))
    assert summary['gates']['quality_within_1pp'] == quality['passed']
    assert (summary['status'] == 'passed') == all(summary['gates'].values())
    runtime_tune_replay = False
    tune_report = summary.get('tune')
    if tune_report:
        assert tune_report['rows'] == 2048 and tune_report['status'] == 'complete'
        assert tune_report['model_sha256'] == hashes and tune_report['model_order'] == list(hashes)
        assert tune_report['adaptive']['K2'] == {m['name']: m['K2'] for m in config['models']}
        assert tune_report['adaptive']['K1'] == {m['name']: m['K1'] for m in config['models']}
        assert not tune_report['adaptive']['scorer_enabled'] and not tune_report['adaptive']['undecided_rows']
        tune_stopped = set(tune_report['adaptive']['stopped_rows'])
        counts = tune_report['deadline_guard']['executed_sample_counts']
        assert len(counts) == 2048 and all(c == sum(m['K1' if i in tune_stopped else 'K2'] for m in config['models']) for i, c in enumerate(counts))
        assert tune_report['single_gpu'] and not tune_report['deadline_guard']['triggered']
        assert tune_report['seed'] == 270042 and tune_report['deterministic_algorithms']
        assert tune_report['execution']['completed_unique'] == tune_report['execution']['unique_programs'] and tune_report['execution']['error'] is None
        assert tune_report['peak_reserved_bytes'] < tune_report['total_device_memory_bytes']
        assert tune_report['peak_allocated_bytes'] + 256*1024**2 < tune_report['total_device_memory_bytes']
        assert tune_report['deadline_guard']['fallback_count'] == tune_report['deadline_guard']['partial_vote_count'] == 0
        assert tune_report['winner_reexecution']['checked_rows'] == 2048
        assert tune_report['winner_reexecution']['complete'] and not tune_report['winner_reexecution']['mismatch_rows']
        ids = fixed_split()['tune2048']
        heldout = read(inputs/'heldout4k.json')
        tune_rows = [dict(question=heldout[i]['question']) for i in ids]
        assert tune_report['input_sha256'] == hashlib.sha256((json.dumps(tune_rows, ensure_ascii=False, indent=2)+'\n').encode()).hexdigest()
        assert tune_report['question_order_sha256'] == hashlib.sha256(json.dumps([r['question'] for r in tune_rows], ensure_ascii=False).encode()).hexdigest()
        if (root/'tune/candidate-journal.json.gz').exists() and (root/'tune/answers.jsonl').exists():
            with gzip.open(root/'tune/candidate-journal.json.gz', 'rt') as f:
                journal = json.load(f)
            tune_answers = validate_answers(root/'tune/answers.jsonl', 2048)
            assert len(journal) == 2048 and sha(root/'tune/answers.jsonl') == tune_report['answers_sha256']
            references = read(inputs/'heldout4k-reference-execution.json')
            references = references['rows'] if isinstance(references, dict) else references
            matches = []
            for i, row in enumerate(journal):
                winner, _ = select_available(row, config['models'])
                assert winner['code'] == tune_answers[i]['code']
                ref = references[ids[i]]
                if ref['status'] == 'numeric' and same(ref, winner['execution']):
                    matches.append(i)
            assert matches == quality['match_indices']
            runtime_tune_replay = True
    previous_version = 8 if version == 10 else 10
    previous_path = WORK/f'submissions/final-ensemble-v{previous_version}.jsonl'
    previous = validate_answers(previous_path, 1919)
    values, execution = execute(answers + previous)
    current, old = values[:1919], values[1919:]
    nonnumeric = [i for i, (a, b) in enumerate(zip(current, old)) if a['status'] != 'numeric' or b['status'] != 'numeric']
    assert not nonnumeric, nonnumeric
    changed_code = [i for i, (a, b) in enumerate(zip(answers, previous)) if a['code'] != b['code']]
    changed_answer = [i for i, (a, b) in enumerate(zip(current, old)) if not same(a, b)]
    comparison = dict(rows=1919, current_sha256=sha(answer_path), previous_sha256=sha(previous_path), previous_version=previous_version,
        changed_code=len(changed_code), changed_code_rows=changed_code, changed_executed_answer=len(changed_answer), changed_executed_answer_rows=changed_answer,
        nonnumeric_rows=nonnumeric, execution=execution, method='fresh bounded production executor; numeric isclose rel/abs 1e-6')
    destination = WORK/f'submissions/final-ensemble-v{version}.jsonl'
    if destination.exists():
        assert destination.read_bytes() == answer_path.read_bytes(), 'Different existing submission preserved'
    else:
        with destination.open('xb') as f:
            f.write(answer_path.read_bytes())
    result = dict(passed=True, status='validated', summary_status=summary['status'], gates=summary['gates'],
        origin='external interactive packet' if root.name == 'interactive' else 'version-bound API packet',
        notebook_code_cells_exact=notebook_exact, all_20_embedded_sources_exact=notebook_exact or root.name != 'interactive',
        source_validation_limitation=None if notebook_exact or root.name != 'interactive' else 'Executed notebook not supplied; remote code is exact but interactive session source parity is unproven.',
        test1919_seconds=test['full_pipeline_seconds'], test_target_140_passed=test['full_pipeline_seconds'] <= 140,
        tune_matches=quality['matches'], tune_n=2048, tune_accuracy_pp=quality['accuracy_pp'], quality=quality,
        tune_seconds=summary.get('tune', {}).get('full_pipeline_seconds'),
        tune_raw_files_available=bool(find(root, 'tune/answers.jsonl') and find(root, 'tune/candidate-journal.json.gz')),
        runtime_tune_replay_verified=runtime_tune_replay,
        tune_validation_limitation=None if runtime_tune_replay else 'Quality from executed report; independent raw tune replay unavailable without tune report/answers/journal.',
        comparison_previous_version=previous_version, changed_code=len(changed_code), changed_executed_answer=len(changed_answer),
        submission=str(destination.relative_to(WORK)), answers_sha256=sha(destination), rows=1919, code_only=True, test_order_verified=True,
        test_early_stops=len(stopped), test_mean_samples=test['adaptive']['mean_sample_count'],
        test_OOM_attempts=test['OOM_attempts'], test_peak_allocated_bytes=test['peak_allocated_bytes'], test_peak_reserved_bytes=test['peak_reserved_bytes'],
        verified_model_sha256=hashes, session_seconds=summary['session_seconds'],
        interactive_session_metadata_available=False if notebook_exact else None,
        split_metrics={name: dict(seconds=r['full_pipeline_seconds'], early_stops=len(r['adaptive']['stopped_rows']),
            mean_stochastic_samples=r['adaptive']['mean_sample_count'], OOM_attempts=r['OOM_attempts'],
            peak_allocated_bytes=r['peak_allocated_bytes'], peak_reserved_bytes=r['peak_reserved_bytes'],
            winner_reexecution_checked_rows=r['winner_reexecution']['checked_rows'], winner_reexecution_mismatch_rows=r['winner_reexecution']['mismatch_rows'])
            for name, r in [('test', test), ('tune', tune_report)] if r},
        artifacts={str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.ipynb')})
    if version == 10:
        result.update(public_accuracy_pp=78.96, public_source='user report', v8_public_accuracy_pp=78.54, v9_public_accuracy_pp=78.44)
    for path, data in [(WORK/f'checks/final-inference-v{version}-comparison.json', comparison),
                       (WORK/f'checks/final-inference-v{version}-output.json', result),
                       (WORK/f'results/final-inference-v{version}.json', result)]:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    return {k: v for k, v in result.items() if k not in ('quality', 'artifacts')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('version', type=int, choices=[10, 11])
    p.add_argument('output', type=Path)
    args = p.parse_args()
    print(json.dumps(check(args.output, args.version), ensure_ascii=False, indent=2))
