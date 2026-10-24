"""Separate L3-weight v4 bundle. V3 source files and bundle remain unchanged."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT/'bundles/final_inference_l3'


def replace(source, old, new, count=1):
    assert source.count(old) == count, f'V3 template drift: {old}'
    return source.replace(old, new)


def build():
    original = json.loads((ROOT/'notebooks/nlp-hw02-final-inference.ipynb').read_text())
    first = next(c['source'] for c in original['cells'] if c['cell_type'] == 'code')
    node = next(n for n in ast.parse(first).body if isinstance(n, ast.Assign) and
                isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES')
    sources = ast.literal_eval(node.value)
    assert all((ROOT/'src'/n).read_text() == s for n, s in sources.items()), 'Rebuild/review V3 source drift first'
    final = json.loads((ROOT.parent/'results/l3-final-results.json').read_text())
    baseline = next(r['matches'] for r in final['voting_tune2048']['rows'] if r['name'] == 'mixed K4+K4 +greedy')
    identity = {'base_source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
                'point_tag': 'l2-62280', 'stored_tune_matches': baseline, 'variant': 'v4 L3 source/metadata only'}
    # Only pinned weight identity/quality metadata changes in the inference worker.
    worker = sources['final_inference.py']
    for name, old in [('C-main', '3806a5827004fd88f533a7307abc0c1e387faafc5672f4224392b363c4d2ad3d'),
                      ('C-long2x', '6227ee2fc8dd12e186137be229fef2b1ceba659848c35bedd90cec6502f41548')]:
        worker = replace(worker, old, final['receipts'][name]['FP16']['sha256'])
    worker = replace(worker, '1492', str(baseline), 3)
    sources['final_inference.py'] = worker
    # Session recipe requested by the user: one cold test plus tune, <=12 minutes.
    session = sources['final_session.py']
    session = replace(session, 'l2-28800', 'l2-62280', 2)
    session = replace(session, '1492', str(baseline))
    session = replace(session, '840.', '660.', 2)
    session = replace(session, '870.', '690.')
    session = replace(session, "'budget_seconds': 900", "'budget_seconds': 720")
    session = replace(session, 'watchdog.start < 660.', 'watchdog.start < 480.')
    session = replace(session, 'range(1, 3 if benchmark else 2)', 'range(1, 2)')
    session = replace(session, 'for i in range(1, 3)', 'for i in range(1, 2)')
    session = replace(session, "report['deterministic_answers'] = len(set(digests)) == 1",
                      "report['deterministic_answers'] = None  # One cold test; repeatability not measured")
    session = replace(session, "                           'deterministic_answers': report['deterministic_answers'],\n", '')
    session = replace(session, 'Check attached L2 first4 baseline', 'Check attached L3 first4 baseline')
    session = replace(session, 'Expected all L2 K8 journals', 'Expected all L3 K8 journals')
    sources['final_session.py'] = session
    cells = []
    def cell(kind, source):
        row = {'cell_type': kind, 'id': f'final-l3-{len(cells):02d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            compile(source, row['id'], 'exec'); row.update(outputs=[], execution_count=None)
        cells.append(row)
    cell('markdown', f'''# ДЗ2 NLP: финальный инференс v4 на L3

Private, Internet off, одна T4 (GPU0). Входы: `friend1/nlp-hw02-l3-live` (обе FP16 `l2-62280`) и `thisarivson/nlp-hw02-26b-inputs`.
Тот же v3 decoder/executor/selection: K4+K4+greedy C-main, T0.7, seed270042, 1024 lanes, cap512, FP16 и повторное исполнение выбранного кода.
Ровно один холодный test1919 и tune2048. Ответ — первый test, JSONL только `{{"code": "..."}}`, исходный порядок test.json. Цель ≤120 с, обязательный потолок <180 с; fallback=0. Tune ориентир {baseline}/2048, допуск ±1 п.п. относительно сохранённых L3 K8 first4; новый batch/bucketing меняет RNG lanes, точное совпадение не обещается. Повторяемость одного test не измеряется.
Сессия ≤720 с: watchdog stop660/exit690 от первой code cell до Torch. Дополнительных запусков, обучения и публикаций нет. Требуется Torch2.11.0+cu128/CUDA12.8. GPU gate будет выполнен отдельно наблюдателем.''')
    init = first[:first.index('SOURCES=')] + 'SOURCES='+repr(sources)+first[first.index('\nfor name,source in SOURCES.items()'):]
    init = init.replace("Path('/kaggle/working/final-inference')", "Path('/kaggle/working/final-inference-l3')").replace("Path('final-inference')", "Path('final-inference-l3')")
    cell('code', init)
    run = next(c['source'] for c in original['cells'] if c['cell_type'] == 'code' and 'def find_input' in c['source'])
    run = run.replace('nlp-hw02-l2-live', 'nlp-hw02-l3-live').replace(
        'BENCHMARK=True  # Ревью: 2 test + tune. Для сдачи после GPU gate: False (один test).',
        'BENCHMARK=True  # Ровно один холодный test1919 + tune2048.')
    cell('code', run)
    cell('markdown', 'Проверить `final-inference-l3/summary.json`, `test-1/timing.json`, `tune/quality.json`. Guard/fallback сохраняется как в v3; неполный mixed требует ревью. CPU-ready не означает пройденный GPU gate.')
    notebook = dict(original, cells=cells)
    name = 'nlp-hw02-final-inference-l3.ipynb'
    target = ROOT/'notebooks'/name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    BUNDLE.mkdir(parents=True, exist_ok=True)
    (BUNDLE/name).write_bytes(target.read_bytes())
    def write(name, value): (BUNDLE/name).write_text(json.dumps(value, indent=2)+'\n')
    write('kernel-metadata.json', {'id': 'friend1/nlp-hw02-final-inference-l3', 'title': 'NLP HW02 final inference L3',
          'code_file': name, 'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True,
          'enable_tpu': False, 'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
          'dataset_sources': ['friend1/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-26b-inputs'],
          'kernel_sources': [], 'competition_sources': []})
    write('session-limits.json', {'kind': 'final_inference_l3', 'session_timeout_seconds': 720,
          'watchdog_stop_seconds': 660, 'watchdog_exit_seconds': 690, 'cold_test_runs': 1, 'tune_runs': 1})
    identity['embedded_source_sha256'] = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    (ROOT/'final-l3-source-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(target)
    return target, sources


if __name__ == '__main__': build()
