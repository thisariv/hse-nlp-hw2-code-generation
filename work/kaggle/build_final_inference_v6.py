"""Build private thisarivson v6 without API calls; weight dataset/names are configurable."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BUNDLE = ROOT/'bundles/final_inference_l3_v6_mine'
READY = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v6.json'
NAME = 'nlp-hw02-adaptive-inference-l3-v6.ipynb'
DEFAULT_DATASET = 'thisarivson/nlp-hw02-l3-live'
DEFAULT_MAIN = 'model-C-main__weights__l2-62280-fp16.pt'
DEFAULT_LONG = 'model-C-long2x__weights__l2-62280-fp16.pt'


def replace(source, old, new):
    assert source.count(old) == 1, f'V5 template drift: {old}'
    return source.replace(old, new)


def build(weights_dataset=DEFAULT_DATASET, main_weight=DEFAULT_MAIN, long_weight=DEFAULT_LONG,
          main_sha256=None, long_sha256=None):
    assert re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', weights_dataset)
    assert all(Path(n).name == n and n.endswith('.pt') for n in [main_weight, long_weight])
    assert main_weight != long_weight, 'Use distinct names for main/long slots'
    assert all(s is None or re.fullmatch('[a-f0-9]{64}', s) for s in [main_sha256, long_sha256])
    assert (main_sha256 is None) == (long_sha256 is None), 'Pin both SHA or neither'
    BUNDLE.mkdir(parents=True, exist_ok=True)
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False}, indent=2)+'\n')
    original = json.loads((ROOT/'notebooks/nlp-hw02-final-inference-l3-v5.ipynb').read_text())
    code = [c['source'] for c in original['cells'] if c['cell_type'] == 'code']
    first = code[0]
    sources = ast.literal_eval(next(n.value for n in ast.parse(first).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    base_identity = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    assert base_identity == json.loads((ROOT/'final-v5-source-identity.json').read_text())['embedded_source_sha256']
    analysis = json.loads((WORK/'checks/final-inference-v6-offline.json').read_text())
    assert analysis['passed']
    k = analysis['selected_K2']
    selected = next(r for r in analysis['rows'] if r['K2'] == k)
    config = {'K1': 2, 'K2': k, 'unanimity_threshold': 1., 'sequence_budget': 1024,
              'greedy_sequence_budget': 1024, 'scorer_enabled': False, 'scorer_weight': 0.,
              'expected_matches': selected['matches']}
    default_pair = (weights_dataset, main_weight, long_weight) == (DEFAULT_DATASET, DEFAULT_MAIN, DEFAULT_LONG)
    pinned = ({'C-main': main_sha256, 'C-long2x': long_sha256} if main_sha256 else
        {'C-main': '8c11b1038855422b6912e243049a554317978f848edb412ed48aebbe4cb6b886',
         'C-long2x': 'd10bf0018d0b14fd6cc3c8b9c7121f127b5cec97057544fd63e5824253a55057'} if default_pair else {})
    weights = {'dataset': weights_dataset, 'filenames': {'C-main': main_weight, 'C-long2x': long_weight},
               'sha256': pinned, 'L3_journal_diagnostic': default_pair and main_sha256 is None}
    for name in ['final_adaptive.py', 'final_speed.py', 'final_execute_v6.py', 'final_executor_v6.py']:
        sources[name] = (ROOT/'src'/name).read_text()
    sources['final_inference.py'] = (ROOT/'src/final_inference_v6.py').read_text()
    session = sources['final_session.py']
    session = replace(session, 'Check attached L3 first4 baseline', 'Check attached L3 adaptive baseline')
    session = replace(session, 'k1=2, k2=4)', f'k1=2, k2={k})')
    session = replace(session, 'assert matches == 1511', f"assert matches == {selected['matches']}")
    session = replace(session, "        models = {name: unique(live_root, f'model-{name}__weights__l2-62280-fp16.pt')\n                  for name in ['C-main', 'C-long2x']}",
                      "        models = {name: unique(live_root, filename) for name, filename in "+repr(weights['filenames'])+".items()}")
    session = replace(session, "            journals, report['stored_tune'] = stored_tune(live_root, heldout)\n            for i in ids:\n                assert journals['C-main'][i]['reference'] == refs[i]",
        "            journals = None\n            report['stored_tune'] = "+repr({'matches': selected['matches'], 'n': 2048, 'origin': 'local verified L3 replay; comparison baseline'})+"\n"+
        ("            journals, report['stored_tune'] = stored_tune(live_root, heldout)\n            for i in ids:\n                assert journals['C-main'][i]['reference'] == refs[i]" if weights['L3_journal_diagnostic'] else "            # A replacement pair need not include L3 journals."))
    oldconfig = json.loads((ROOT/'final-v5-source-identity.json').read_text())['adaptive_config']
    session = replace(session, repr(oldconfig), repr(config))
    session = replace(session, "'workers': os.cpu_count() or 1,", "'workers': 2, 'verify_weights': "+repr(bool(pinned))+", 'weight_sha256': "+repr(pinned)+",")
    session = replace(session, "if (output/'tune/candidate-journal.json.gz').exists():", "if journals is not None and (output/'tune/candidate-journal.json.gz').exists():")
    session = replace(session, "'test_with_25s_margin': all(r['target_155_seconds'] for r in report['test']),", "'test_with_30s_margin': all(r['target_150_seconds'] for r in report['test']),")
    sources['final_session.py'] = session
    cells = []
    def cell(kind, source):
        item = {'cell_type': kind, 'id': f'final-v6-{len(cells):02d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            compile(source, item['id'], 'exec'); item.update(outputs=[], execution_count=None)
        cells.append(item)
    cell('markdown', f'''# ДЗ2 NLP: adaptive inference L3 v6

Private thisarivson, Internet off, одна T4/GPU0. Greedy C-main → K2+K2 → до K{k}+K{k} для неединогласных. Глобальный приоритет: минимальная доля лидирующего ответа, затем длина/ID; main и long подряд. Frozen vote count/greedy/mean LP/order, голосование по готовым кандидатам. Decoder all: compaction, async EOS32, greedy cross-KV sharing, Exp race sampler, lean greedy, preallocated self-KV. Во время генерации2 nice10 workers; после неё до4, та же очередь/кэш. CUDA dispatcher сохраняет приоритет.

Один холодный test1919 + tune2048, ≤720с; watchdog660/690. Guard160 / execution166 / verification170 / supervisor172; полный timer включает imports/load/warmup/execution/vote/reexecution/write, цель≤150с, потолок<180с. K{k} оценён≤{selected['timing']['conservative_seconds']:.1f}с с запасом; K8≈{analysis['rows'][-1]['timing']['conservative_seconds']:.1f}с не прошёл порог150. Это прогноз, полный GPU v6 ещё не измерен. L3 frozen baseline {selected['matches']}/2048, для новых stochastic samples допуск±1п.п. Известное допустимое FP16 greedy отличие all:1/1919 без нагрузки,3/1919 с нагрузкой; формы батча меняют округление. RNG сэмплера изменён, categorical semantics сохранена.

Веса: {weights_dataset}, `{main_weight}` и `{long_weight}`. Custom pair не требует L3 journals; baseline остаётся L3 для сравнения. Model-only format/spec/strict load проверяются, фактические SHA записываются. Без обучения/API/Secrets/push.''')
    init = first[:first.index('SOURCES=')]+'SOURCES='+repr(sources)+first[first.index('\nfor name,source in SOURCES.items()'):]
    init = init.replace('final-inference-l3-v5', 'adaptive-inference-l3-v6')
    cell('code', init)
    run = replace(code[1], "LIVE_ROOT=find_input('nlp-hw02-l3-live')", 'LIVE_ROOT=find_input('+repr(weights_dataset.split('/')[1])+')')
    cell('code', run)
    cell('markdown', 'Проверить summary/timing/quality и worker_history [2,4], service_state nice. Ответ — test-1 в исходном порядке, только code. CPU-ready не является новым GPU gate.')
    notebook = dict(original, cells=cells)
    target = ROOT/'notebooks'/NAME
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    (BUNDLE/NAME).write_bytes(target.read_bytes())
    metadata = {'id': 'thisarivson/nlp-hw02-adaptive-inference-l3-v6',
        'title': 'NLP HW02 adaptive inference L3 v6', 'code_file': NAME, 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': list(dict.fromkeys([weights_dataset, 'thisarivson/nlp-hw02-26b-inputs'])),
        'kernel_sources': [], 'competition_sources': []}
    limits = {'kind': 'final_inference_l3_v6', 'session_timeout_seconds': 720,
        'watchdog_stop_seconds': 660, 'watchdog_exit_seconds': 690, 'cold_test_runs': 1, 'tune_runs': 1,
        'adaptive': config, 'weights': weights}
    identity = {'base_v5_sha256': base_identity, 'adaptive_config': config, 'weights': weights,
        'embedded_source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
        'notebook_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    for name, value in [('kernel-metadata.json', metadata), ('session-limits.json', limits), ('source-identity.json', identity)]:
        (BUNDLE/name).write_text(json.dumps(value, indent=2)+'\n')
    (ROOT/'final-v6-source-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(target)
    return target, sources


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dataset', default=DEFAULT_DATASET)
    parser.add_argument('--main-weight', default=DEFAULT_MAIN)
    parser.add_argument('--long-weight', default=DEFAULT_LONG)
    parser.add_argument('--main-sha256')
    parser.add_argument('--long-sha256')
    build(**vars(parser.parse_args()))
