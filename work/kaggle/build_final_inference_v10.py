"""Build private friend1 v10; no API calls, fail-closed readiness.

--config accepts a JSON config with 2–4 ordered models (name, filename, sha256,
dataset, K1, K2). Without it, use the selected, verified offline configuration.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from final_ensemble import DEFAULT_MODELS, parameters

BUNDLE = ROOT/'bundles/final_inference_v10_mine'
READY = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v10.json'
NAME = 'nlp-hw02-ensemble-inference-v10.ipynb'
INPUTS_DATASET = 'thisarivson/nlp-hw02-26b-inputs'


def default_config():
    models = [dict(name='C-long2x-B6', filename='b6-final-fp16.pt',
                   relative_path='model-GPU1/weights/b6-final-fp16.pt',
                   sha256='ee30794048bb6200cf97ac4d7e5c1606f0e97893ea04f6567c9487f6c04345fa',
                   dataset='friend1/nlp-hw02-b6', source_kind='notebooks', source_ref='friend1/nlp-hw02-b6/1', K1=1, K2=1),
              dict(DEFAULT_MODELS[2], filename='long-truncation-fp16.pt',
                   relative_path='model-pilot-b/weights/long-truncation-fp16.pt',
                   dataset='friend1/nlp-hw02-kaggle-p1', source_kind='notebooks',
                   source_ref='friend1/nlp-hw02-kaggle-p1/3', K1=1, K2=5),
              dict(name='W-B6', filename='b6-final-fp16.pt',
                   relative_path='model-GPU0/weights/b6-final-fp16.pt',
                   sha256='cb34c81e6cda83c73bf4662c5c425446be8cdb48c30dd5e9e217fd6cbc1cd496',
                   dataset='friend1/nlp-hw02-b6', source_kind='notebooks', source_ref='friend1/nlp-hw02-b6/1', K1=1, K2=6)]
    return {'models': models, 'unanimity_threshold': 1., 'sequence_budget': 1024,
            'greedy_sequence_budget': 1024, 'scorer_enabled': False,
            'scorer_weight': 0., 'expected_matches': 1553}


def build(config=None, owner="friend1"):
    assert owner in ('friend1', 'thisarivson')
    config = default_config() if config is None else dict(config)
    models, threshold = parameters(config)
    assert all('dataset' in m for m in models)
    config.update(models=models, unanimity_threshold=threshold)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    READY.parent.mkdir(parents=True, exist_ok=True)
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False}, indent=2)+'\n')
    base = ROOT/'notebooks'
    original = json.loads((base/'nlp-hw02-ensemble-inference-v8.ipynb').read_text())
    first = next(c['source'] for c in original['cells'] if c['cell_type'] == 'code')
    sources = ast.literal_eval(next(n.value for n in ast.parse(first).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    base_identity = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    assert base_identity == json.loads((ROOT/'final-v8-source-identity.json').read_text())['embedded_source_sha256']
    for name in ['qk_attention.py', 'final_core.py', 'final_fast.py', 'final_speed.py', 'final_adaptive.py', 'final_ensemble.py', 'final_execute_v6.py', 'final_executor_v6.py']:
        sources[name] = (ROOT/'src'/name).read_text()
    sources['final_inference.py'] = (ROOT/'src/final_inference_v7.py').read_text()
    sources['final_mounts.py'] = (ROOT/'src/final_mounts.py').read_text()
    sources['final_session.py'] = (ROOT/'src/final_session_v8.py').read_text()
    datasets = list(dict.fromkeys(m.get('source_ref', m['dataset']) for m in models if m.get('source_kind', 'datasets') == 'datasets'))
    kernels = list(dict.fromkeys(m.get('source_ref', m['dataset']) for m in models if m.get('source_kind') == 'notebooks'))
    cells = []
    def cell(kind, source):
        item = {'cell_type': kind, 'id': f'final-v10-{len(cells):02d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            compile(source, item['id'], 'exec'); item.update(outputs=[], execution_count=None)
        cells.append(item)
    cell('markdown', '''# ДЗ2 NLP: ensemble inference v10

Private, Internet off, одна T4/GPU0. Greedy первой модели → K1 каждой модели → для неединогласных до K2 каждой модели. Порядок моделей и индивидуальные лимиты указаны в ENSEMBLE_CONFIG. Неопределённость: доля лидирующей численной компоненты, затем длина/ID; все модели подряд внутри priority wave. Frozen count/greedy/meanLP/order и прежний выбор кода; голосование по готовым кандидатам при guard. Decoder all и executor2 nice10→до4 сохранены из v6.

Ровно один cold test1919 + tune2048, ≤720с, watchdog660/690. Guard160 / execution166 / verification170 / supervisor172. Timer включает imports/load/warmup/execution/vote/reexecution/write. Цель test≤140с, потолок<180с. C-long2x B6/P1-B/W B6 resident, генерация одной модели за раз, sequence_budget1024 и OOM retry с восстановлением RNG. W: 6+6 d768/12 heads, headwise QK LayerNorm (FP32 statistics), z-loss обучающих весов. K1=1/1/1 → K2=1/5/6; greedy C-long2x B6. Реальные peak allocated/reserved и веса записываются в timing.

Состав и K заданы оркестратором, повторного поиска нет. Offline1553 против v8 1541, CI[−0,34;+1,51]п.п., осторожный proxy136,8с; максимум ранее исследованной сетки и оценка смещена вверх. Выбор K по tune — exploratory; парный CI не скорректирован за выбор конфигурации. Прогноз времени и памяти — локальная экстраполяция T4 profile/v5, полный v10 GPU gate измеряется этим запуском. Compaction меняет FP16 формы и RNG новых samples; offline replay проверяет решения/выбор кода. Известные FP16 greedy отличия all из v6 сохранены. Без обучения и обращения к credentials внутри ноутбука.''')
    init = first[:first.index('SOURCES=')]+'SOURCES='+repr(sources)+first[first.index('\nfor name,source in SOURCES.items()'):]
    cell('code', init.replace('ensemble-inference-v8', 'ensemble-inference-v10'))
    run = '''BENCHMARK=True  # Ровно один холодный test1919 + tune2048.
from final_mounts import find_input, resolve_cache_mount
INPUTS_ROOT=find_input('nlp-hw02-26b-inputs')
'''+"ENSEMBLE_CONFIG="+repr(config)+"\nWEIGHTS_ROOTS={m['dataset']:resolve_cache_mount(Path('/kaggle/input'),m['dataset'],m.get('source_kind','datasets')) for m in ENSEMBLE_CONFIG['models']}\n"+'''from final_session import suite
REPORT=suite(INPUTS_ROOT,WEIGHTS_ROOTS,OUTPUT,MODULES,WATCHDOG,benchmark=BENCHMARK,ensemble_config=ENSEMBLE_CONFIG)
print(json.dumps({'status':REPORT['status'],'test_seconds':[r['full_pipeline_seconds'] for r in REPORT['test']],
                 'quality':REPORT.get('quality'),'gates':REPORT.get('gates'),'session_seconds':REPORT['session_seconds']},indent=2))
'''
    cell('code', run)
    cell('markdown', 'Ответ — единственный test-1, исходный порядок, только code. Проверить summary/timing/quality, model_order/SHA, peak memory, guard/partial/fallback и service_state. Tune gold доступен только для итоговой диагностики; решения его не получают.')
    notebook = dict(original, cells=cells)
    target = ROOT/'notebooks'/NAME
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    (BUNDLE/NAME).write_bytes(target.read_bytes())
    metadata = {'id': owner+'/nlp-hw02-ensemble-inference-v10',
        'title': 'NLP HW02 ensemble inference v10', 'code_file': NAME, 'language': 'python',
        'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': False, 'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': list(dict.fromkeys(datasets+[INPUTS_DATASET])),
        'kernel_sources': kernels, 'competition_sources': []}
    limits = {'kind': 'final_inference_v10', 'session_timeout_seconds': 720,
        'watchdog_stop_seconds': 660, 'watchdog_exit_seconds': 690, 'cold_test_runs': 1, 'tune_runs': 1,
        'adaptive': config, 'generation_guard_seconds': 160, 'test_target_seconds': 140}
    identity = {'base_v8_sha256': base_identity, 'ensemble_config': config,
        'embedded_source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
        'notebook_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    for name, value in [('kernel-metadata.json', metadata), ('session-limits.json', limits), ('source-identity.json', identity)]:
        (BUNDLE/name).write_text(json.dumps(value, indent=2)+'\n')
    (ROOT/'final-v10-source-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    (ROOT/'final-v10-config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(target)
    return target, sources


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--owner', choices=['friend1', 'thisarivson'], default='friend1')
    args = parser.parse_args()
    build(json.loads(args.config.read_text()) if args.config else None, args.owner)
