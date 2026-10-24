"""Build v11 from the frozen v10 notebook: only K2 changes to 1/8/8.

No API calls. Run All starts exactly one cold test1919 and a fresh tune2048.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BUNDLE = ROOT/'bundles/final_inference_v11_mine'
NAME = 'nlp-hw02-ensemble-inference-v11.ipynb'


def read(path):
    return json.loads(path.read_text())


def sources(notebook):
    code = next(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    return ast.literal_eval(next(n.value for n in ast.parse(code).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'SOURCES'))


def default_config():
    config = copy.deepcopy(read(ROOT/'final-v10-config.json'))
    config['models'][1]['K2'] = config['models'][2]['K2'] = 8
    # Keep the existing diagnostic threshold too; it does not affect selection.
    return config


def build():
    original = read(ROOT/'notebooks/nlp-hw02-ensemble-inference-v10.ipynb')
    notebook = copy.deepcopy(original)
    identity10 = read(ROOT/'final-v10-source-identity.json')
    embedded = sources(original)
    assert {n: hashlib.sha256(s.encode()).hexdigest() for n, s in embedded.items()} == identity10['embedded_source_sha256']
    config = default_config()
    for cell in notebook['cells']:
        cell['id'] = cell['id'].replace('v10', 'v11')
        if cell['cell_type'] == 'code':
            code = cell['source'].replace('ensemble-inference-v10', 'ensemble-inference-v11')
            if 'ENSEMBLE_CONFIG=' in code:
                before, rest = code.split('ENSEMBLE_CONFIG=', 1)
                _, after = rest.split('\n', 1)
                code = before + 'ENSEMBLE_CONFIG=' + repr(config) + '\n' + after
            cell.update(source=code, outputs=[], execution_count=None)
            compile(code, cell['id'], 'exec')
    notebook['cells'][0]['source'] = '''# ДЗ2 NLP: ensemble inference v11

Private, Internet off, одна T4/GPU0. Веса, decoder all, selector/vote, executor, RNG/OOM retry, sequence budget1024 и guards160/166/170/172 побайтно сохранены из v10. Единственное изменение политики: greedy C-long2x B6; K1=1/1/1 → K2=1/8/8, C/P1-B/W B6, T0.7/seed270042. После единогласного stage1 остановиться; иначе добавить7 P1-B и7 W. Максимум17 stochastic samples + greedy.

E1: frozen primary выбрана на tune1550/2048 до eval; eval1574 против baseline1559, Δ+0.732п.п. CI[+0.098;+1.367]. Прогноз137.322с — не cold timer. Новые формы/RNG могут изменить samples и качество. Диагностический expected_matches1553 оставлен из v10 для exact config delta; это прежний offline reference, не обещание v11-score.

Run All: ровно один cold test1919 с полным timer и отдельный свежий tune2048; timeout720, watchdog660/690. Скачайте из ensemble-inference-v11/: test-1/answers.jsonl, test-1/timing.json, test-1/summary.json, summary.json. Для проверки exact code parity сохраните исполненный notebook; для независимого replay tune дополнительно tune/answers.jsonl, tune/candidate-journal.json.gz, tune/summary.json, tune/quality.json. Ответ всегда test-1, не выбирается по tune. Обучения нет.'''
    notebook['cells'][-1]['source'] = 'Скачать четыре файла, указанные выше. Проверить status, gates, test time, guard/fallback/partial, реальные SHA весов и порядок строк. Review_required из-за quality — диагностический результат; политика после запуска не меняется.'
    assert sources(notebook) == embedded
    BUNDLE.mkdir(parents=True, exist_ok=True)
    target = ROOT/'notebooks'/NAME
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    (BUNDLE/NAME).write_bytes(target.read_bytes())
    metadata = dict(id='friend1/nlp-hw02-ensemble-inference-v11', title='NLP HW02 ensemble inference v11', code_file=NAME,
        language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
        enable_internet=False, machine_shape='NvidiaTeslaT4', dataset_sources=['thisarivson/nlp-hw02-26b-inputs'],
        kernel_sources=list(dict.fromkeys(m['source_ref'] for m in config['models'])), competition_sources=[])
    limits = dict(kind='final_inference_v11', session_timeout_seconds=720, watchdog_stop_seconds=660,
        watchdog_exit_seconds=690, cold_test_runs=1, tune_runs=1, adaptive=config,
        generation_guard_seconds=160, test_target_seconds=140)
    identity = dict(base_v10_notebook_sha256=identity10['notebook_sha256'], ensemble_config=config,
        embedded_source_sha256=identity10['embedded_source_sha256'],
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    for name, value in [('kernel-metadata.json', metadata), ('session-limits.json', limits), ('source-identity.json', identity)]:
        (BUNDLE/name).write_text(json.dumps(value, indent=2)+'\n')
    for name, value in [('final-v11-config.json', config), ('final-v11-source-identity.json', identity)]:
        (ROOT/name).write_text(json.dumps(value, indent=2)+'\n')
    print(target)
    return target, embedded


if __name__ == '__main__':
    build()
