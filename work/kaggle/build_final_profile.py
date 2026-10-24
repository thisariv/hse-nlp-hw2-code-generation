"""Build thisarivson's private offline T4 profile bundle. No Kaggle API calls."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BUNDLE = ROOT/'bundles/final_profile_mine'
READY = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-profile.json'
NAME = 'nlp-hw02-decode-profile.ipynb'
FILES = ['final_core.py', 'final_fast.py', 'final_decode.py', 'final_representation.py',
         'final_executor.py', 'final_execute.py', 'final_numeric.py', 'final_speed.py',
         'final_profile.py', 'final_profile_load.py']


def build(scope='paired-subset'):
    BUNDLE.mkdir(parents=True, exist_ok=True)
    READY.parent.mkdir(parents=True, exist_ok=True)
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False})+'\n')
    sources = {name: (ROOT/'src'/name).read_text() for name in FILES}
    for name, source in sources.items():
        compile(source, name, 'exec')
    cells = []
    def cell(kind, source):
        row = {'cell_type': kind, 'id': f'decode-profile-{len(cells):02d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            compile(source, row['id'], 'exec')
            row.update(execution_count=None, outputs=[])
        cells.append(row)
    cell('markdown', '''# NLP HW02 decode profile

thisarivson, private, Internet off, одна T4/GPU0. Один запуск, без обучения и голосования.
Входы: thisarivson/nlp-hw02-l3-live и thisarivson/nlp-hw02-26b-inputs. Только C-main L3 l2-62280, SHA проверяется.

Сначала full test1919 baseline/all × greedy/K4 × без нагрузки/с нагрузкой исполнения; greedy token IDs сравниваются после дополнения PAD до512, то есть ответы побитово. После — парные абляции отдельных флагов на192 вопросах, равномерно по рангу длины test1919. Их нельзя выдавать за full1919 throughput. SCOPE='full' переключает все абляции на1919, но матрица может не уложиться в лимит. Таблица всегда содержит scope/questions.

Флаги: share_cross; async_eos (32 шага вместо синхронных16); compact_done; race_sampler; lean_greedy; workers2; nice10; all. Предвыделенный self-KV, однократный cross-KV и Torch threads1 уже есть в baseline: отмечены как тождественные, без фиктивных повторов. CUDA graphs отклонены из-за динамического prefix/адресов/форм и риска изменения greedy. Для compaction GPU bit-exact — обязательный gate, не предположение. RNG-поток сэмплов может измениться; softmax/temperature и категориальный закон сохранены.

Нагрузка — непрерывные синтетические численные программы через реальный frozen executor: базово4 workers/nice0, отдельно2 workers или nice10, вместе2/nice10. Это контролируемая нагрузка, не воспроизведение реального распределения кандидатов. После генерации драйвер завершает сервис; рабочая final inference пока не меняется.

Working deadline650с, watchdog690с, внешний предел720с от первой code cell. summary.json/timings.csv сохраняются после каждого замера. Четыре32-step torch.profiler trace (.json.gz), K4/1024 lanes, и summary: kernels/CPU ops/launch/sync, доли перекрываются; GPU idle — верхняя оценка host overhead, а не точное Python time. budget_stop_partial не проходит GPU gate. Цель2× выводится из full baseline/all; успешность и качество K8+K8 требуют следующего отдельного end-to-end прогона.''')
    cell('code', """import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
SESSION_START = time.monotonic()
OUTPUT = Path('/kaggle/working/decode-profile')
MODULES = OUTPUT/'modules'
MODULES.mkdir(parents=True, exist_ok=True)
"""+'SOURCES = '+repr(sources)+"""
for name, source in SOURCES.items():
    (MODULES/name).write_text(source)
stop_watchdog = threading.Event()
children = set()
def kill_children():
    for pid in list(children):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
def watch():
    while not stop_watchdog.wait(.2):
        if time.monotonic()-SESSION_START >= 690:
            kill_children()
            (OUTPUT/'watchdog.json').write_text(json.dumps({'triggered': True, 'seconds': time.monotonic()-SESSION_START}))
            return
threading.Thread(target=watch, daemon=True, name='decode-profile-watchdog').start()
print('Private offline profile; no API calls, no training; watchdog690s')
""")
    cell('code', f"SCOPE = {scope!r}\n"+'''def find_input(slug, root=Path('/kaggle/input')):
    direct = root/slug
    if direct.is_dir():
        return direct
    hits = [p for p in root.rglob(slug) if p.is_dir()]
    assert len(hits) == 1, f'Cannot resolve mount {slug}: {hits}'
    return hits[0]
inputs = find_input('nlp-hw02-26b-inputs')
live = find_input('nlp-hw02-l3-live')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='0')
command = [sys.executable, str(MODULES/'final_profile.py'), '--inputs', str(inputs), '--live', str(live),
           '--output', str(OUTPUT), '--start', str(SESSION_START), '--deadline', str(SESSION_START+650),
           '--scope', SCOPE, '--ablation-questions', '192']
try:
    with (OUTPUT/'run.log').open('w') as log:
        child = subprocess.Popen(command, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        children.add(child.pid)
        while child.poll() is None:
            if time.monotonic()-SESSION_START >= 695:
                kill_children()
                break
            time.sleep(.2)
        child.wait(timeout=3)
        children.discard(child.pid)
    report = json.loads((OUTPUT/'summary.json').read_text())
    print(json.dumps({k: report.get(k) for k in ['status', 'matrix_complete', 'GPU_gate_passed', 'full_greedy_bit_exact',
                     'full_speedups', 'target_2x_all_generation', 'session_seconds']}, indent=2))
    if (OUTPUT/'timings.csv').exists():
        print((OUTPUT/'timings.csv').read_text())
    print('Exit:', child.returncode, 'artifacts: /kaggle/working/decode-profile')
finally:
    kill_children()
    stop_watchdog.set()
''')
    cell('markdown', 'Скачать decode-profile/summary.json, timings.csv, trace-*.summary.json и trace-*.json.gz. GPU_gate_passed означает отсутствие greedy token mismatches; target_2x_all_generation проверяется отдельно. Проверка CPU и готовность бандла не заменяют эти GPU gates. Если матрица partial — не делать выводов по отсутствующим ячейкам.')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.11'}},
                 'nbformat': 4, 'nbformat_minor': 5}
    raw = json.dumps(notebook, indent=1, ensure_ascii=False)+'\n'
    (ROOT/'notebooks'/NAME).write_text(raw)
    (BUNDLE/NAME).write_text(raw)
    metadata = {'id': 'thisarivson/nlp-hw02-decode-profile', 'title': 'NLP HW02 decode profile',
                'code_file': NAME, 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'machine_shape': 'NvidiaTeslaT4',
                'dataset_sources': ['thisarivson/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-26b-inputs'],
                'kernel_sources': [], 'competition_sources': []}
    limits = {'session_timeout_seconds': 720, 'working_deadline_seconds': 650,
              'watchdog_stop_seconds': 690, 'training': False, 'voting': False, 'scope': scope,
              'full_greedy_gate_rows': 1919, 'ablation_questions': 192, 'GPU_runs': 1,
              'CUDA_graphs': False, 'automatic_push': False}
    identity = {'source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
                'notebook_sha256': hashlib.sha256(raw.encode()).hexdigest()}
    for filename, value in [('kernel-metadata.json', metadata), ('session-limits.json', limits),
                             ('source-identity.json', identity)]:
        (BUNDLE/filename).write_text(json.dumps(value, indent=2)+'\n')
    (ROOT/'final-profile-source-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    return BUNDLE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['full', 'paired-subset'], default='paired-subset')
    args = parser.parse_args()
    print(build(args.scope))
