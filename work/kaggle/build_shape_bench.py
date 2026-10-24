"""Build only; no Kaggle/network/auth operations."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
BUNDLE=ROOT/'bundles/shape_bench_mine'
READY=WORK/'results/next-shape-bench.json'
FILES=['final_core.py','final_fast.py','final_decode.py','final_speed.py','shape_specs.py','shape_bench.py']


def build():
    BUNDLE.mkdir(parents=True,exist_ok=True)
    READY.write_text(json.dumps(dict(bundle=str(BUNDLE.relative_to(WORK)),ready=False))+'\n')
    sources={n:(ROOT/'src'/n).read_text() for n in FILES}
    for n,s in sources.items(): compile(s,n,'exec')
    lengths=json.loads((ROOT/'shape-lengths.json').read_text())
    cells=[]
    def cell(kind,source):
        value=dict(cell_type=kind,id=f'shape-bench-{len(cells)}',metadata={},source=source)
        if kind=='code': compile(source,value['id'],'exec'); value.update(execution_count=None,outputs=[])
        cells.append(value)
    cell('markdown', '''# NLP HW02 shape bench

Один offline прогон на одной T4, ≤20 минут. Случайные веса, без полезного обучения и сохранения checkpoints.
13 форм; контроль C6+6/d512. 8 одинаковых effective batches128 с реальными joint lengths из verified cache v2; AMP/fused AdamW, longest-row micro probe64→1, warmup вне таймеров. Обновления нужны только для измерения forward/backward/optimizer. Перед decode исходные random weights восстановлены.

Frozen v6 decoder all (compaction, preallocated KV, cross projection, race sampler); natural EOS запрещён, завершение принудительно по общему распределению длин L3 C-main. Greedy1024 вопросов; sampling256×K4=1024 последовательности. OOM уменьшает lanes с явной записью. Encoder отдельно1919 вопросов, batch256, 3 повтора. Только длины, текст/targets не используются. Это сравнение форм; качество и полный production timer не измеряются.

Входы: thisarivson/nlp-hw02-26b-inputs, pinned CPU kernel thisarivson/nlp-hw02-compact-cache-verified/2. Длины уже встроены из этого кеша; монтирования не читаются, повторной сборки кеша нет. Internet off, без pip и API. Working deadline1100с, watchdog1150с, stop1180с; partial честно сохраняется при недостающей строке. Скачайте shape-bench/summary.json, timings.csv и run.log.''')
    cell('code', '''import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
SESSION_START=time.monotonic()
ROOT=Path('/kaggle/working/shape-bench-modules'); ROOT.mkdir(exist_ok=True)
OUTPUT=Path('/kaggle/working/shape-bench'); OUTPUT.mkdir(exist_ok=True)
stop=threading.Event(); children=set()
def kill_children():
    for pid in list(children):
        try: os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError: pass
def watch():
    while not stop.wait(.2):
        if time.monotonic()-SESSION_START>=1150:
            kill_children()
            (OUTPUT/'watchdog.json').write_text(json.dumps({'triggered':True,'wall_s':time.monotonic()-SESSION_START}))
            return
threading.Thread(target=watch,daemon=True).start()
'''+f'\nSOURCES={sources!r}\nLENGTHS={lengths!r}\n'+'''
for name,source in SOURCES.items(): (ROOT/name).write_text(source)
(ROOT/'lengths.json').write_text(json.dumps(LENGTHS))
''')
    cell('code', '''env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
command=[sys.executable,'-c',"import shape_bench; shape_bench.run('lengths.json',"+repr(str(OUTPUT))+","+repr(SESSION_START)+")"]
try:
    with (OUTPUT/'run.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        children.add(child.pid)
        while child.poll() is None:
            if time.monotonic()-SESSION_START>=1180: kill_children(); break
            time.sleep(.2)
        child.wait(timeout=3); children.discard(child.pid)
    print('Exit',child.returncode)
    if (OUTPUT/'summary.json').exists():
        report=json.loads((OUTPUT/'summary.json').read_text()); print(report['status'],report.get('wall_s'))
    if (OUTPUT/'timings.csv').exists(): print((OUTPUT/'timings.csv').read_text())
    if child.returncode: print((OUTPUT/'run.log').read_text()[-3000:])
finally:
    kill_children(); stop.set()
''')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    raw=json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
    for folder in (ROOT/'notebooks',BUNDLE):
        folder.mkdir(exist_ok=True); (folder/'nlp-hw02-shape-bench.ipynb').write_text(raw)
    meta=dict(id='thisarivson/nlp-hw02-shape-bench',title='NLP HW02 shape bench',code_file='nlp-hw02-shape-bench.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=False,
        machine_shape='NvidiaTeslaT4',dataset_sources=['thisarivson/nlp-hw02-26b-inputs'],
        kernel_sources=['thisarivson/nlp-hw02-compact-cache-verified/2'],competition_sources=[])
    identity=dict(source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
                  lengths_sha256=hashlib.sha256((ROOT/'shape-lengths.json').read_bytes()).hexdigest(),
                  notebook_sha256=hashlib.sha256(raw.encode()).hexdigest())
    for n,value in [('kernel-metadata.json',meta),('source-identity.json',identity),
                    ('session-limits.json',dict(limit_s=1200,working_s=1100,watchdog_s=1150,push=False,GPU_runs=1))]:
        (BUNDLE/n).write_text(json.dumps(value,indent=2)+'\n')
    return BUNDLE


if __name__=='__main__': print(build())
