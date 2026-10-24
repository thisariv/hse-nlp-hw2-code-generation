"""Offline builder: frozen v8 inference modules + standalone bounded TTA pool."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parent; WORK=ROOT.parent
BUNDLE=ROOT/'bundles/tta_mine'; NAME='nlp-hw02-tta.ipynb'

def sha(b): return hashlib.sha256(b).hexdigest()
def build():
    original=json.loads((ROOT/'notebooks/nlp-hw02-ensemble-inference-v8.ipynb').read_text())
    code=next(c['source'] for c in original['cells'] if c['cell_type']=='code')
    sources=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets)))
    frozen={k:sha(v.encode()) for k,v in sources.items()}
    assert frozen==json.loads((ROOT/'final-v8-source-identity.json').read_text())['embedded_source_sha256']
    for name in ['tta_rules.py','tta_worker.py']: sources[name]=(ROOT/'src'/name).read_text()
    config=dict(models=json.loads((ROOT/'final-v8-config.json').read_text())['models'],
                K=8,batch=64,rules=['dollar-words','ungroup','price-trim'],seed=270042,output='/kaggle/working/tta')
    init='''import os,sys,json,time,signal,subprocess,threading
from pathlib import Path
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
MODULES=Path('/kaggle/working/tta-src'); MODULES.mkdir(exist_ok=True)
SOURCES='''+repr(sources)+'''
for name,source in SOURCES.items(): (MODULES/name).write_text(source)
sys.path.insert(0,str(MODULES))
'''
    run='CONFIG='+repr(config)+'''
OUT=Path(CONFIG['output']);OUT.mkdir(exist_ok=True)
(OUT/'config.json').write_text(json.dumps(CONFIG))
os.environ['TTA_DEADLINE']=str(START+1140.)
child=subprocess.Popen([sys.executable,str(MODULES/'tta_worker.py'),str(OUT/'config.json')],start_new_session=True,stdout=(OUT/'worker.log').open('w'),stderr=subprocess.STDOUT)
def kill():
    try: os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError: pass
def watchdog():
    while child.poll() is None:
        if time.monotonic()-START>=1140.:kill();return
        time.sleep(.2)
threading.Thread(target=watchdog,daemon=True).start()
while child.poll() is None:
    print('TTA heartbeat',round(time.monotonic()-START,1),flush=True)
    time.sleep(20.)
print('exit',child.returncode,'seconds',round(time.monotonic()-START,1),flush=True)
if (OUT/'summary.json').exists(): print((OUT/'summary.json').read_text())
assert child.returncode==0,'TTA worker failed or timed out; retain partial journals, do not repush'
'''
    cells=[nbformat.v4.new_markdown_cell('# ДЗ2 NLP: TTA\n\nPrivate, Internet off, одна видимая T4. Веса/decoder/executor v8. Train-only дизайн; greedy исходного и каждого применимого варианта, K8 T0.7 с исполнением, tune2048/eval2048. Watchdog1140с, backend1200с; 2400 device-секунд даже при резервировании пары T4. Без обучения.'),nbformat.v4.new_code_cell(init),nbformat.v4.new_code_cell(run)]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    for name,s in sources.items():compile(s,name,'exec')
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],'cell','exec')
    nbformat.validate(nb);BUNDLE.mkdir(parents=True,exist_ok=True)
    data=nbformat.writes(nb).encode(); (BUNDLE/NAME).write_bytes(data);(ROOT/'notebooks'/NAME).write_bytes(data)
    meta=dict(id='thisarivson/nlp-hw02-tta',title='NLP HW02 TTA',code_file=NAME,language='python',kernel_type='notebook',is_private=True,
              enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
              dataset_sources=['thisarivson/nlp-hw02-l3-live','thisarivson/nlp-hw02-p1-weights','thisarivson/nlp-hw02-26b-inputs'],
              kernel_sources=['thisarivson/nlp-hw02-b4'],competition_sources=[])
    identity=dict(notebook_sha256=sha(data),frozen_v8_source_sha256=frozen,source_sha256={k:sha(v.encode()) for k,v in sources.items()},config=config)
    for name,x in [('kernel-metadata.json',meta),('source-identity.json',identity),('session-limits.json',dict(session_timeout_seconds=1200,watchdog_stop_seconds=1140,quota_budget_seconds=2400))]:
        (BUNDLE/name).write_text(json.dumps(x,indent=2)+'\n')
    (ROOT/'tta-source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nbformat=True,frozen_v8_modules=len(frozen),notebook_sha256=sha(data),config=config),indent=2))

if __name__=='__main__':build()
