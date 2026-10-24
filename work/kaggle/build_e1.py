"""Offline E1 builder. One private run, two GPUs, smoke barrier before full pool."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BUNDLE = ROOT/'bundles/e1_mine'
NAME = 'nlp-hw02-e1-ensembles.ipynb'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(check=False):
    # Committed v8 has exactly the same20runtime modules as v10. E1 owns its
    # config, so rebuilding never requires the unrelated uncommitted v10 files.
    original = json.loads((ROOT/'notebooks/nlp-hw02-ensemble-inference-v8.ipynb').read_text())
    first = next(c['source'] for c in original['cells'] if c['cell_type'] == 'code')
    if isinstance(first,list):first=''.join(first)
    sources = ast.literal_eval(next(n.value for n in ast.parse(first).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets)))
    frozen = {n: sha(s.encode()) for n, s in sources.items()}
    assert frozen == json.loads((ROOT/'final-v8-source-identity.json').read_text())['embedded_source_sha256']
    for name in ('e1_decode.py', 'e1_worker.py'):
        sources[name] = (ROOT/'src'/name).read_text()
    config = json.loads((ROOT/'e1-config.json').read_text())
    init = '''import os,sys,json,time,signal,subprocess,threading,hashlib
from pathlib import Path
START=time.monotonic()
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
MODULES=Path('/kaggle/working/e1-src');MODULES.mkdir(exist_ok=True)
SOURCES='''+repr(sources)+'''
for name,source in SOURCES.items(): (MODULES/name).write_text(source)
sys.path.insert(0,str(MODULES))
'''
    run = 'CONFIG='+repr(config)+'''
OUT=Path(CONFIG['output']);OUT.mkdir(exist_ok=True)
(OUT/'config.json').write_text(json.dumps(CONFIG))
CHILDREN=set();FINISHED=threading.Event();TRIGGERED=threading.Event()
def kill_all():
    for child in list(CHILDREN):
        try: os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError: pass
def watchdog():
    if not FINISHED.wait(max(0.,START+4080.-time.monotonic())):
        TRIGGERED.set();kill_all()
        (OUT/'watchdog.json').write_text(json.dumps({'triggered':True,'seconds':time.monotonic()-START}))
threading.Thread(target=watchdog,daemon=True).start()
REPORT={'status':'running','training':False,'test_read':False,'splits':{},'GPU0':'tune2048','GPU1':'eval2048'}
def save():
    REPORT['seconds']=time.monotonic()-START
    (OUT/'summary.json').write_text(json.dumps(REPORT,indent=2))
def launch(phase):
    assert not TRIGGERED.is_set(),'E1 watchdog triggered'
    jobs=[]
    for gpu,split in enumerate(('tune2048','eval2048')):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),E1_DEADLINE=str(START+4050.))
        log=(OUT/(phase+'-'+split+'.log')).open('w')
        child=subprocess.Popen([sys.executable,str(MODULES/'e1_worker.py'),str(OUT/'config.json'),split,phase],
            env=env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        CHILDREN.add(child);jobs.append((split,child,log))
    try:
        while any(c.poll() is None for _,c,_ in jobs):
            assert not TRIGGERED.is_set(),'E1 session watchdog; retain output, do not repush'
            failed=[s for s,c,_ in jobs if c.poll() not in (None,0)]
            assert not failed,'E1 '+phase+' failed: '+str(failed)+'; inspect worker log'
            print('E1 heartbeat',phase,round(time.monotonic()-START,1),flush=True)
            time.sleep(20.)
        for split,child,_ in jobs:
            assert child.returncode==0,'E1 '+phase+' failed for '+split
            report=json.loads((OUT/phase/split/'summary.json').read_text())
            assert report['status']=='complete' and report['rows']==(64 if phase=='smoke' else 2048)
            assert report['candidates_per_row']==87 and report['test_read'] is False
            assert report['executor']['unique_programs']==report['executor']['completed_unique']
            REPORT['splits'][phase+'/'+split]=report
        save()
    finally:
        for _,child,log in jobs:
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            child.wait(timeout=5);CHILDREN.discard(child);log.close()
try:
    save();launch('smoke')
    # Hard barrier: neither full split starts unless BOTH 64-row smoke jobs pass.
    REPORT['smoke_passed']=True;save()
    launch('full')
    REPORT['status']='complete';save()
except BaseException as error:
    REPORT.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
finally:
    FINISHED.set();kill_all()
print(json.dumps({'status':REPORT['status'],'smoke_passed':REPORT.get('smoke_passed'),
                  'seconds':REPORT['seconds']},indent=2))
'''
    cells = [nbformat.v4.new_markdown_cell('''# E1: другие способы ансамблирования

Private / Internet off / T4×2. GPU0 — tune2048, GPU1 — eval2048, frozen split.
Сначала оба smoke64; при ошибке полный пул не запускается. Greedy C/P/W,
log-linear ensemble и soups α0.3/0.5/0.7; K16 T0.7 и K8 T1.0 каждой модели,
K8 T0.7 ансамбля. 87 кандидатов на строку, v10 executor и cross LL трёх моделей.
Test не читается, accuracy здесь не считается. Run All запускает тот же протокол.
Ожидание ≤60 мин wall; watchdog4080с, backend4200с. Сжатые журналы в e1/.
'''), nbformat.v4.new_code_cell(init), nbformat.v4.new_code_cell(run)]
    target=ROOT/'notebooks'/NAME
    if target.exists():
        previous=json.loads(target.read_text())
        assert len(cells)==len(previous['cells'])
        for cell,old in zip(cells,previous['cells']):cell['id']=old['id']
    else:
        for i,cell in enumerate(cells):cell['id']=f'e1-{i:02d}'
    nb = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec':{
        'display_name':'Python 3','language':'python','name':'python3'}})
    for name, source in sources.items():
        compile(source, name, 'exec')
    for cell in cells:
        if cell['cell_type'] == 'code':
            compile(cell['source'], 'E1 cell', 'exec')
    nbformat.validate(nb)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    data = nbformat.writes(nb).encode()
    if check:
        assert data==target.read_bytes(),'Rebuild differs from the reviewed notebook'
        assert sha(data)==json.loads((ROOT/'e1-source-identity.json').read_text())['notebook_sha256']
        print(json.dumps(dict(passed=True,byte_exact_rebuild=True,notebook_sha256=sha(data))))
        return
    target.write_bytes(data); (BUNDLE/NAME).write_bytes(data)
    meta = dict(id='friend1/nlp-hw02-e1-ensembles', title='NLP HW02 E1 ensembles', code_file=NAME,
                language='python', kernel_type='notebook', is_private=True, enable_gpu=True,
                enable_tpu=False, enable_internet=False, machine_shape='NvidiaTeslaT4',
                dataset_sources=['thisarivson/nlp-hw02-26b-inputs'],
                kernel_sources=['friend1/nlp-hw02-b6/1','friend1/nlp-hw02-kaggle-p1/3'],competition_sources=[])
    identity = dict(notebook_sha256=sha(data), frozen_v10_source_sha256=frozen,
                    source_sha256={n:sha(s.encode()) for n,s in sources.items()}, config=config)
    for name, obj in [('kernel-metadata.json',meta),('source-identity.json',identity),
                      ('session-limits.json',dict(session_timeout_seconds=4200,watchdog_seconds=4080,
                                                expected_wall_seconds=3600,maximum_device_seconds=8400))]:
        (BUNDLE/name).write_text(json.dumps(obj,indent=2)+'\n')
    (ROOT/'e1-source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(dict(built=True,notebook_sha256=sha(data),frozen_modules=len(frozen))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    args=parser.parse_args();build(args.check)
