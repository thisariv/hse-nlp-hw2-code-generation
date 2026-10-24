"""Build a fresh E1-format single-model pool; no GPU/network actions."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parent; WORK=ROOT.parent
BUNDLE=ROOT/'bundles/e2_mine'; NAME='nlp-hw02-e2-policy.ipynb'


def sha(data): return hashlib.sha256(data).hexdigest()
def read(p): return json.loads(Path(p).read_text())


def worker_source(original):
    """Explicit, checked removal of token ensemble/soups/cross LL from E1."""
    source=original
    begin=source.index("        generate('E/g', models")
    end=source.index("        assert executor.drain(deadline)",begin)
    source=source[:begin]+'''        # E2 only needs execution voting: no token ensemble, soups, or LL.
        jobs = [(i,u) for i in ids for u in range(len(records[i]['candidates']))]
'''+source[end:]
    old=" and set(c['ll']) == {'C','P','W'}"
    assert source.count(old)==1; source=source.replace(old,'')
    old="ll_tokenization='canonical represented_code + EOS; sum and mean; PAD/BOS masked'"
    assert source.count(old)==1; source=source.replace(old,"cross_LL=False, token_ensemble=False, soups=False")
    source=source.replace('def prepare(root, target):','def prepare(root, target, expected_sha=None):')
    source=source.replace("assert sha(path) == INPUT_SHA[name]", "assert sha(path) == (expected_sha or INPUT_SHA)[name]")
    source=source.replace("prepare(find_input('nlp-hw02-26b-inputs'), out/'inputs')",
        "prepare(find_input(config['input_slug']), out/'inputs', config['input_sha256'])")
    compile(source,'e2_worker.py','exec')
    return source


def build():
    nb=read(ROOT/'notebooks/nlp-hw02-e1-ensembles.ipynb')
    init=''.join(nb['cells'][1]['source'])
    sources=ast.literal_eval(next(n.value for n in ast.parse(init).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets)))
    identity=read(ROOT/'e1-source-identity.json')
    assert {n:sha(s.encode()) for n,s in sources.items()}==identity['source_sha256']
    sources['e2_worker.py']=worker_source(sources.pop('e1_worker.py'))
    config=read(ROOT/'e1-config.json')
    for s in config['models']:
        if s['name']=='pilot-b':
            s.update(dataset='thisarivson/nlp-hw02-p1-weights',source_kind='datasets',
                     relative_path='model-pilot-b__weights__long-truncation-fp16.pt')
        else:
            s.update(dataset='thisarivson/nlp-hw02-b6-fp16',source_kind='datasets',
                     relative_path=s['relative_path'].replace('/','__'))
        s.pop('source_ref',None)
        s['filename']=Path(s['relative_path']).name
    config.update(seed=2026102001,output='/kaggle/working/e2',
                  input_slug='nlp-hw02-26b-inputs',
                  input_sha256={
                    'tokenizer.json':'261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d',
                    'heldout4k.json':'28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2',
                    'heldout4k-reference-execution.json':'3855680e9a37292aa2e351b8e8e4f7a5cbbe76444096d136470e4a9be8797111'},
                  selection_sha256=sha((WORK/'results/e2-selection.json').read_bytes()))
    assert config['selection_sha256']==read(WORK/'checks/e2-selection.json')['selection_sha256']
    for k in ('KE','alphas','ensemble_lanes','ll_batch','ll_token_budget'): config.pop(k)
    init='''import os,sys,json,time,signal,subprocess,threading,hashlib
from pathlib import Path
START=time.monotonic()
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0')
MODULES=Path('/kaggle/working/e2-src');MODULES.mkdir(exist_ok=True)
SOURCES='''+repr(sources)+'''
for name,source in SOURCES.items(): (MODULES/name).write_text(source)
sys.path.insert(0,str(MODULES))
'''
    # Same E1 smoke barrier and two split workers, with the smaller pool.
    run=''.join(nb['cells'][2]['source'])
    config_end=run.index('\nOUT=')
    run='CONFIG='+repr(config)+run[config_end:]
    run=run.replace('4080.','1740.').replace('4050.','1710.')
    run=run.replace("MODULES/'e1_worker.py'", "MODULES/'e2_worker.py'")
    run=run.replace("report['candidates_per_row']==87", "report['candidates_per_row']==75")
    run=run.replace("'E1", "'E2")
    cells=[nbformat.v4.new_markdown_cell('''# E2: свежий execution pool

Private / Internet off / T4×2. Run All выполняет smoke64 обеих GPU,
затем один полный пул4096строк (два непересекающихся split2048).
Greedy C/P/W,16samples T0.7 (1+15),8samples T1.0 каждой модели:
75кандидатов на строку. Без token ensemble/soups/crossLL.
Test не читается; accuracy и выбор политики внутри notebook не считаются.
Набор heldout4k совпадает с E1: это проверка нового sampling на тех же задачах.
Primary заморожена до запуска; независимый holdout по задачам не заявляется.
'''),nbformat.v4.new_code_cell(init),nbformat.v4.new_code_cell(run)]
    for i,c in enumerate(cells):
        c['id']=f'e2-{i:02d}'
        if c['cell_type']=='code':compile(c['source'],'E2cell','exec')
    for name,source in sources.items(): compile(source,name,'exec')
    built=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(built); data=nbformat.writes(built).encode()
    BUNDLE.mkdir(parents=True,exist_ok=True)
    (ROOT/'notebooks'/NAME).write_bytes(data); (BUNDLE/NAME).write_bytes(data)
    result=dict(notebook_sha256=sha(data),source_sha256={n:sha(s.encode()) for n,s in sources.items()},
                e1_source_sha256=identity['source_sha256'], config=config,
                independent_tasks=False, derivative='E1 without E/token/soup/LL; fresh seed; datasets')
    metadata=dict(id='thisarivson/nlp-hw02-e2-policy',title='NLP HW02 E2 policy',code_file=NAME,
                  language='python',kernel_type='notebook',is_private=True,enable_gpu=True,
                  enable_tpu=False,enable_internet=False,machine_shape='NvidiaTeslaT4',
                  dataset_sources=['thisarivson/nlp-hw02-26b-inputs','thisarivson/nlp-hw02-p1-weights',
                                   'thisarivson/nlp-hw02-b6-fp16'],
                  kernel_sources=[],competition_sources=[])
    for name,obj in [('source-identity.json',result),('kernel-metadata.json',metadata),
                     ('session-limits.json',dict(session_timeout_seconds=1800,watchdog_seconds=1740,
                       maximum_device_seconds=3600))]:
        (BUNDLE/name).write_text(json.dumps(obj,indent=2)+'\n')
    (ROOT/'e2-source-identity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(built=True,notebook_sha256=result['notebook_sha256'],phase2_started=False)))


if __name__=='__main__': build()
