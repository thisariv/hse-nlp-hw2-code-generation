"""Собрать private dataset финала; ничего не загружает и не запускает на GPU."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SUBMISSION = WORK.parent / 'submission'
STAGE = ROOT / 'staging/final-package-v11-budget640'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def build():
    STAGE.mkdir(parents=True,exist_ok=True)
    config=json.loads((SUBMISSION/'code/hw2_codegen/config.json').read_text())
    sources=[WORK/'results/kaggle-runs/nlp-hw02-b6/v10-pinned-1/model-GPU1/weights/b6-final-fp16.pt',
             WORK/'results/kaggle-runs/nlp-hw02-kaggle-p1/v9-pinned-3/model-pilot-b/weights/long-truncation-fp16.pt',
             WORK/'results/kaggle-runs/nlp-hw02-b6/v10-pinned-1/model-GPU0/weights/b6-final-fp16.pt']
    for spec,source in zip(config['models'],sources):
        assert sha(source)==spec['sha256']
        shutil.copyfile(source,STAGE/spec['filename'])
    inputs=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    expected={'tokenizer.json':'261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d',
              'test.json':'20cacb334e3fc81291433d2173d9085b17a15e1834bd84aea67b350403182503',
              'heldout4k.json':'28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2',
              'heldout4k-reference-execution.json':'3855680e9a37292aa2e351b8e8e4f7a5cbbe76444096d136470e4a9be8797111'}
    for name,digest in expected.items():assert sha(inputs/name)==digest,name
    for name in ['tokenizer.json','test.json']:shutil.copyfile(inputs/name,STAGE/name)
    heldout=json.loads((inputs/'heldout4k.json').read_text())
    refs=json.loads((inputs/'heldout4k-reference-execution.json').read_text())
    if isinstance(refs,dict):refs=refs['rows']
    ordered=sorted(range(4096),key=lambda i:hashlib.sha256(f'candidate-selection-v1:heldout4k:{i}'.encode()).digest())
    ids=sorted(ordered[:2048])
    for name,rows in [('tune2048.json',[dict(question=heldout[i]['question']) for i in ids]),('tune2048-reference.json',[refs[i] for i in ids])]:
        (STAGE/name).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    # Kaggle extracts uploaded ZIPs: mount an ordinary code/ directory instead.
    (STAGE/'code.zip').unlink(missing_ok=True)
    package=STAGE/'code/hw2_codegen'
    package.mkdir(parents=True,exist_ok=True)
    code_files={}
    for f in sorted((SUBMISSION/'code/hw2_codegen').iterdir()):
        if f.suffix in ('.py','.json'):
            shutil.copyfile(f,package/f.name)
            code_files['code/hw2_codegen/'+f.name]=dict(bytes=f.stat().st_size,sha256=sha(f))
    files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f))for f in sorted(STAGE.iterdir())if f.suffix in ('.pt','.json','.zip')and f.name not in ('dataset-metadata.json','package-manifest.json')}
    manifest=dict(policy='v11-budget640',files=files,code_files=code_files,models=config['models'],tune_ids=ids,private=True)
    (STAGE/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (STAGE/'dataset-metadata.json').write_text(json.dumps(dict(id='thisarivson/nlp-hw02-final-weights',title='NLP HW02 final weights',licenses=[dict(name='other')],description='Private university homework. Three model-only FP16 checkpoints, own tokenizer, readable inference package, fixed original test/tune inputs. No redistribution license granted.'),indent=2)+'\n')
    receipt=dict(private=True,weights={m['name']:dict(filename=m['filename'],sha256=m['sha256'])for m in config['models']},files=files,code_files=code_files,manifest_sha256=sha(STAGE/'package-manifest.json'),notebook_sha256=sha(SUBMISSION/'inference.ipynb'))
    (WORK/'checks/final-package-v11-budget640-local.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(stage=str(STAGE),files=len(files),manifest_sha256=receipt['manifest_sha256'])))

if __name__=='__main__':build()
