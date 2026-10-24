"""Validate four private R1 account/mode assemblies without running a notebook."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import nbformat
from build_l3 import ROOT,literals,write
from build_r1 import build,default_branches


def main():
    reports=[]
    inherited=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-p1.ipynb').read_text())
    frozen=literals(next(c['source'] for c in inherited['cells'] if c['cell_type']=='code'))['SOURCES']
    final=json.loads((ROOT/'notebooks/nlp-hw02-ensemble-inference-v7.ipynb').read_text())
    final_sources=literals(next(c['source'] for c in final['cells'] if c['cell_type']=='code'))['SOURCES']
    for owner in ('thisarivson','friend1','friend2'):
        for smoke in (False,True):
            path,cfg=build(owner,smoke=smoke)
            nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
            codes=[c.source for c in nb.cells if c.cell_type=='code'];values=literals(codes[0])
            sources=values['SOURCES'];assert values['CONFIG']==cfg
            for name,source in sources.items():
                compile(source,name,'exec')
                original=ROOT/name if (ROOT/name).is_file() else ROOT/'src'/name
                assert (final_sources[name] if name in final_sources else original.read_text())==source,name
            for name,source in frozen.items():assert sources[name]==source, 'Frozen P1/L3 module changed: '+name
            assert not any(n.startswith('d1_') for n in sources)
            assert cfg['port_sha256']==hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
            for c in nb.cells:
                if c.cell_type=='code':compile(c.source,'R1 notebook','exec');assert c.execution_count is None and not c.outputs
            tree=ast.parse(codes[0])
            watchdog=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Watchdog')
            torch_line=next(n.lineno for n in tree.body if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
            assert watchdog<torch_line
            assert 'GATE=optional_gate' in codes[0] and "os.environ.get('KAGGLE_USERNAME')!=CONFIG['owner']" in codes[0]
            assert not cfg['require_secrets'] and "if GATE.get('live_enabled'):" in codes[2]
            assert "from r1_live import R1LiveSync" in codes[2]
            assert codes[-1].index('verify=True')<codes[-1].index('report=launch')
            assert cfg['owner']==owner and cfg['live_dataset']==owner+'/nlp-hw02-r1-live'
            kind='r1_gpu_smoke' if smoke else 'r1';bundle=ROOT/'bundles'/kind
            meta=json.loads((bundle/'kernel-metadata.json').read_text())
            assert meta['is_private'] and meta['id'].startswith(owner+'/') and meta['machine_shape']=='NvidiaTeslaT4'
            assert (bundle/path.name).read_bytes()==path.read_bytes()
            if not smoke:
                assert cfg['hard_shutdown_seconds']==40800 and cfg['worker_deadline_seconds']==40260
                for b in cfg['branches']:
                    assert 480+b['main_seconds']+b['cooldown_seconds']+b['eval_seconds']+180<=40260
                    assert b['q']==.1 and b['cooldown_seconds']==1200
            reports.append(dict(owner=owner,smoke=smoke,notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                source_modules=len(sources),private=True,hard_wall=cfg['hard_shutdown_seconds']))
    # Leave canonical local bundles on friend account; switching is one build flag.
    build('friend2');build('thisarivson',smoke=True)
    for file in [ROOT/'r1_worker.py',*ROOT.glob('r1_*.py'),*ROOT.glob('src/r1_*.py')]:compile(file.read_text(),str(file),'exec')
    regression=json.loads((ROOT.parent/'checks/kaggle-r1-regressions.json').read_text());assert regression['passed']
    receipts={}
    for name in ('inherited-regressions','real-resume-C-main','real-resume-C-long2x',
                 'real-order','cpu-benchmark','states-copy','live-thisarivson','live-friend1'):
        r=json.loads((ROOT.parent/f'checks/kaggle-r1-{name}.json').read_text());assert r['passed'],name
        if name.startswith('real-resume'):
            assert r['steps_bit_exact']==2 and r['downloaded_anchor_removed']
        receipts[name]=r
    write(ROOT.parent/'checks/kaggle-r1-static.json',dict(passed=True,assemblies=reports,CPU_tests=regression['tests'],
          inherited_modules_unchanged=True,nbformat_valid=True,receipts=receipts,GPU=False,push=False,
          production_requires='reviewer T4 smoke and shared inputs; owner Secrets optional'))
    print('R1 STATIC PASSED; three accounts/production+smoke; no GPU/push')


if __name__=='__main__':main()
