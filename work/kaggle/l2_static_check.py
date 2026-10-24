#!/usr/bin/env python3
"""Compile artifacts, verify embedded sources/private metadata/budgets; never execute notebook."""
import ast, inspect, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha,write
import l2_runtime as R

def literals(source):
    result={}
    for statement in ast.parse(source).body:
        if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id
            if name in ('CONFIG','SOURCES'):result[name]=ast.literal_eval(statement.value)
    return result

def main():
    sources={p.name:p.read_text() for p in (ROOT/'src').glob('*.py')}
    for name in ('l2_worker.py','l2_compare.py'):sources[name]=(ROOT/name).read_text()
    for p in list((ROOT/'src').glob('*.py'))+list(ROOT.glob('l2*.py'))+[ROOT/'build_l2.py']:compile(p.read_text(),str(p),'exec')
    anchor=json.loads((ROOT/'l2-anchor.json').read_text());real=json.loads((ROOT.parent/'checks/kaggle-l2-real-resume.json').read_text())
    assert real['passed'] and real['main_bit_exact_CPU_continuation'] and real['fork_initial_state_exact']
    assert (anchor['updates'],anchor['cursor'],anchor['main_seconds'])==(47155,6035840,17569.935878398)
    for n,h in anchor['identity']['recipe']['implementation_sha256'].items():assert sha(ROOT/'src'/n)==h
    outputs={}
    for kind,slug in [('l2','nlp-hw02-kaggle-l2'),('l2_gpu_smoke','nlp-hw02-l2-gpu-smoke')]:
        path=ROOT/'notebooks'/f'{slug}.ipynb';bundle=ROOT/'bundles'/kind
        nb=json.loads(path.read_text());codes=[c for c in nb['cells'] if c['cell_type']=='code']
        for cell in codes:
            assert cell['outputs']==[] and cell['execution_count'] is None
            compile(cell['source'],str(path),'exec')
        values=literals(codes[0]['source']);cfg=values['CONFIG']
        assert values['SOURCES']==sources and not cfg.get('preparation_blocker')
        assert cfg['anchor_pin']==anchor
        tree=ast.parse(codes[0]['source']);calls={node.func.id:node.lineno for node in ast.walk(tree)
             if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        assert 'verify_anchor' in calls and 'prepare_starter' not in calls
        assert calls['Watchdog'] < calls['gate'] < calls['verify_anchor']
        meta=json.loads((bundle/'kernel-metadata.json').read_text());limits=json.loads((bundle/'session-limits.json').read_text())
        assert meta['is_private'] is True and meta['enable_internet'] is True and meta['machine_shape']=='NvidiaTeslaT4'
        assert meta['dataset_sources']==['thisarivson/nlp-hw02-26b-inputs','thisarivson/nlp-hw02-l1-live']
        assert meta['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
        assert sha(bundle/path.name)==sha(path)
        if kind=='l2':
            assert cfg['worker_deadline_seconds']==39600 and cfg['hard_shutdown_seconds']==41040
            assert limits['session_timeout_seconds']==43200
        else:assert cfg['gpu_smoke'] and cfg['hard_shutdown_seconds']==1020
        outputs[kind]={'notebook_sha256':sha(path),'metadata':meta,'limits':limits}
    defaults=inspect.signature(R.run).parameters
    assert defaults['main_targets'].default==(10800.,21600.,28800.)
    assert defaults['cooldown_seconds'].default==(900.,900.,3000.)
    assert defaults['eval_seconds'].default==(900.,900.,1800.)
    # Setup + main + cooldowns + evaluations + I/O fits workers; final sync inside hard ceiling.
    assert 900+28800+1800+1800+3000+1800+1500==39600
    assert 39600+1200+240==41040 < 43200 < 45000
    created=json.loads((ROOT.parent/'checks/kaggle-l2-live-created.json').read_text())
    assert created['private'] and created['status']=='ready' and not created['training_started']
    write(ROOT.parent/'checks/kaggle-l2-static.json',{'passed':True,'artifacts':outputs,'GPU':False,
          'production_push':False,'smoke_push':False,'private_live_precreated':created,
          'cache_versioned_status':'COMPLETE via kaggle_versioned.py status thisarivson/nlp-hw02-compact-cache-verified/2'})
    print('L2 STATIC PASSED: source parity, compile, private metadata, anchor/order pins, budgets')

if __name__=='__main__':main()
