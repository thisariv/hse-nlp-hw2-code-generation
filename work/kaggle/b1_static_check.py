#!/usr/bin/env python3
"""Offline notebook/config/source checks. Rebuild defaults last; no API or GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
from build_b1 import build, ROOT, OWNERS
from build_l3 import literals


def main():
    assemblies=[]
    for owner in OWNERS:
        for smoke in (False, True):
            target,cfg=build(owner=owner,smoke=smoke,dropout=.1)
            nb=json.loads(target.read_text());cells=[c for c in nb['cells'] if c['cell_type']=='code']
            for c in cells:compile(c['source'],target.name,'exec')
            embedded=literals(cells[0]['source']);sources=embedded['SOURCES']
            for name,source in sources.items():compile(source,name,'exec')
            for name in ('b1_runtime.py','b1_data.py','b1_session.py','b1_gpu_smoke.py','b1_eval.py','b1_resume.py'):
                assert sources[name]==(ROOT/'src'/name).read_text()
            assert sources['b1_worker.py']==(ROOT/'b1_worker.py').read_text()
            assert cfg['dropout']==.1 and not cfg['live_sync_enabled']
            if smoke:
                assert cfg['worker_deadline_seconds']==780 and cfg['hard_shutdown_seconds']==840 and cfg['session_ceiling_seconds']==900
            assert cfg['hard_shutdown_seconds']<=40800 and cfg['session_ceiling_seconds']<=40800
            assert cells[0]['source'].index('\nWATCHDOG=')<cells[0]['source'].index('\nimport torch\n')
            folder=ROOT/'bundles'/('b1_gpu_smoke' if smoke else 'b1')
            meta=json.loads((folder/'kernel-metadata.json').read_text())
            assert meta['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
            assert meta['dataset_sources']==['thisarivson/nlp-hw02-26b-inputs']
            assert 'require_secrets=False' in cells[0]['source']
            assert 'input-preflight.json' in cells[-1]['source']
            assert meta['is_private'] and meta['enable_gpu'] and meta['id'].startswith(owner+'/')
            assert hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()==cfg['port_sha256']
            assemblies.append(dict(owner=owner,smoke=smoke,modules=len(sources),passed=True))
    _,cfg=build(owner='thisarivson',resume_source='thisarivson/b1-states-b1',transfer_sha='a'*64,final=True)
    assert cfg['final_session'] and cfg['final_q']==.1 and cfg['main_seconds']==28800.
    # Full 3-probe envelope, setup and final save reserve, excludes unused slack.
    fresh=34200+3*(900+600)+480+240
    final=28800+3*(900+600)+3600+2400+480+240
    assert max(fresh,final)<=cfg['worker_deadline_seconds']
    for p in [*ROOT.glob('b1_*.py'),ROOT/'build_b1.py',*(ROOT/'src').glob('b1_*.py')]:ast.parse(p.read_text(),str(p))
    build();build(smoke=True)
    result=dict(passed=True,assemblies=assemblies,resume_final_assembly=True,
                session_envelope_seconds=dict(fresh=fresh,final=final),hard_wall_seconds=40800,
                GPU_run=False,push=False)
    (ROOT.parent/'checks/b1-static.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
