"""Assemble fresh B2 production and reviewer-only smoke; never push or execute GPU."""
import argparse
import hashlib
import json
import shutil
from build_l3 import ROOT,literals,write
from build import INSTALL
from build_b1 import OWNERS


def build(owner='friend2',smoke=False,resume_source=None,transfer_sha=None,hard_seconds=31500):
    assert owner in OWNERS and bool(resume_source)==bool(transfer_sha)
    assert (1800 if smoke else 4800)<=hard_seconds<=31500
    if transfer_sha:assert len(transfer_sha)==64 and all(c in '0123456789abcdef' for c in transfer_sha)
    old=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-b1.ipynb').read_text())
    literal=literals(next(c['source'] for c in old['cells'] if c['cell_type']=='code'))
    sources={n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in literal['SOURCES']}
    for p in (ROOT/'src').glob('b2_*.py'):sources[p.name]=p.read_text()
    sources['qk_attention.py']=(ROOT/'src/qk_attention.py').read_text()
    sources['b2_worker.py']=(ROOT/'b2_worker.py').read_text()
    cfg=dict(literal['CONFIG'],owner=owner,session_kind='b2',gpu_smoke=smoke,dropout=0.,
             main_seconds=min(27000.,hard_seconds-4200),resume_source=resume_source,transfer_sha256=transfer_sha,
             final_session=False,final_q=0.,probe_cooldown_seconds=900.,probe_eval_seconds=600.,
             budget_seconds=hard_seconds-300,worker_deadline_seconds=hard_seconds-300,
             hard_shutdown_seconds=hard_seconds-40,session_ceiling_seconds=hard_seconds,
             io_reserve_seconds=240.,setup_cap_seconds=480.,live_sync_enabled=False)
    if smoke:
        assert not resume_source
        cfg.update(main_seconds=0.,budget_seconds=780,worker_deadline_seconds=780,hard_shutdown_seconds=840,
                   session_ceiling_seconds=900,io_reserve_seconds=15.)
    cfg['comparison_protocol']=dict(status='offline two W peak-LR branches at equal main pairs; C reference at 3.1M; no online selection')
    cfg['port_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    cells=[]
    def cell(kind,source):
        row=dict(id=f'b2-{len(cells):02d}',cell_type=kind,metadata={},source=source)
        if kind=='code':compile(source,'B2 cell','exec');row.update(outputs=[],execution_count=None)
        cells.append(row)
    cell('markdown','# B2: fresh W6+6/768, QK LayerNorm + z-loss\n\n'
        +('Reviewer-only smoke15мин; production не запускается.\n' if smoke else 'GPU0 peak0.001, GPU1 peak0.0006; warmup2000, z-loss1e-4; seed42.\n')
        +'Проба после3 100 032 main pairs и в конце; каждая timed linear cooldown900с, q0, greedy heldout4k+old991+CE, restore. Secrets-free; hard wall≤8ч45мин. notes/b1-ev-b2-plan.md.')
    init=next(c['source'] for c in old['cells'] if c['cell_type']=='code')
    tail=init[init.index('WORKING=Path'):];tail=tail[:tail.index('SOURCES=')]+'SOURCES='+repr(sources)+tail[tail.index('\nfor name,source in SOURCES.items()'):]
    cell('code',init[:init.index('CONFIG=')]+'CONFIG='+repr(cfg)+'\n'+tail.replace('B1 T4','B2 T4'))
    cell('code',INSTALL)
    setup=[c['source'] for c in old['cells'] if c['cell_type']=='code'][-1]
    setup=setup.replace("MODULE_DIR/'b1_worker.py'","MODULE_DIR/'b2_worker.py'").replace('B1 setup','B2 setup').replace('B1 incomplete','B2 incomplete')
    cell('code',setup)
    slug='nlp-hw02-b2-gpu-smoke' if smoke else 'nlp-hw02-kaggle-b2'
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,dict(old,cells=cells))
    folder=ROOT/'bundles'/('b2_gpu_smoke_'+owner if smoke else 'b2_'+owner);folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,folder/target.name)
    datasets=['thisarivson/nlp-hw02-26b-inputs']+([resume_source] if resume_source else [])
    # Kaggle canonicalizes a new kernel from its title; match the existing production slug.
    resource_slug=slug if smoke else 'nlp-hw02-b2'
    write(folder/'kernel-metadata.json',dict(id=owner+'/'+resource_slug,title='NLP HW02 B2'+(' GPU smoke' if smoke else ''),
        code_file=target.name,language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
        enable_internet=True,machine_shape='NvidiaTeslaT4',dataset_sources=datasets,
        kernel_sources=[cfg['cache_source']],competition_sources=[]))
    write(folder/'session-limits.json',dict(session_timeout_seconds=cfg['session_ceiling_seconds'],budget_seconds=cfg['budget_seconds'],
                                         watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(folder/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}))
    return target,cfg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--owner',choices=OWNERS,default='friend2');p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--resume-source');p.add_argument('--transfer-sha');p.add_argument('--hard-seconds',type=int,default=31500)
    a=p.parse_args();print(build(a.owner,a.gpu_smoke,a.resume_source,a.transfer_sha,a.hard_seconds)[0])
