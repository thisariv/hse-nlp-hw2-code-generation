"""Prepare thisarivson B3 and <=12min reviewer smoke. Never push or execute GPU."""
import argparse, hashlib, json, shutil
from build_l3 import ROOT, literals, write
from build import INSTALL


def build(smoke=False):
    receipt=json.loads((ROOT.parent/'checks/b3-transfer.json').read_text())
    assert receipt['passed'] and receipt['private'] and receipt['remote_SHA_verified'] and receipt['real_CPU_checks']
    pins=json.loads((ROOT/'b3-anchors.json').read_text())
    assert receipt['source']==pins['source'] and receipt['dataset']=='thisarivson/nlp-hw02-b2-states'
    old=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-b2.ipynb').read_text())
    literal=literals(next(c['source'] for c in old['cells'] if c['cell_type']=='code'))
    sources={n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in literal['SOURCES']}
    for p in (ROOT/'src').glob('b3_*.py'): sources[p.name]=p.read_text()
    sources['b3_worker.py']=(ROOT/'b3_worker.py').read_text()
    for pin in pins['anchors'].values():
        for n,h in pin['identity']['recipe']['implementation_sha256'].items():
            assert hashlib.sha256(sources[n].encode()).hexdigest()==h,n
    hard=720 if smoke else 40800
    cfg=dict(literal['CONFIG'],owner='thisarivson',session_kind='b3',gpu_smoke=smoke,
             anchors=pins['anchors'],resume_source=receipt['dataset']+'/'+str(receipt['version']),
             transfer_sha256=receipt['transfer_sha256'],main_seconds=32400.,probe_main_marks=[37800.,48600.],
             final_session=True,final_q=0.,probe_cooldown_seconds=900.,probe_eval_seconds=450.,
             final_cooldown_seconds=3000.,final_eval_seconds=1800.,io_reserve_seconds=240.,
             checkpoint_seconds=900.,setup_cap_seconds=480.,budget_seconds=hard-300,
             worker_deadline_seconds=hard-300,hard_shutdown_seconds=hard-60,session_ceiling_seconds=hard,
             live_sync_enabled=False,live_sync_optional=True)
    if smoke: cfg.update(budget_seconds=660,worker_deadline_seconds=660,hard_shutdown_seconds=690,
                         setup_cap_seconds=300,io_reserve_seconds=20.,smoke_updates=200)
    cfg['comparison_protocol']=dict(status='B3 strict both W-QK continuations; cumulative main_s and pairs; C ensemble analysis offline')
    cfg['port_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    cells=[]
    def cell(kind,src):
        row=dict(id=f'b3-{len(cells):02d}',cell_type=kind,metadata={},source=src)
        if kind=='code': compile(src,'B3 cell','exec'); row.update(outputs=[],execution_count=None)
        cells.append(row)
    cell('markdown','# B3: strict B2 W-QK continuation\n\n'+
         ('Reviewer smoke, hard 12min. Real B2 states; no fresh fallback.\n' if smoke else
          'GPU0 W-lr1000 / GPU1 W-lr600. Main cap +9h, probes +3h/+6h, final 50min linear cooldown. Hard 11h20min.\n')+
         'Private thisarivson; Secrets-free. Manual launch after 2026-10-17 00:00 UTC. notes/b3-plan.md.')
    init=next(c['source'] for c in old['cells'] if c['cell_type']=='code')
    tail=init[init.index('WORKING=Path'):]
    tail=tail[:tail.index('SOURCES=')]+'SOURCES='+repr(sources)+tail[tail.index('\nfor name,source in SOURCES.items()'):]
    tail=tail.replace('B2 T4','B3 T4')
    env_gate="""expected=CONFIG['anchors']['0']['identity']['recipe']
assert str(torch.__version__)==expected['torch'] and torch.version.cuda==expected['CUDA'],'B2 resume runtime changed'
write(WORKING/'resume-runtime-gate.json',dict(passed=True,torch=str(torch.__version__),CUDA=torch.version.cuda))
"""
    tail=tail.replace("assert GATE['passed'],GATE['reason']\n","assert GATE['passed'],GATE['reason']\n"+env_gate)
    cell('code',init[:init.index('CONFIG=')]+'CONFIG='+repr(cfg)+'\n'+tail)
    cell('code',INSTALL)
    setup=[c['source'] for c in old['cells'] if c['cell_type']=='code'][-1]
    setup=setup.replace("MODULE_DIR/'b2_worker.py'","MODULE_DIR/'b3_worker.py'").replace('B2 setup','B3 setup')
    # Keep only the resume arm; no fresh-order fallback even exists in B3.
    begin=setup.index("if CONFIG['resume_source']:"); end=setup.index("write(WORKING/'input-preflight.json'")
    arm=setup[begin:end].split('\n',1)[1].split('\nelse:',1)[0]
    arm='\n'.join(line[4:] if line.startswith('    ') else line for line in arm.splitlines())
    setup=setup[:begin]+"assert CONFIG['resume_source'],'B3 requires B2 states'\n"+arm+'\n'+setup[end:]
    setup=setup.replace("source=resolve_cache_mount(inp,CONFIG['cache_source'],kind)",
                        "source=resolve_cache_mount(inp,CONFIG['cache_source'],kind);cache_mount=source")
    setup=setup.replace('cache_mount=str(source),','cache_mount=str(cache_mount),resume_mount=str(source),')
    setup=setup.replace("assert report['status']=='complete' and report['exit_codes']==[0,0],'Inspect per-worker stable/output'",
                        "write(WORKING/'b3-publication.json',dict(status=report['status'],exit_codes=report['exit_codes'],saved_output=True))")
    cell('code',setup)
    slug='nlp-hw02-b3-gpu-smoke' if smoke else 'nlp-hw02-b3'
    target=ROOT/'notebooks'/f'{slug}.ipynb'; write(target,dict(old,cells=cells))
    folder=ROOT/'bundles'/('b3_gpu_smoke_thisarivson' if smoke else 'b3_thisarivson'); folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',dict(id='thisarivson/'+slug,title=slug,code_file=target.name,
          language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
          enable_internet=True,machine_shape='NvidiaTeslaT4',
          dataset_sources=[receipt['dataset'],'thisarivson/nlp-hw02-26b-inputs'],
          kernel_sources=['thisarivson/nlp-hw02-compact-cache-verified/2'],competition_sources=[]))
    write(folder/'session-limits.json',dict(session_timeout_seconds=hard,budget_seconds=cfg['budget_seconds'],
                                         watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(folder/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}))
    return target,cfg


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gpu-smoke',action='store_true'); a=p.parse_args(); print(build(a.gpu_smoke)[0])
