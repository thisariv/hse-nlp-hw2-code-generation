"""Generate private B6 + <=12min smoke without API mutations or GPU launches."""
import argparse,hashlib,json,shutil
from build_l3 import ROOT,literals,write
from b5_sources import c_overrides


def build(smoke=False):
    pins=json.loads((ROOT/'b6-anchors.json').read_text())
    nb=json.loads((ROOT/'notebooks/nlp-hw02-b5.ipynb').read_text())
    codes=[c['source'] for c in nb['cells'] if c['cell_type']=='code'];v=literals(codes[0])
    sources={n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in v['SOURCES']}
    for p in (ROOT/'src').glob('b6_*.py'):sources[p.name]=p.read_text()
    sources['p1_math.py']=(ROOT/'src/p1_math.py').read_text()
    sources['b6_worker.py']=(ROOT/'b6_worker.py').read_text()
    overrides=c_overrides()
    for g,pin in pins['anchors'].items():
        for n,h in pin['identity']['recipe']['implementation_sha256'].items():
            actual=overrides.get(n,sources[n]) if g=='1' else sources[n]
            assert hashlib.sha256(actual.encode()).hexdigest()==h,(g,n)
    cfg=dict(v['CONFIG'],owner='friend1',session_kind='b6',anchors=pins['anchors'],b6_pins=pins,
        resume_source=pins['source'],cache_source=pins['cache_source'],gpu_smoke=smoke,
        N_by_gpu={g:b['N'] for g,b in pins['budget'].items()},main_seconds=0.,final_q=0.,
        final_eval_seconds=1800.,checkpoint_seconds=900.,io_reserve_seconds=600.,setup_cap_seconds=480.,
        budget_seconds=18900,worker_deadline_seconds=18900,hard_shutdown_seconds=19140,session_ceiling_seconds=19200,
        live_sync_enabled=False,save_midpoint=True,
        comparison_protocol=dict(status='B6 independent B5 end-main states; q0 fixed-N 1-sqrt cooldown only; compare B5 final50 offline'))
    for k in ('resume_sources','transfer_source','transfer_sha256','C_weighted_order','probe_offset_main_seconds',
              'probe_cooldown_seconds','probe_eval_seconds','final_cooldown_seconds'):
        cfg.pop(k,None)
    if smoke:cfg.update(N_by_gpu={'0':200,'1':200},budget_seconds=660,worker_deadline_seconds=660,
        hard_shutdown_seconds=690,session_ceiling_seconds=720,setup_cap_seconds=300.,
        final_eval_seconds=90.,io_reserve_seconds=20.,checkpoint_seconds=90.)
    cfg['port_sha256']=hashlib.sha256(json.dumps(dict(SOURCES=sources,C_OVERRIDES=overrides),sort_keys=True).encode()).hexdigest()
    init=codes[0];a=init.index('CONFIG=');b=init.index('WORKING=Path',a)
    init=init[:a]+'CONFIG='+repr(cfg)+'\n'+init[b:]
    a=init.index('SOURCES=');b=init.index('\nfor name,source in SOURCES.items()',a)
    init=init[:a]+'SOURCES='+repr(sources)+init[b:]
    a=init.index('C_OVERRIDES=');b=init.index('\nMODULE_C_DIR=',a)
    init=init[:a]+'C_OVERRIDES='+repr(overrides)+init[b:]
    init=init.replace('nlp-b5-','nlp-b6-').replace('B5 T4','B6 T4').replace('B5 resume runtime changed','B6 resume runtime changed')
    setup=codes[-1];a=setup.index('from b5_resume import');b=setup.index("write(WORKING/'input-preflight.json'",a)
    setup=setup[:a]+'''from b6_resume import mount
b5_source=resolve_cache_mount(inp,CONFIG['resume_source'],'notebooks')
CONFIG['resume_roots']=mount(b5_source,WORKING,validated,CONFIG['b6_pins'])
'''+setup[b:]
    setup=setup.replace('W_transfer_mount=str(transfer_source),C_live_mount=str(live)','B5_mount=str(b5_source)')
    setup=setup.replace("MODULE_DIR/'b5_worker.py'","MODULE_DIR/'b6_worker.py'").replace('b5-publication.json','b6-publication.json').replace('B5 setup','B6 setup')
    cells=[]
    for kind,s in [('markdown','# B6: длинный q0 cooldown W600 / C-long2x\n\n'+
        ('Smoke: fixed N200 on each GPU, mini greedy/CE/test/K8, full state + FP16. Hard12min.' if smoke else
         'Only fixed-N 1-sqrt cooldown from B5v6 uncooled states, no main. N16500/32900, hard5h20min.')+
        ' Private friend1, T4x2, internet on, no Secrets. Launch by independent reviewer.'),
        ('code',init),('code',codes[1]),('code',setup)]:
        cell=dict(id=f'b6-{len(cells):02d}',cell_type=kind,metadata={},source=s)
        if kind=='code':compile(s,'B6 notebook','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    slug='nlp-hw02-b6-gpu-smoke' if smoke else 'nlp-hw02-b6'
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,dict(nb,cells=cells))
    folder=ROOT/'bundles'/('b6_gpu_smoke_friend1' if smoke else 'b6_friend1');folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',dict(id='friend1/'+slug,title=slug,code_file=target.name,language='python',
        kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        machine_shape='NvidiaTeslaT4',dataset_sources=[pins['starter_source']],
        kernel_sources=[pins['cache_source'],pins['source']],competition_sources=[]))
    write(folder/'session-limits.json',dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
        budget_seconds=cfg['budget_seconds'],watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(folder/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
        C_overrides={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()},
        config_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()))
    return target,cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',action='store_true');a=p.parse_args();print(build(a.gpu_smoke)[0])
