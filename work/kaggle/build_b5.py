"""Build local private friend1 B5 and <=12min smoke; never pushes a kernel."""
import argparse,hashlib,json,shutil
from build_l3 import ROOT,literals,write
from b5_sources import c_overrides


def build(smoke=False):
    pins=json.loads((ROOT/'b5-anchors.json').read_text())
    transfer=dict(version=pins['transfer_version'],transfer_sha256=pins['transfer_sha256'])
    old=json.loads((ROOT/'notebooks/nlp-hw02-b4.ipynb').read_text())
    codes=[c['source'] for c in old['cells'] if c['cell_type']=='code'];v=literals(codes[0])
    sources={n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in v['SOURCES']}
    for p in (ROOT/'src').glob('b5_*.py'):sources[p.name]=p.read_text()
    sources['b5_worker.py']=(ROOT/'b5_worker.py').read_text()
    overrides=c_overrides()
    for gpu,pin in pins['anchors'].items():
        for n,h in pin['identity']['recipe']['implementation_sha256'].items():
            actual=overrides.get(n,sources[n]) if gpu=='1' else sources[n]
            assert hashlib.sha256(actual.encode()).hexdigest()==h,(gpu,n)
    cfg=dict(v['CONFIG'],owner='friend1',session_kind='b5',anchors=pins['anchors'],gpu_smoke=smoke,
             resume_sources=pins['sources'],transfer_source=pins['transfer_dataset']+'/'+str(transfer['version']),
             transfer_sha256=transfer['transfer_sha256'],C_weighted_order=pins['C_weighted_order'],
             cache_source=pins['cache_source'],main_seconds=33840.,probe_offset_main_seconds=16200.,
             probe_cooldown_seconds=900.,probe_eval_seconds=300.,final_cooldown_seconds=3000.,final_eval_seconds=1800.,
             budget_seconds=40500,worker_deadline_seconds=40500,hard_shutdown_seconds=40740,session_ceiling_seconds=40800,
             setup_cap_seconds=480.,io_reserve_seconds=240.,checkpoint_seconds=900.,live_sync_enabled=False,final_q=0.,
             comparison_protocol=dict(status='independent historical W and C states/orders/cursors; equal added main-time cap; no shared-start invariant'))
    for key in ('resume_source','source_manifest_sha256','probe_main_marks','truncation_q_by_gpu','shape_bench_lengths'):
        cfg.pop(key,None)
    if smoke:cfg.update(budget_seconds=660,worker_deadline_seconds=660,hard_shutdown_seconds=690,
                        session_ceiling_seconds=720,setup_cap_seconds=300.,io_reserve_seconds=20.,smoke_updates=200)
    cfg['port_sha256']=hashlib.sha256(json.dumps(dict(SOURCES=sources,C_OVERRIDES=overrides),sort_keys=True).encode()).hexdigest()
    init=codes[0];a=init.index('SOURCES=');b=init.index('\nfor name,source in SOURCES.items()')
    init=init[:a]+'SOURCES='+repr(sources)+init[b:]
    after="for name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)"
    init=init.replace(after,after+"\nC_OVERRIDES="+repr(overrides)+'''
MODULE_C_DIR=WORKING/'src-C';MODULE_C_DIR.mkdir(exist_ok=True)
for name in SOURCES:
    target=MODULE_C_DIR/name
    if name in C_OVERRIDES:target.write_text(C_OVERRIDES[name])
    else:target.write_text(SOURCES[name])
''')
    init=init[:init.index('CONFIG=')]+'CONFIG='+repr(cfg)+'\n'+init[init.index('WORKING=Path'):]
    init=init.replace('nlp-b1-','nlp-b5-').replace('B4 T4','B5 T4').replace('B3 resume runtime changed','B5 resume runtime changed')
    setup=codes[-1];a=setup.index("assert CONFIG['resume_source']");b=setup.index("write(WORKING/'input-preflight.json'",a)
    setup=setup[:a]+'''from b5_resume import transfer_mount,c_mount,verify_order
from l3_resume import restore_order
transfer_source=resolve_cache_mount(inp,CONFIG['transfer_source'],'datasets')
recovered=transfer_mount(transfer_source,scratch/'W-inputs',CONFIG['transfer_sha256'],CONFIG['anchors']['0'])
live=resolve_cache_mount(inp,CONFIG['resume_sources']['1'],'datasets')
croot=c_mount(live,scratch/'C-inputs',CONFIG['anchors']['1'])
CONFIG['resume_roots']={'0':str(recovered/'model-GPU0'),'1':str(croot)}
for name in ('order.npy','order.json'):
    dst=WORKING/'weighted-order-GPU0'/name;dst.parent.mkdir(exist_ok=True)
    shutil.copyfile(recovered/'weighted-order'/name,dst)
restore_order(live,WORKING/'weighted-order-GPU1',CONFIG['C_weighted_order'])
for gpu in ('0','1'):
    verify_order(WORKING/f'weighted-order-GPU{gpu}',validated,CONFIG['anchors'][gpu])
'''+setup[b:]
    setup=setup.replace('resume_mount=str(source)','W_transfer_mount=str(transfer_source),C_live_mount=str(live)')
    setup=setup.replace('from b4_session import launch','from b1_session import launch')
    setup=setup.replace("MODULE_DIR/'b4_worker.py'","MODULE_DIR/'b5_worker.py'").replace('b4-publication.json','b5-publication.json').replace('B4 setup','B5 setup')
    setup=setup.replace('from b1_data import build_order,verify_order','')
    cells=[]
    for kind,s in [('markdown','# B5: W600 + C-long2x\n\nStrict independent continuation; no truncation/warmup. '+
                   ('Reviewer smoke <=12min.' if smoke else 'Main cap9.4h; one +4.5h probe; final50min linear; hard11h20min.')+
                   ' Private friend1, T4x2, internet on, Secrets-free. GPU launch by independent reviewer.'),
                   ('code',init),('code',codes[1]),('code',setup)]:
        cell=dict(id=f'b5-{len(cells):02d}',cell_type=kind,metadata={},source=s)
        if kind=='code':compile(s,'B5 notebook','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    slug='nlp-hw02-b5-gpu-smoke' if smoke else 'nlp-hw02-b5'
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,dict(old,cells=cells))
    folder=ROOT/'bundles'/('b5_gpu_smoke_friend1' if smoke else 'b5_friend1');folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',dict(id='friend1/'+slug,title=slug,code_file=target.name,language='python',
          kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
          dataset_sources=[pins['starter_source'],cfg['transfer_source'],pins['sources']['1']],
          kernel_sources=[pins['cache_source']],competition_sources=[]))
    write(folder/'session-limits.json',dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
          budget_seconds=cfg['budget_seconds'],watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(folder/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
          C_overrides={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()}))
    return target,cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',action='store_true');a=p.parse_args();print(build(a.gpu_smoke)[0])
