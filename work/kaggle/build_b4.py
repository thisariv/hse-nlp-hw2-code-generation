"""Build B4 and <=12min smoke; local only. No GPU calls or pushes."""
import argparse,hashlib,json,shutil
from build_l3 import ROOT,literals,write


def build(smoke=False):
    pins=json.loads((ROOT/'b4-anchor.json').read_text());pin=pins['anchor']
    old=json.loads((ROOT/'notebooks/nlp-hw02-b3.ipynb').read_text())
    codes=[c['source'] for c in old['cells'] if c['cell_type']=='code'];literal=literals(codes[0])
    sources={n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in literal['SOURCES']}
    for p in (ROOT/'src').glob('b4_*.py'):sources[p.name]=p.read_text()
    for n in ('p1_math.py','b4_worker.py'):sources[n]=(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text()
    for n,h in pin['identity']['recipe']['implementation_sha256'].items():
        assert hashlib.sha256(sources[n].encode()).hexdigest()==h,n
    cfg=dict(literal['CONFIG'],session_kind='b4',anchors={'0':pin,'1':pin},gpu_smoke=smoke,
             resume_source=pins['source'],source_manifest_sha256=pins['source_manifest_sha256'],
             probe_main_marks=[pin['main_seconds']+10800.,pin['main_seconds']+21600.],
             truncation_q_by_gpu=[0.,.1],main_seconds=32400.,budget_seconds=40500,
             worker_deadline_seconds=40500,hard_shutdown_seconds=40740,session_ceiling_seconds=40800,
             setup_cap_seconds=480.,io_reserve_seconds=240.,final_q=.1,
             comparison_protocol=dict(status='same B3 W600 full state/order/cursor/RNG; q0 vs length-stratified q10%; equal-wall endpoints, matched-prefix analysis'))
    cfg.pop('transfer_sha256',None)
    if smoke:cfg.update(budget_seconds=660,worker_deadline_seconds=660,hard_shutdown_seconds=690,
                        session_ceiling_seconds=720,setup_cap_seconds=300.,io_reserve_seconds=20.,smoke_updates=200)
    cfg['port_sha256']=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    init=codes[0];begin=init.index('SOURCES=');end=init.index('\nfor name,source in SOURCES.items()')
    init=init[:begin]+'SOURCES='+repr(sources)+init[end:]
    init=init[:init.index('CONFIG=')]+'CONFIG='+repr(cfg)+'\n'+init[init.index('WORKING=Path'):]
    init=init.replace('B3 T4','B4 T4').replace('B2 resume runtime changed','B3 resume runtime changed')
    setup=codes[-1];begin=setup.index('from b1_resume import materialize');end=setup.index("(WORKING/'weighted-order').mkdir")
    setup=setup[:begin]+'''from b4_resume import verify_mount
source=resolve_cache_mount(inp,CONFIG['resume_source'],'notebooks')
recovered=verify_mount(source,CONFIG['anchors']['0'],CONFIG['source_manifest_sha256'])
CONFIG['resume_root']=str(recovered)
'''+setup[end:]
    setup=setup.replace('from b1_session import launch','from b4_session import launch')
    setup=setup.replace('B3 requires B2 states','B4 requires B3 full state').replace('B3 setup','B4 setup')
    setup=setup.replace("MODULE_DIR/'b3_worker.py'","MODULE_DIR/'b4_worker.py'").replace('b3-publication.json','b4-publication.json')
    cells=[]
    for kind,source in [('markdown','# B4: W600 / W600-trunc\n\nBoth GPUs resume the same uncooled B3 W600. '+
                         ('Reviewer smoke <=12min.' if smoke else 'Main ~9h; +3/+6h probes; final50min linear; hard11h20min.')+
                         ' Private thisarivson, T4x2, internet on; Secrets-free. Launch by reviewer after review.'),
                        ('code',init),('code',codes[1]),('code',setup)]:
        cell=dict(id=f'b4-{len(cells):02d}',cell_type=kind,metadata={},source=source)
        if kind=='code':compile(source,'B4 cell','exec');cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    slug='nlp-hw02-b4-gpu-smoke' if smoke else 'nlp-hw02-b4'
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,dict(old,cells=cells))
    folder=ROOT/'bundles'/('b4_gpu_smoke_thisarivson' if smoke else 'b4_thisarivson');folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',dict(id='thisarivson/'+slug,title=slug,code_file=target.name,language='python',
          kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
          dataset_sources=['thisarivson/nlp-hw02-26b-inputs'],
          kernel_sources=['thisarivson/nlp-hw02-b3','thisarivson/nlp-hw02-compact-cache-verified/2'],competition_sources=[]))
    write(folder/'session-limits.json',dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
          budget_seconds=cfg['budget_seconds'],watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(folder/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}))
    return target,cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',action='store_true');a=p.parse_args();print(build(a.gpu_smoke)[0])
