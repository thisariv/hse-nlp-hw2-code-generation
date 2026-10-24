"""Frozen B1 implementation hashes for strict restore, plus EV-only worker. No training main."""
import hashlib
import json
import shutil
from build_l3 import ROOT,literals,write
from build import INSTALL


def build(receipt=None):
    receipt=receipt or json.loads((ROOT.parent/'checks/b1-ev-transfer.json').read_text())
    assert receipt['passed'] and receipt['remote_SHA_verified'] and receipt['dataset']=='thisarivson/nlp-hw02-b1-ev-inputs'
    old=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-b1.ipynb').read_text())
    original=literals(next(c['source'] for c in old['cells'] if c['cell_type']=='code'))
    sources=original['SOURCES'].copy()
    for name in ('b1_ev.py',):sources[name]=(ROOT/'src'/name).read_text()
    sources['b1_ev_worker.py']=(ROOT/'b1_ev_worker.py').read_text()
    cfg=dict(original['CONFIG'],owner='thisarivson',session_kind='b1_ev',gpu_smoke=False,
             resume_source=receipt['dataset'],transfer_sha256=receipt['transfer_sha256'],
             budget_seconds=2640,worker_deadline_seconds=2640,hard_shutdown_seconds=2680,session_ceiling_seconds=2700,
             setup_cap_seconds=240,io_reserve_seconds=30,cooldown_seconds=900.,eval_seconds=700.)
    cfg["port_sha256"]=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    cells=[]
    def cell(kind,source):
        row=dict(id=f'ev-{len(cells):02d}',cell_type=kind,metadata={},source=source)
        if kind=='code':compile(source,'EV cell','exec');row.update(outputs=[],execution_count=None)
        cells.append(row)
    cell('markdown','# B1 EV: E/W @28370 и @34200\n\nPrivate thisarivson, T4×2, Secrets-free. Готовые FP16 @28370; @34200 только C timed linear cooldown900с и greedy heldout4k+old991+CE. Main training0. Hard wall45мин.')
    init=next(c['source'] for c in old['cells'] if c['cell_type']=='code')
    start=init[:init.index('CONFIG=')]
    tail=init[init.index("WORKING=Path"):]
    tail=tail[:tail.index('SOURCES=')]+'SOURCES='+repr(sources)+tail[tail.index('\nfor name,source in SOURCES.items()'):]
    cell('code',start+'CONFIG='+repr(cfg)+'\n'+tail)
    cell('code',INSTALL)
    setup=[c['source'] for c in old['cells'] if c['cell_type']=='code'][-1]
    setup=setup[:setup.index('from b1_session import launch')]+'''from b1_session import launch
report=launch(CONFIG,MODULE_DIR/'b1_ev_worker.py',WATCHDOG)
print(json.dumps(report,indent=2))
assert report['status']=='complete' and report['exit_codes']==[0,0]
'''
    cell('code',setup)
    slug='nlp-hw02-b1-ev';target=ROOT/'notebooks'/f'{slug}.ipynb'
    nb=dict(old,cells=cells);write(target,nb)
    folder=ROOT/'bundles/b1_ev';folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',dict(id='thisarivson/'+slug,title='NLP HW02 B1 EV',code_file=target.name,
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,enable_internet=True,
        machine_shape='NvidiaTeslaT4',dataset_sources=['thisarivson/nlp-hw02-26b-inputs',receipt['dataset']],
        kernel_sources=[cfg['cache_source']],competition_sources=[]))
    write(folder/'session-limits.json',dict(session_timeout_seconds=2700,budget_seconds=2640,watchdog_exit_seconds=2680))
    write(folder/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}))
    return target,cfg

if __name__=='__main__':print(build()[0])
