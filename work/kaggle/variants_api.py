"""friend1 only: pinned weight reads, exactly one private push, read-only watch/output.

SDK authenticates without inspecting credential files. No cancellation/deletion.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import os
import tempfile
from pathlib import Path
from variants_common import *

def now(): return datetime.now(timezone.utc).isoformat()
def connect():
    os.environ['KAGGLE_CONFIG_DIR'] = str(Path('~/.kaggle-friend').expanduser())
    for key in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'): os.environ.pop(key, None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
    return api

def metadata(api, ref=KERNEL, version=None):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    req=ApiGetKernelRequest(); req.user_name,req.kernel_slug=ref.split('/')
    if version is not None: req.version_label='v'+str(version)
    with api.build_kaggle_client() as client: return client.kernels.kernels_api_client.get_kernel(req)

def codes(nb): return [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
def parity(remote):
    m=remote.metadata
    assert m.is_private and m.enable_gpu and not m.enable_internet and not m.enable_tpu
    assert m.machine_shape == 'NvidiaTeslaT4'
    assert set(m.kernel_data_sources)==set(SOURCES) and not m.dataset_data_sources
    assert codes(json.loads(remote.blob.source))==codes(read(BUNDLE/'variants.ipynb'))
    return dict(private=True, exact_code_parity=True, current_version=m.current_version_number,
                kernel_sources=m.kernel_data_sources, machine_shape=m.machine_shape)

def sources(api):
    from b1_output import download
    receipts=[]
    for source in SOURCES:
        owner,slug,version=source.split('/')
        remote=metadata(api,owner+'/'+slug,version)
        assert remote.metadata.is_private and remote.metadata.current_version_number == int(version)
        folder=STAGE/'source-audit'/slug; folder.mkdir(parents=True,exist_ok=True)
        for m in [x for x in models().values() if x['source_ref']==source]:
            checkpoint=m['source_path'].split('/weights/')[0]+'/points/'+('long-truncation' if m['name']=='pilot-b' else 'b6-final')+'/checkpoint.json'
            point=download(api,source,checkpoint,folder)
            p=read(point)
            assert p['sha256']==m['sha256'] and p['optimizer_included'] is False and p['dtype']=='FP16'
            target=WORK/m['local_path']
            if not target.is_file(): download(api,source,m['source_path'],target.parents[2])
            assert target.is_file() and sha(target)==m['sha256'] and target.stat().st_size==p['bytes']
            receipts.append(dict(name=m['name'],source=source,source_path=m['source_path'],
                sha256=m['sha256'],checkpoint_sha256=sha(point),bytes=target.stat().st_size))
    write(WORK/'checks/final-variants-source-weights.json',dict(passed=True,private=True,models=receipts,at_utc=now()))
    print(json.dumps(dict(passed=True,models=len(receipts),account='friend1')))

def push(api):
    assert not LEDGER.exists(), 'Exactly one push: durable ledger exists; observe only'
    assert read(WORK/'checks/final-variants-cpu.json')['passed']
    assert read(WORK/'checks/final-variants-source-weights.json')['passed']
    identity=read(WORK/'checks/final-variants-build.json')
    assert identity['notebook_sha256']==sha(BUNDLE/'variants.ipynb')
    assert read(WORK/'checks/final-variants-cpu.json')['notebook_sha256']==identity['notebook_sha256']
    assert (BUNDLE/'variants.ipynb').stat().st_size<900_000
    owned=[]
    for page in range(1,101):
        rows=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.extend(x.ref for x in rows)
        if not rows: break
    else: raise AssertionError('Incomplete owned inventory')
    assert owned and all(x.startswith('friend1/') for x in owned), 'Expected friend account'
    assert KERNEL not in owned, 'Kernel exists; inspect instead of pushing'
    with api.build_kaggle_client() as client:
        q=client.kernels.kernels_api_client.get_accelerator_quota_statistics().gpu_quota
    free=(q.total_time_allowed-q.time_used-q.time_reserved).total_seconds()
    assert free>=7200, 'Need quota for backend T4 allocation and 3600s cap'
    receipt=dict(ref=KERNEL,push_calls=1,requested_at_utc=now(),private=True,
        notebook_sha256=identity['notebook_sha256'],free_quota_seconds=free,timeout_seconds=3600)
    write(LEDGER,receipt)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result=api.kernels_push(str(BUNDLE),timeout='3600',acc='NvidiaTeslaT4')
    assert not result.error, 'Push failed/uncertain: do not retry'
    receipt.update(version=result.version_number,accepted=True,finished_at_utc=now());write(LEDGER,receipt)
    proof=parity(metadata(api,version=result.version_number))
    write(WORK/'checks/final-variants-remote.json',dict(ref=KERNEL,checked_at_utc=now(),**proof))
    print(json.dumps(receipt))

def latest(api):
    launch=read(LEDGER); remote=metadata(api); proof=parity(remote)
    version=proof['current_version']; assert version>=launch.get('version',0)
    parity(metadata(api,version=version))
    return version,proof

def observe(api):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest,ApiGetKernelSessionStatusRequest
    version,proof=latest(api)
    req=ApiGetKernelSessionStatusRequest(); req.user_name,req.kernel_slug=KERNEL.split('/');req.version_label='v'+str(version)
    outreq=ApiListKernelSessionOutputRequest();outreq.user_name,outreq.kernel_slug=KERNEL.split('/');outreq.version_label='v'+str(version);outreq.page_size=1
    with api.build_kaggle_client() as client:
        status=client.kernels.kernels_api_client.get_kernel_session_status(req).status.name
        out=client.kernels.kernels_api_client.list_kernel_session_output(outreq)
    path=WORK/'checks/final-variants-observations.json'; history=read(path) if path.exists() else []
    matching=[x for x in history if x['status']==status and x['version']==version]
    seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(matching[0]['at_utc'])).total_seconds() if matching else 0
    record=dict(at_utc=now(),status=status,version=version,own_push_calls=1,
        output_visible=bool(out.files),log_chars=len(out.log or ''),status_observed_seconds=seconds,
        wait_threshold_crossed=(status=='QUEUED' and seconds>600) or (status=='RUNNING' and seconds>720 and not out.files),**proof)
    history.append(record);write(path,history);print(json.dumps(record))

def output(api):
    from kaggle_versioned import output as download
    version,proof=latest(api)
    target=WORK/'results/kaggle-runs/nlp-hw02-variants'/str(version)
    # The helper pins calls to the exact version; no latest-output race.
    build = api.build_kaggle_client
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            download(api,KERNEL+'/'+str(version),target)
    finally:
        api.build_kaggle_client = build
    assert latest(api)[0]==version, 'Version changed during output download'
    write(WORK/'checks/final-variants-download.json',dict(ref=KERNEL,version=version,
        output=str(target.relative_to(WORK)),downloaded_at_utc=now(),**proof))
    print(json.dumps(dict(version=version,output=str(target.relative_to(WORK)))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['sources','push','observe','output']);args=p.parse_args()
    try: globals()[args.action](connect())
    except Exception as e:
        print(json.dumps(dict(error_type=type(e).__name__,http=getattr(getattr(e,'response',None),'status_code',None),
            reason=str(e) if isinstance(e,AssertionError) else 'Operation failed; raw response suppressed')))
        raise SystemExit(1)
