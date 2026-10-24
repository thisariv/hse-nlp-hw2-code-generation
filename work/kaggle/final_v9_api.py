"""Private v9 API: one persistent push ledger; read-only status/metadata/output.

Authentication is delegated to the Kaggle SDK/CLI. Never inspect credentials or
print raw HTTP responses, signed download URLs, environment values, or auth data.
"""
import argparse, contextlib, hashlib, io, json, os, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
BUNDLE=ROOT/'bundles/final_inference_v9_mine'
REF='friend1/nlp-hw02-ensemble-inference-v9'
LEDGER=WORK/'checks/final-inference-v9-launch.json'

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def env():
    e=dict(os.environ,KAGGLE_CONFIG_DIR=str(Path('~/.kaggle-friend').expanduser()))
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):e.pop(k,None)
    return e

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['push','status','metadata','output','quota','observe']);a=p.parse_args()
    os.environ.clear();os.environ.update(env_saved)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    def metadata(version=None):
        req=ApiGetKernelRequest();req.user_name='friend1';req.kernel_slug=REF.split('/')[1]
        if version:req.version_label='v'+str(version)
        with api.build_kaggle_client()as c:return c.kernels.kernels_api_client.get_kernel(req)
    def quota():
        with api.build_kaggle_client()as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        return dict(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),reset_utc=str(q.quota_refresh_time))
    def parity(remote):
        meta=remote.metadata;local=read(BUNDLE/'kernel-metadata.json')
        assert meta.is_private and meta.enable_gpu and not meta.enable_internet and not meta.enable_tpu
        assert meta.machine_shape=='NvidiaTeslaT4'
        nb=json.loads(remote.blob.source);expected=read(BUNDLE/local['code_file'])
        def codes(nb):return [''.join(c['source'])if isinstance(c['source'],list)else c['source']for c in nb['cells']if c['cell_type']=='code']
        assert codes(nb)==codes(expected),'Remote code differs'
        assert set(meta.dataset_data_sources)==set(local['dataset_sources'])
        assert {'/'.join(x.split('/')[:2])for x in meta.kernel_data_sources}=={'/'.join(x.split('/')[:2])for x in local['kernel_sources']}
        return dict(private=True,machine_shape=meta.machine_shape,current_version=meta.current_version_number,all_code_cells_exact=True,last_run_time_utc=str(meta.last_run_time),dataset_sources=meta.dataset_data_sources,kernel_sources=meta.kernel_data_sources)
    if a.action=='observe':
        remote=metadata();proof=parity(remote);version=remote.metadata.current_version_number
        status=api.kernels_status(REF).status.name;q=quota()
        launch=read(LEDGER);previous=launch.get('monitor_version',launch['version'])
        assert version>=previous,'Current version moved backwards; inspect manually'
        versions=launch.get('external_versions',[])
        if version!=launch['version'] and version not in versions:versions.append(version)
        states=launch.get('observed_version_statuses',{'1':'CANCEL_ACKNOWLEDGED','2':'CANCEL_ACKNOWLEDGED','3':'CANCEL_ACKNOWLEDGED'})
        states[str(version)]=status
        launch.update(monitor_version=version,external_versions=versions,observed_version_statuses=states,status=status,last_quota_observation=q)
        write(LEDGER,launch)
        handpath=WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v9.json'
        if handpath.exists():
            hand=read(handpath);hand.update(pushes=launch['push_calls'],own_push_calls=launch['push_calls'],own_pushed_version=launch['version'],monitor_version=version,status=status,external_versions=versions)
            write(handpath,hand)
        rec=dict(at_utc=now(),version=version,status=status,own_push_calls=launch['push_calls'],code_cells_exact=True,private=True,quota=q)
        path=WORK/'checks/final-inference-v9-observations.json';rows=read(path)if path.exists()else[];rows.append(rec);write(path,rows)
        path=WORK/'checks/final-inference-v9-status.json';rows=read(path);rows.append(dict(at_utc=rec['at_utc'],version=version,status=status));write(path,rows)
        write(WORK/'checks/final-inference-v9-remote.json',dict(checked_at_utc=rec['at_utc'],ref=REF,version=version,**proof))
        path=WORK/'results/final-inference-v9.json'
        partial=read(path)if path.exists()else{}
        if not partial.get('passed'):
            partial.update(status=status,monitor_version=version,observed_versions=states,updated_at_utc=rec['at_utc'],quota_after=q,
                observed_quota_used_delta_seconds=q['used_seconds']-launch['quota_before']['used_seconds'],reason='Waiting for output of reviewed code; no additional agent push.')
            partial.pop('final_observed_quota_used_delta_seconds',None);write(path,partial)
        print(json.dumps(rec));return
    if a.action=='quota':print(json.dumps(quota()));return
    if a.action=='push':
        assert not LEDGER.exists(),'Exactly one push: ledger already exists; inspect status'
        for mode in ['offline','weights','static','pipeline','regressions']:
            assert read(WORK/f'checks/final-inference-v9-{mode}.json')['passed']
        identity=read(BUNDLE/'source-identity.json');m=read(BUNDLE/'kernel-metadata.json');lim=read(BUNDLE/'session-limits.json')
        assert m['id']==REF and m['is_private'] and not m['enable_internet'] and m['machine_shape']=='NvidiaTeslaT4'
        assert lim['session_timeout_seconds']==720 and sha(BUNDLE/m['code_file'])==identity['notebook_sha256']
        inventory=[];fully=False
        for page in range(1,101):
            rows=api.kernels_list(mine=True,page=page,page_size=100)or[]
            inventory.extend(r.ref for r in rows)
            if not rows:fully=True;break
        remote=None
        try:remote=metadata()
        except Exception as e:
            assert getattr(getattr(e,'response',None),'status_code',None)in(403,404),'Cannot establish target state'
        if remote is None:
            assert fully and REF not in inventory,'Cannot establish target absence'
            preflight=dict(target_absent_in_complete_owned_inventory=True,owned_count=len(inventory),active_version=False)
        else:
            status=api.kernels_status(REF).status.name
            assert status in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'),'Server has active/uncertain target version'
            preflight=dict(server_status=status,server_version=remote.metadata.current_version_number,active_version=False)
        q=quota();assert q['free_seconds']>=1440+120
        rec=dict(ref=REF,requested_at_utc=now(),push_calls=1,private=True,preflight=preflight,quota_before=q,
            timeout_seconds=720,maximum_T4_pair_quota_seconds=1440,quota_budget_seconds=2400,notebook_sha256=identity['notebook_sha256'],push_started=True)
        # Written before network mutation; any uncertain response is never retried.
        write(LEDGER,rec)
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(BUNDLE),'--timeout','720','--accelerator','NvidiaTeslaT4'],env=env_saved,capture_output=True,text=True,timeout=150)
        rec['CLI_returncode']=run.returncode
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout)
        rec['push_accepted']=run.returncode==0 and bool(match)
        if match:rec['version']=int(match.group(1))
        rec['push_finished_at_utc']=now();write(LEDGER,rec)
        assert rec['push_accepted'],'Push response uncertain/failed; do not retry'
        print(json.dumps(rec));return
    launch=read(LEDGER);version=launch.get('monitor_version',launch['version'])
    if a.action=='metadata':
        rec=dict(checked_at_utc=now(),ref=REF,version=version,**parity(metadata(version)))
        write(WORK/'checks/final-inference-v9-remote.json',rec);print(json.dumps(rec));return
    if a.action=='status':
        from kaggle_versioned import pin_client
        pin_client(api,REF+'/'+str(version));status=api.kernels_status(REF).status.name
        path=WORK/'checks/final-inference-v9-status.json';hist=read(path)if path.exists()else[]
        rec=dict(at_utc=now(),version=version,status=status);hist.append(rec);write(path,hist);print(json.dumps(rec));return
    from kaggle_versioned import output
    current=metadata();parity(current);assert current.metadata.current_version_number==version
    dest=WORK/'results/kaggle-runs/nlp-hw02-ensemble-inference-v9'/str(version)
    original_build=api.build_kaggle_client
    output(api,REF+'/'+str(version),dest)
    receipt=read(dest/'download-version.json');binding='explicit version label + exact code cells and output source SHA'
    if not receipt['files']:
        # CLI 2.2.4 / Kaggle can return an empty versioned output listing.
        # Latest fallback is allowed only while the exact reviewed version is current.
        api.build_kaggle_client=original_build
        token=None;count=0
        while True:
            files,token=api.kernels_output(REF,str(dest),quiet=True,page_size=200,page_token=token)
            count+=len(files)
            if not token:break
        receipt['files']=count;binding='latest fallback, current version and full code parity verified before/after download'
    api.build_kaggle_client=original_build
    current=metadata();parity(current);assert current.metadata.current_version_number==version
    rec=dict(ref=REF,version=version,files=receipt['files'],version_binding=binding,output=str(dest.relative_to(WORK)),quota_after=quota())
    write(WORK/'checks/final-inference-v9-output-download.json',rec)
    print(json.dumps(rec))

if __name__=='__main__':
    env_saved=env()
    try:main()
    except Exception as e:
        print(json.dumps(dict(error_type=type(e).__name__,http=getattr(getattr(e,'response',None),'status_code',None),reason=str(e)if isinstance(e,AssertionError)else'API/CLI failed; raw response suppressed')));raise SystemExit(1)
