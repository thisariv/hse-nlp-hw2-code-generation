"""Exactly one private push, default Kaggle auth; sanitized read-only monitoring."""
import argparse,contextlib,csv,hashlib,io,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;BUNDLE=ROOT/'bundles/tta_mine'
REF='thisarivson/nlp-hw02-tta';LEDGER=WORK/'checks/tta-launch.json'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def environment():
    e=dict(os.environ)
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN','KAGGLE_CONFIG_DIR'):e.pop(k,None)
    return e
def main(action):
    os.environ.clear();os.environ.update(environment_saved)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    def metadata():
        req=ApiGetKernelRequest();req.user_name='thisarivson';req.kernel_slug='nlp-hw02-tta'
        with api.build_kaggle_client()as c:return c.kernels.kernels_api_client.get_kernel(req)
    def parity(remote):
        m=remote.metadata;assert m.is_private and m.enable_gpu and not m.enable_internet and not m.enable_tpu and m.machine_shape=='NvidiaTeslaT4'
        nb=json.loads(remote.blob.source);local=read(BUNDLE/'nlp-hw02-tta.ipynb')
        def codes(n):return [''.join(c['source']) if isinstance(c['source'],list) else c['source']for c in n['cells']if c['cell_type']=='code']
        assert codes(nb)==codes(local),'Remote code differs'
        expected=read(BUNDLE/'kernel-metadata.json')
        assert set(m.dataset_data_sources)==set(expected['dataset_sources'])
        assert {'/'.join(x.split('/')[:2])for x in m.kernel_data_sources}==set(expected['kernel_sources'])
        return m.current_version_number
    def quota():
        with api.build_kaggle_client()as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        return dict(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),reset_utc=str(q.quota_refresh_time))
    if action=='quota':print(json.dumps(quota()));return
    if action=='push':
        assert not LEDGER.exists(),'Push already attempted; never retry'
        for mode in ['static','pipeline','rules']:assert read(WORK/f'checks/tta-{mode}.json')['passed']
        design=WORK/'notes/tta-experiment.md';assert 'ПРЕДРЕГИСТРАЦИЯ' in design.read_text()
        identity=read(BUNDLE/'source-identity.json');assert hashlib.sha256((BUNDLE/'nlp-hw02-tta.ipynb').read_bytes()).hexdigest()==identity['notebook_sha256']
        inventory=[];fully=False
        for page in range(1,101):
            rows=api.kernels_list(mine=True,page=page,page_size=100)or[];inventory.extend(r.ref for r in rows)
            if not rows:fully=True;break
        if REF not in inventory:
            assert fully;preflight=dict(target_absent_in_complete_owned_inventory=True,owned_count=len(inventory),active_version=False)
        else:
            remote=metadata();status=api.kernels_status(REF).status.name
            assert status in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'),'Target active or uncertain'
            preflight=dict(server_status=status,server_version=remote.metadata.current_version_number,active_version=False)
        q=quota();assert q['free_seconds']>=2400
        rec=dict(ref=REF,push_calls=1,requested_at_utc=now(),private=True,preflight=preflight,quota_before=q,timeout_seconds=1200,
                 maximum_reserved_pair_device_seconds=2400,notebook_sha256=identity['notebook_sha256'],preregistration_sha256=hashlib.sha256(design.read_bytes()).hexdigest())
        write(LEDGER,rec)
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(BUNDLE),'--timeout','1200','--accelerator','NvidiaTeslaT4'],env=environment_saved,capture_output=True,text=True,timeout=150)
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout)
        rec.update(CLI_returncode=run.returncode,push_accepted=run.returncode==0 and bool(match),push_finished_at_utc=now())
        if match:rec['version']=rec['monitor_version']=int(match.group(1))
        write(LEDGER,rec);assert rec['push_accepted'],'Push uncertain; no retry'
        print(json.dumps(rec));return
    launch=read(LEDGER);remote=metadata();version=parity(remote)
    assert version>=launch.get('monitor_version',launch['version'])
    status=api.kernels_status(REF).status.name
    if action=='observe':
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        request=ApiListKernelSessionOutputRequest();request.user_name='thisarivson';request.kernel_slug='nlp-hw02-tta';request.page_size=200
        with api.build_kaggle_client()as c:
            response=c.kernels.kernels_api_client.list_kernel_session_output(request)
        output_files=len(response.files or [])
        launch.update(monitor_version=version,status=status,last_observed_at_utc=now());write(LEDGER,launch)
        rec=dict(at_utc=now(),version=version,status=status,output_files=output_files,own_push_calls=launch['push_calls'],code_cells_exact=True,private=True,quota=quota())
        p=WORK/'checks/tta-observations.json';hist=read(p)if p.exists()else[];hist.append(rec);write(p,hist)
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(launch['requested_at_utc'])).total_seconds()
        handoff=dict(ref=REF,status=status,version=version,own_push_calls=launch['push_calls'],output_files=output_files,
            requested_at_utc=launch['requested_at_utc'],last_observed_at_utc=rec['at_utc'],elapsed_minutes=elapsed/60,
            notebook_sha256=launch['notebook_sha256'],preregistration_sha256=launch['preregistration_sha256'],
            bundle='kaggle/bundles/tta_mine',backend_timeout_seconds=1200,watchdog_seconds=1140,quota_budget_seconds=2400,
            external_orchestrator_only_command='env -u KAGGLE_USERNAME -u KAGGLE_KEY -u KAGGLE_API_TOKEN -u KAGGLE_CONFIG_DIR kaggle kernels push -p work/kaggle/bundles/tta_mine --timeout 1200 --accelerator NvidiaTeslaT4',
            agent_must_not_repush=True,read_only_monitor='kaggle/tta_api.py observe',download='kaggle/tta_api.py output',
            analyze_in_order=['research/tta-analysis.py tune --raw results/kaggle-runs/nlp-hw02-tta/VERSION','research/tta-analysis.py confirm --raw results/kaggle-runs/nlp-hw02-tta/VERSION'])
        write(WORK/'results/next-tta.json',handoff);print(json.dumps(rec));return
    from kaggle_versioned import output
    dest=WORK/f'results/kaggle-runs/nlp-hw02-tta/{version}'
    original_build=api.build_kaggle_client
    output(api,REF+'/'+str(version),dest)
    receipt=read(dest/'download-version.json');binding='explicit version label + exact source parity'
    api.build_kaggle_client=original_build
    if not receipt['files']:
        assert parity(metadata())==version
        files,token=api.kernels_output(REF,str(dest),quiet=True,page_size=200)
        assert not token
        receipt['files']=len(files);binding='latest fallback; exact current version/code checked before and after'
    assert parity(metadata())==version
    write(WORK/'checks/tta-download.json',dict(version=version,files=receipt['files'],version_binding=binding,output=str(dest.relative_to(WORK)),code_cells_exact=True,quota=quota()))
    print(json.dumps(dict(version=version,output=str(dest.relative_to(WORK)))))
if __name__=='__main__':
    environment_saved=environment()
    p=argparse.ArgumentParser();p.add_argument('action',choices=['push','observe','output','quota']);a=p.parse_args()
    try:main(a.action)
    except Exception as e:
        print(json.dumps(dict(error_type=type(e).__name__,http=getattr(getattr(e,'response',None),'status_code',None),reason=str(e)if isinstance(e,AssertionError)else'API/CLI failed; raw response suppressed')));raise SystemExit(1)
