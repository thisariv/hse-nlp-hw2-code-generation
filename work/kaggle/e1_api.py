"""E1: one durable push; read-only, code-checked monitoring/download thereafter.

Only the Kaggle SDK/CLI accesses authentication. No credential inspection,
raw HTTP output, signed URLs, cancellation, deletion or automatic retry.
"""
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent; WORK=ROOT.parent
BUNDLE=ROOT/'bundles/e1_mine'; REF='friend1/nlp-hw02-e1-ensembles'
LEDGER=WORK/'checks/e1-launch.json'

def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prereg(): return (WORK/'notes/e1-ensembles.md').read_text().split('## CPU-проверки')[0]


def main(action):
    env=dict(os.environ,KAGGLE_CONFIG_DIR=str(Path('~/.kaggle-friend').expanduser()))
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):env.pop(k,None)
    os.environ.clear();os.environ.update(env)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
    def metadata(version=None,ref=REF):
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=ref.split('/')
        if version:req.version_label='v'+str(version)
        with api.build_kaggle_client()as client:return client.kernels.kernels_api_client.get_kernel(req)
    def quota():
        with api.build_kaggle_client()as client:q=client.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        return dict(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),
            used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),reset_utc=str(q.quota_refresh_time))
    def parity(remote):
        m=remote.metadata;local=read(BUNDLE/'kernel-metadata.json')
        assert m.is_private and m.enable_gpu and not m.enable_internet and not m.enable_tpu
        assert m.machine_shape=='NvidiaTeslaT4'
        def codes(nb):return [''.join(c['source'])if isinstance(c['source'],list)else c['source']for c in nb['cells']if c['cell_type']=='code']
        assert codes(json.loads(remote.blob.source))==codes(read(BUNDLE/local['code_file'])),'Remote code differs'
        assert set(m.dataset_data_sources)==set(local['dataset_sources'])
        assert {'/'.join(s.split('/')[:2])for s in m.kernel_data_sources}=={'/'.join(s.split('/')[:2])for s in local['kernel_sources']}
        return dict(private=True,internet=False,T4_pair=True,all_code_cells_exact=True,
                    current_version=m.current_version_number,last_run_time_utc=str(m.last_run_time),
                    dataset_sources=m.dataset_data_sources,kernel_sources=m.kernel_data_sources)
    if action=='quota':print(json.dumps(quota()));return
    if action=='push':
        assert not LEDGER.exists(),'E1 ledger exists: never repeat push'
        assert read(WORK/'checks/e1-cpu.json')['passed']
        identity=read(BUNDLE/'source-identity.json');meta=read(BUNDLE/'kernel-metadata.json')
        assert read(WORK/'checks/e1-cpu.json')['notebook_sha256']==identity['notebook_sha256'],'CPU receipt does not bind this notebook'
        assert meta['id']==REF and meta['is_private']and not meta['enable_internet']
        assert sha(BUNDLE/meta['code_file'])==identity['notebook_sha256']
        # Verify dependency versions are accessible without downloading any weights.
        dependencies={ref:metadata(int(v),ref).metadata.current_version_number for ref,v in
                      [('friend1/nlp-hw02-b6','1'),('friend1/nlp-hw02-kaggle-p1','3')]}
        inventory=[];complete=False
        for page in range(1,101):
            entries=api.kernels_list(mine=True,page=page,page_size=100)or[]
            inventory.extend(x.ref for x in entries)
            if not entries:complete=True;break
        remote=None
        try:remote=metadata()
        except Exception as error:
            assert getattr(getattr(error,'response',None),'status_code',None)in(403,404),'Cannot establish target state'
        if remote is None:
            assert complete and REF not in inventory
            preflight=dict(target_absent_in_full_owned_inventory=True,owned_count=len(inventory))
        else:
            status=api.kernels_status(REF).status.name
            assert status in ('COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'),'Active target: no push'
            preflight=dict(status=status,version=remote.metadata.current_version_number)
        q=quota();assert q['free_seconds']>=8520,'Insufficient pair quota for4200s run'
        rec=dict(ref=REF,requested_at_utc=now(),own_push_calls=1,private=True,preflight=preflight,
            dependencies=dependencies,quota_before=q,timeout_seconds=4200,maximum_device_seconds=8400,
            notebook_sha256=identity['notebook_sha256'],notes_sha256=sha(WORK/'notes/e1-ensembles.md'),
            preregistration_sha256=hashlib.sha256(prereg().encode()).hexdigest(),
            grid_sha256=sha(WORK/'research/e1_policies.py'),timing_sha256=sha(ROOT/'e1-timing-reference.json'))
        # The ledger is committed before the sole network mutation; uncertain response is never retried.
        write(LEDGER,rec)
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(BUNDLE),
            '--accelerator','NvidiaTeslaT4','--timeout','4200'],env=env,capture_output=True,text=True,timeout=150)
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout)
        rec.update(CLI_returncode=run.returncode,push_accepted=run.returncode==0 and bool(match),push_finished_at_utc=now())
        if match:rec['version']=int(match.group(1))
        write(LEDGER,rec)
        assert rec['push_accepted'],'Push failed/uncertain: do not retry'
        rec['server_parity']=parity(metadata(rec['version']));write(LEDGER,rec)
        print(json.dumps(rec));return
    launch=read(LEDGER);remote=metadata();proof=parity(remote)
    version=remote.metadata.current_version_number
    assert version>=launch.get('monitor_version',launch['version'])
    status=api.kernels_status(REF).status.name
    if action=='recover-sources':
        assert status=='COMPLETE'
        nb=json.loads(remote.blob.source)
        code=next(c['source']for c in nb['cells']if c['cell_type']=='code')
        code=''.join(code)if isinstance(code,list)else code
        sources=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name)and t.id=='SOURCES'for t in n.targets)))
        identity=read(ROOT/'e1-source-identity.json');recovered=[]
        dest=WORK/'results/kaggle-runs/nlp-hw02-e1-ensembles'/str(version)/'e1-src'
        dest.mkdir(parents=True,exist_ok=True)
        for name,digest in identity['source_sha256'].items():
            assert hashlib.sha256(sources[name].encode()).hexdigest()==digest
            path=dest/name
            if not path.exists():path.write_text(sources[name]);recovered.append(dict(file=name,sha256=digest))
            assert sha(path)==digest
        assert metadata().metadata.current_version_number==version
        rec=dict(at_utc=now(),version=version,all_code_cells_exact=True,
            recovered=recovered,origin='SOURCES literal from exact server notebook; candidate journals untouched')
        write(WORK/'checks/e1-source-recovery.json',rec);print(json.dumps(rec));return
    if action=='history':
        from kaggle_versioned import pin_client
        original=api.build_kaggle_client;rows=[]
        for v in sorted({launch['version'],version,*launch.get('external_versions',[])}):
            proof_v=parity(metadata(v))
            try:
                pin_client(api,REF+'/'+str(v))
                state=api.kernels_status(REF).status.name
            finally:api.build_kaggle_client=original
            rows.append(dict(requested_version=v,status=state,all_code_cells_exact=proof_v['all_code_cells_exact']))
        rec=dict(at_utc=now(),latest_version=version,own_push_calls=launch['own_push_calls'],versions=rows)
        write(WORK/'checks/e1-versions.json',rec);print(json.dumps(rec));return
    if action=='observe':
        req=ApiListKernelSessionOutputRequest();req.user_name='friend1';req.kernel_slug=REF.split('/')[1];req.version_label='v'+str(version);req.page_size=1
        with api.build_kaggle_client()as client:out=client.kernels.kernels_api_client.list_kernel_session_output(req)
        visible=bool(out.files);chars=len(out.log or '')
        if not visible:
            req.version_label=''
            with api.build_kaggle_client()as client:out=client.kernels.kernels_api_client.list_kernel_session_output(req)
            assert metadata().metadata.current_version_number==version,'Version changed; observe again'
            visible=bool(out.files);chars=max(chars,len(out.log or ''))
        path=WORK/'checks/e1-observations.json';history=read(path)if path.exists()else[]
        first=next((r['at_utc']for r in history if r['version']==version and r['status']==status),now())
        seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(first)).total_seconds()
        wait=(status=='QUEUED'and seconds>900)or(status=='RUNNING'and seconds>900 and not visible)
        rec=dict(at_utc=now(),version=version,status=status,output_visible=visible,log_chars=chars,
                 status_seconds=seconds,wait_threshold_crossed=wait,own_push_calls=launch['own_push_calls'],quota=quota(),**proof)
        history.append(rec);write(path,history)
        launch.update(monitor_version=version,status=status,last_observation=rec)
        launch['external_versions']=sorted({r['version']for r in history if r['version']!=launch['version']})
        launch['observed_version_statuses']={str(r['version']):r['status']for r in history}
        if wait:launch.setdefault('wait_threshold_crossed_at_utc',rec['at_utc'])
        write(LEDGER,launch)
        handoff=WORK/'results/next-e1.json'
        if handoff.exists():
            hand=read(handoff);hand.update(latest_version=version,status=status,external_versions=launch['external_versions'],
                last_observed_at_utc=rec['at_utc'],exact_code_parity=True);write(handoff,hand)
        if not (WORK/'results/e1-confirm.json').exists():
            write(WORK/'results/e1-experiment.json',dict(status=status,monitor_version=version,
                own_push_calls=launch['own_push_calls'],exact_code_parity=True,updated_at_utc=rec['at_utc'],
                cpu_ready=True,analysis_pending=True,quality=None,decision='await complete pool; keep v10 meanwhile'))
        print(json.dumps(rec));return
    assert status in ('COMPLETE','ERROR'),'Only download finished output; no partial analysis'
    from e1_download import output
    dest=WORK/'results/kaggle-runs/nlp-hw02-e1-ensembles'/str(version)
    receipt=output(api,REF+'/'+str(version),dest)
    binding=receipt['binding']+'; source parity before/after'
    assert metadata().metadata.current_version_number==version
    parity(metadata())
    rec=dict(version=version,files=receipt['files'],output=str(dest.relative_to(WORK)),binding=binding,quota=quota())
    write(WORK/'checks/e1-download.json',rec);print(json.dumps(rec))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['quota','push','observe','history','download','recover-sources'])
    args=parser.parse_args()
    try:main(args.action)
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__,http=getattr(getattr(error,'response',None),'status_code',None),
            reason=str(error)if isinstance(error,AssertionError)else'API/CLI failed; raw response suppressed')))
        raise SystemExit(1)
