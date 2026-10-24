"""E2: one durable push on thisarivson, then read-only parity/status/download.

Only the Kaggle SDK accesses authentication. No credential files are inspected.
The launch ledger precedes the sole push; uncertain responses are never retried.
"""
import argparse
import ast
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
BUNDLE=ROOT/'bundles/e2_mine';REF='thisarivson/nlp-hw02-e2-policy'
LEDGER=WORK/'checks/e2-launch.json'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main(action):
    env=dict(os.environ,KAGGLE_CONFIG_DIR=str(Path('~/.kaggle').expanduser()))
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):env.pop(k,None)
    os.environ.clear();os.environ.update(env)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest,ApiListKernelSessionOutputRequest
    def metadata(version=None):
        req=ApiGetKernelRequest();req.user_name,req.kernel_slug=REF.split('/')
        if version:req.version_label='v'+str(version)
        with api.build_kaggle_client() as c:return c.kernels.kernels_api_client.get_kernel(req)
    def quota():
        with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        return dict(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),
            used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds())
    def parity(remote):
        m=remote.metadata;local=read(BUNDLE/'kernel-metadata.json')
        assert m.is_private and m.enable_gpu and not m.enable_internet and not m.enable_tpu
        assert m.machine_shape=='NvidiaTeslaT4' and not m.kernel_data_sources
        def codes(nb):return [''.join(c['source']) if isinstance(c['source'],list) else c['source']
                             for c in nb['cells'] if c['cell_type']=='code']
        assert codes(json.loads(remote.blob.source))==codes(read(BUNDLE/local['code_file'])),'Server code differs'
        assert set(m.dataset_data_sources)==set(local['dataset_sources']),'Server datasets differ'
        return dict(current_version=m.current_version_number,all_code_cells_exact=True,
            private=True,internet=False,T4_pair=True,dataset_sources=m.dataset_data_sources,
            last_run_time_utc=str(m.last_run_time))
    if action=='quota':print(json.dumps(quota()));return
    if action=='push':
        assert not LEDGER.exists(),'Push already attempted; never repeat'
        identity=read(BUNDLE/'source-identity.json');meta=read(BUNDLE/'kernel-metadata.json')
        assert read(WORK/'checks/e2-pipeline.json')['passed']
        assert read(WORK/'checks/e2-pipeline.json')['notebook_sha256']==identity['notebook_sha256']
        assert read(WORK/'checks/e2-resume-verification.json')['passed']
        assert read(WORK/'checks/e2-b6-fp16-transfer.json')['roundtrip_sha256_passed']
        assert read(WORK/'checks/e2-phase2-design.json')['fresh_sampling_on_shared_tasks']
        assert sha(WORK/'results/e2-selection.json')==read(WORK/'checks/e2-selection.json')['selection_sha256']==identity['config']['selection_sha256']
        assert meta['id']==REF and sha(BUNDLE/meta['code_file'])==identity['notebook_sha256']
        assert meta['is_private'] and not meta['enable_internet']
        inventory=[]
        for page in range(1,101):
            rr=api.kernels_list(mine=True,page=page,page_size=100) or []
            inventory.extend(r.ref for r in rr)
            if not rr:break
        else:raise AssertionError('Owned kernel inventory incomplete')
        assert REF not in inventory,'Target already exists; verify state, never auto-push again'
        q=quota();assert q['free_seconds']>=3720,'Insufficient quota for bounded T4 pair'
        rec=dict(ref=REF,requested_at_utc=now(),own_push_calls=1,quota_before=q,
            target_absent_in_full_owned_inventory=True,timeout_seconds=1800,
            maximum_device_seconds=3600,notebook_sha256=identity['notebook_sha256'],
            selection_sha256=identity['config']['selection_sha256'],
            phase2_design_sha256=sha(WORK/'checks/e2-phase2-design.json'))
        write(LEDGER,rec)
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(BUNDLE),
             '--accelerator','NvidiaTeslaT4','--timeout','1800'],env=env,capture_output=True,text=True,timeout=150)
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout)
        rec.update(CLI_returncode=run.returncode,push_accepted=run.returncode==0 and bool(match),
                   push_finished_at_utc=now())
        if match:rec['version']=int(match[1])
        write(LEDGER,rec)
        assert rec['push_accepted'],'Push failed/uncertain; never retry'
        proof=parity(metadata(rec['version']))
        assert proof['current_version']==rec['version'],'Server version differs'
        rec['server_parity']=proof;write(LEDGER,rec);print(json.dumps(rec));return
    launch=read(LEDGER);remote=metadata();proof=parity(remote);version=proof['current_version']
    assert version>=launch.get('monitor_version',launch['version'])
    status=api.kernels_status(REF).status.name
    if action=='observe':
        req=ApiListKernelSessionOutputRequest();req.user_name,req.kernel_slug=REF.split('/')
        req.version_label='v'+str(version);req.page_size=1
        with api.build_kaggle_client() as c:out=c.kernels.kernels_api_client.list_kernel_session_output(req)
        visible=bool(out.files);chars=len(out.log or '')
        if not visible:
            req.version_label=''
            with api.build_kaggle_client() as c:out=c.kernels.kernels_api_client.list_kernel_session_output(req)
            assert parity(metadata())['current_version']==version,'Version changed; observe again'
            visible=bool(out.files);chars=max(chars,len(out.log or ''))
        path=WORK/'checks/e2-observations.json';history=read(path) if path.exists() else []
        at=now();previous=[r for r in history if r['current_version']==version]
        first=next((r['at_utc'] for r in reversed(previous) if r['status']!=status),None)
        if first is None:
            first=previous[0]['at_utc'] if previous else launch['requested_at_utc']
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(first)).total_seconds()
        wait=(status=='QUEUED' and elapsed>600) or (status=='RUNNING' and elapsed>720 and not visible)
        rec=dict(at_utc=at,status=status,output_visible=visible,log_chars=chars,status_seconds=elapsed,
                 wait_threshold_crossed=wait,own_push_calls=launch['own_push_calls'],quota=quota(),**proof)
        history.append(rec);write(path,history)
        launch.update(monitor_version=version,status=status,last_observation=rec,
                      external_versions=sorted({r['current_version'] for r in history if r['current_version']!=launch['version']}))
        if wait:launch.setdefault('wait_threshold_crossed_at_utc',at)
        write(LEDGER,launch)
        handoff=dict(rec,ref=REF,bundle='kaggle/bundles/e2_mine',
            notebook_sha256=launch['notebook_sha256'],selection_sha256=launch['selection_sha256'],
            requested_at_utc=launch['requested_at_utc'],agent_must_not_repush=True,
            monitor='kaggle/e2_api.py observe',download='kaggle/e2_api.py download',
            analysis_pending=not (WORK/'results/e2-confirm.json').exists())
        if handoff['analysis_pending']:
            handoff['external_restart_command']=('env -u KAGGLE_USERNAME -u KAGGLE_KEY -u KAGGLE_API_TOKEN '
                'KAGGLE_CONFIG_DIR="$HOME/.kaggle" kaggle kernels push -p kaggle/bundles/e2_mine '
                '--accelerator NvidiaTeslaT4 --timeout 1800')
        else:
            confirmed=read(WORK/'results/e2-confirm.json')
            handoff.update(completed=True,confirmation='results/e2-confirm.json',
                decision=confirmed['decision'],delta_pp=confirmed['delta_pp'],
                ci95_pp=confirmed['ci95_pp'],forecast_seconds=confirmed['forecast_seconds'],
                independent_tasks=confirmed['independent_tasks'])
        write(WORK/'results/next-e2.json',handoff)
        print(json.dumps(rec));return
    assert action=='download' and status in ('COMPLETE','ERROR'),'Download only finished output'
    from e1_download import output
    dest=WORK/'results/kaggle-runs/nlp-hw02-e2-policy'/str(version)
    result=output(api,REF+'/'+str(version),dest)
    assert parity(metadata())['current_version']==version,'Version changed during download'
    # Recover only notebook-embedded sources if Kaggle omits them from artifacts.
    nb=json.loads(remote.blob.source)
    init=next(c['source'] for c in nb['cells'] if c['cell_type']=='code')
    if isinstance(init,list):init=''.join(init)
    sources=ast.literal_eval(next(n.value for n in ast.parse(init).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets)))
    mods=dest/'e2-src';mods.mkdir(exist_ok=True)
    identity=read(BUNDLE/'source-identity.json')
    for name,digest in identity['source_sha256'].items():
        assert hashlib.sha256(sources[name].encode()).hexdigest()==digest
        p=mods/name
        if not p.exists():p.write_text(sources[name])
        assert sha(p)==digest
    rec=dict(at_utc=now(),version=version,status=status,files=result['files'],
        binding=result['binding']+'; exact code/version before and after',
        output=str(dest.relative_to(WORK)),notebook_sha256=identity['notebook_sha256'])
    write(WORK/'checks/e2-download.json',rec);print(json.dumps(rec))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['quota','push','observe','download'])
    try:main(p.parse_args().action)
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__,http=getattr(getattr(error,'response',None),'status_code',None),
            reason=str(error) if isinstance(error,AssertionError) else 'API/CLI failed; raw response suppressed')))
        raise SystemExit(1)
