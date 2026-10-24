"""B3 reviewer smoke API only; production push is deliberately unavailable, account-scoped auth; no credential inspection."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import signal
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from b1_transfer import Client
from kaggle_versioned import pin_client
from build_l3 import literals

ROOT=Path(__file__).resolve().parent

def small_outputs(api,ref,path):
    token=None;count=0;seen=set()
    while True:
        files,token=api.kernels_output(ref,str(path),file_pattern=r'.*\.(json|jsonl|log)$',quiet=True,
                                       page_size=200,page_token=token)
        count+=len(files)
        if not token:return count
        assert token not in seen,'Repeated output page token'
        seen.add(token)

def pushed_ref(stdout,owner):
    refs=set(re.findall(r'https://www\.kaggle\.com/code/([a-z0-9_-]+/[a-z0-9-]+)',stdout))
    assert len(refs)==1, 'Push accepted but canonical URL missing; inspect own kernel list before any retry'
    ref=refs.pop();assert ref.split('/')[0]==owner
    return ref

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['push','status','metadata','logs','output','probe'])
    p.add_argument('ref');p.add_argument('--path',type=Path,required=True);a=p.parse_args()
    owner,slug,*version=a.ref.split('/')
    assert (owner,slug)==('thisarivson','nlp-hw02-b3-gpu-smoke'), 'Only the authorized smoke can run'
    env=Client().env(owner)
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(k,None)
    os.environ['KAGGLE_CONFIG_DIR']=env['KAGGLE_CONFIG_DIR']
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    rec=dict(ref=a.ref,checked_at_utc=datetime.now(timezone.utc).isoformat())
    if a.action=='probe':
        with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        rec.update(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),reset_utc=str(q.quota_refresh_time))
        transfer=json.loads((ROOT.parent/'checks/b3-transfer.json').read_text())
        info=Client().private_files(transfer['dataset'])
        assert info['private'] and info['version']==transfer['version'] and info['status']=='ready'
        assert info['files']=={r['file']:r['bytes'] for r in transfer['files']}
        rec.update(dataset=transfer['dataset'],private=True,inventory_verified=True,files=info['files'])
    elif a.action=='push':
        assert not version
        folder=a.path;m=json.loads((folder/'kernel-metadata.json').read_text())
        lim=json.loads((folder/'session-limits.json').read_text());hard=lim['session_timeout_seconds']
        assert m['id']==a.ref and m['is_private'] is True and m['machine_shape']=='NvidiaTeslaT4'
        assert hard==720
        with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota;free=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds()
        assert free>=hard+120, 'Not enough smoke quota'
        rec.update(quota_free_seconds=free)
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(folder),
             '--timeout',str(hard),'--accelerator','NvidiaTeslaT4'],env=env,capture_output=True,text=True,timeout=150)
        assert run.returncode==0,'CLI push failed; raw response suppressed'
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout);assert match,'No push receipt'
        rec.update(requested_ref=a.ref,ref=pushed_ref(run.stdout,owner))
        rec.update(version=int(match.group(1)),private=True,backend_timeout_seconds=hard,
                   pushed_at_utc=datetime.now(timezone.utc).isoformat())
        a.path=folder/'push-receipt.json'
    else:
        assert len(version)==1 and version[0].isdecimal()
        if a.action!='logs':pin_client(api,a.ref)
        if a.action=='status':rec['status']=api.kernels_status(a.ref).status.name
        elif a.action=='metadata':
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
            req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug;req.version_label='v'+version[0]
            with api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(req)
            nb=json.loads(r.blob.source);v=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
            port=hashlib.sha256(json.dumps(v['SOURCES'],sort_keys=True).encode()).hexdigest()
            assert r.metadata.is_private and port==v['CONFIG']['port_sha256'] and v['CONFIG']['owner']==owner
            rec.update(private=True,config=v['CONFIG'],source_port_sha256=port,
                       modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['SOURCES'].items()})
        elif a.action=='logs':
            def timeout(*_):raise TimeoutError()
            signal.signal(signal.SIGALRM,timeout);signal.alarm(20);rows=[]
            try:
                for event in api.kernels_logs_stream(owner+'/'+slug):
                    rows.append(event.get('data',''))
                    if len(rows)>=500:break
            except TimeoutError:pass
            finally:signal.alarm(0)
            a.path.parent.mkdir(parents=True,exist_ok=True);a.path.write_text(''.join(rows))
            print(''.join(rows)[-4500:]);return
        else:
            a.path.mkdir(parents=True,exist_ok=True)
            count=small_outputs(api,a.ref,a.path)
            print(json.dumps(dict(files=count)));return
    a.path.parent.mkdir(parents=True,exist_ok=True);a.path.write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rec.items() if k not in ('modules','config')}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('B3 smoke API failed:',type(e).__name__,getattr(getattr(e,'response',None),'status_code',None));raise SystemExit(1)
