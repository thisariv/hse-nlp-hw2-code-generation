"""Authorized private B4 smoke/production API; SDK auth only, no credential inspection."""
import argparse,contextlib,hashlib,io,json,os,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
from b1_transfer import Client
from build_l3 import ROOT,literals,write
from b3_review_api import small_outputs,pushed_ref
from kaggle_versioned import pin_client


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['quota','inputs','push','status','metadata','output'])
    p.add_argument('ref');p.add_argument('--path',type=Path,required=True);a=p.parse_args()
    owner,slug,*version=a.ref.split('/')
    assert owner=='thisarivson' and slug in ('nlp-hw02-b4','nlp-hw02-b4-gpu-smoke')
    env=Client().env(owner)
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(k,None)
    os.environ['KAGGLE_CONFIG_DIR']=env['KAGGLE_CONFIG_DIR']
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    def metadata(ref):
        user,name,*v=ref.split('/');req=ApiGetKernelRequest();req.user_name=user;req.kernel_slug=name
        if v:req.version_label='v'+v[0]
        with api.build_kaggle_client() as c:return c.kernels.kernels_api_client.get_kernel(req)
    def quota():
        with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        return dict(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),
                    used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),
                    reset_utc=str(q.quota_refresh_time))
    rec=dict(ref=a.ref,checked_at_utc=datetime.now(timezone.utc).isoformat())
    if a.action=='quota':rec.update(quota())
    elif a.action=='inputs':
        rows=[]
        for ref in ('thisarivson/nlp-hw02-b3/1','thisarivson/nlp-hw02-compact-cache-verified/2'):
            r=metadata(ref);assert r.metadata.is_private
            rows.append(dict(ref=ref,private=True,source_sha256=hashlib.sha256(r.blob.source.encode()).hexdigest()))
        info=Client().private_files('thisarivson/nlp-hw02-26b-inputs');assert info['exists'] and info['private'] and info['status']=='ready'
        rec.update(passed=True,kernel_inputs=rows,dataset=dict(ref='thisarivson/nlp-hw02-26b-inputs',private=True,ready=True))
    elif a.action=='push':
        assert not version
        folder=a.path;m=json.loads((folder/'kernel-metadata.json').read_text())
        hard=json.loads((folder/'session-limits.json').read_text())['session_timeout_seconds']
        smoke=slug.endswith('gpu-smoke');assert hard==(720 if smoke else 40800)
        assert m['id']==a.ref and m['is_private'] is True and m['machine_shape']=='NvidiaTeslaT4'
        assert m['kernel_sources']==['thisarivson/nlp-hw02-b3','thisarivson/nlp-hw02-compact-cache-verified/2']
        modules=json.loads((folder/'source-identity.json').read_text())['modules']
        if not smoke:
            hand=json.loads((ROOT.parent/'results/next-b4.json').read_text())
            assert hand['ready'] and hand['smoke_passed'] and hand['modules']==modules
            evidence=json.loads((ROOT.parent/hand['GPU_smoke_receipt']).read_text())
            assert evidence['passed'] is True and evidence['modules']==modules
        q=quota();assert q['free_seconds']>=(840 if smoke else 42000),'Insufficient quota'
        rec.update(quota_before=q,request_started_at_utc=datetime.now(timezone.utc).isoformat())
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(folder),
             '--timeout',str(hard),'--accelerator','NvidiaTeslaT4'],env=env,capture_output=True,text=True,timeout=150)
        assert run.returncode==0,'Push failed; raw response suppressed; inspect status before retry'
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout);assert match,'Push receipt missing; do not retry blindly'
        rec.update(ref=pushed_ref(run.stdout,owner),version=int(match.group(1)),private=True,
                   backend_timeout_seconds=hard,pushed_at_utc=datetime.now(timezone.utc).isoformat(),modules=modules)
        a.path=ROOT.parent/'checks'/('b4-review-smoke-push-v' if smoke else 'b4-review-production-push-v')/(str(rec['version'])+'.json')
    else:
        assert len(version)==1 and version[0].isdecimal()
        if a.action=='metadata':
            r=metadata(a.ref);assert r.metadata.is_private
            nb=json.loads(r.blob.source);v=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
            port=hashlib.sha256(json.dumps(v['SOURCES'],sort_keys=True).encode()).hexdigest()
            assert port==v['CONFIG']['port_sha256'] and v['CONFIG']['owner']==owner
            rec.update(private=True,last_run_time_utc=str(r.metadata.last_run_time),config=v['CONFIG'],source_port_sha256=port,
                       modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['SOURCES'].items()})
        elif a.action=='status':
            pin_client(api,a.ref);rec['status']=api.kernels_status(a.ref).status.name
        else:
            # Kaggle's versioned output list can be empty. Latest output is subsequently
            # bound to this version by all source SHA and the downloaded manifest.
            a.path.mkdir(parents=True,exist_ok=True)
            count=small_outputs(api,owner+'/'+slug,a.path)
            print(json.dumps(dict(ref=a.ref,files=count,version_binding='review audit: source SHA and manifest')));return
    write(a.path,rec)
    print(json.dumps({k:v for k,v in rec.items() if k not in ('config','modules')}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('B4 review API failed:',type(e).__name__,getattr(getattr(e,'response',None),'status_code',None));raise SystemExit(1)
