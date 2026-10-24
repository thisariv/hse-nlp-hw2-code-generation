"""Authorized private B5 smoke/production API; SDK auth only, no credential inspection."""
import argparse,contextlib,hashlib,io,json,os,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
from b1_transfer import Client
from build_l3 import ROOT,literals,write
from b3_review_api import small_outputs,pushed_ref
from kaggle_versioned import pin_client


def require_known_version(remote_version, known_versions, smoke, cancelled_retry=False):
    """Refuse concurrent pushes; permit one explicitly verified zero-GPU cancelled retry."""
    known=max(known_versions,default=0)
    if cancelled_retry:
        assert not smoke and len(known_versions)==1 and remote_version>=known, 'Only one zero-GPU cancelled production retry'
        return
    assert remote_version==known, 'Remote version changed; inspect other session before push'
    assert not smoke or len(known_versions)<2, 'The two authorized reviewer smoke pushes are exhausted'
    assert smoke or known==0, 'Production already pushed; inspect it instead of launching again'


def require_absent_target(ref, owned_refs, known_versions, fully_listed):
    """A 403 alone never proves that a private target does not exist."""
    assert not known_versions and fully_listed and ref not in owned_refs, 'Cannot establish new target absence'


def require_cancelled_retry(status, has_output, before, now, code_exact):
    assert status=='CANCEL_ACKNOWLEDGED' and not has_output and code_exact, 'Cancelled retry evidence incomplete'
    assert now['used_seconds']==before['used_seconds'] and now['reserved_seconds']==0, 'Production used/reserved GPU; do not retry'


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['quota','inputs','push','status','metadata','output'])
    p.add_argument('ref');p.add_argument('--path',type=Path,required=True);p.add_argument('--retry-cancelled',action='store_true');a=p.parse_args()
    owner,slug,*version=a.ref.split('/')
    assert owner=='friend1' and slug in ('nlp-hw02-b5','nlp-hw02-b5-gpu-smoke')
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
        for ref in ('thisarivson/nlp-hw02-compact-cache-verified/2',):
            r=metadata(ref);assert r.metadata.is_private
            rows.append(dict(ref=ref,private=True,source_sha256=hashlib.sha256(r.blob.source.encode()).hexdigest()))
        datasets=[]
        for ref in ('friend1/nlp-hw02-b5-inputs','friend1/nlp-hw02-l3-live'):
            info=Client().private_files(ref);assert info['exists'] and info['private'] and info['status']=='ready'
            datasets.append(dict(ref=ref,private=True,version=info['version'],files=info['files']))
        hand=json.loads((ROOT.parent/'results/next-b5.json').read_text())
        assert datasets[0]['version']==hand['transfer']['version']
        assert datasets[0]['files']=={r['file']:r['bytes'] for r in hand['transfer']['files']}
        assert datasets[1]['version']==12
        starter=Client().private_files('thisarivson/nlp-hw02-26b-inputs')
        assert starter['exists'] and starter['private'] and starter['status']=='ready'
        assert api.dataset_list_files('thisarivson/nlp-hw02-26b-inputs',page_size=100).files
        rec.update(passed=True,kernel_inputs=rows,datasets=datasets,starter_private_accessible=True)
    elif a.action=='push':
        assert not version
        folder=a.path;m=json.loads((folder/'kernel-metadata.json').read_text())
        hard=json.loads((folder/'session-limits.json').read_text())['session_timeout_seconds']
        smoke=slug.endswith('gpu-smoke');assert hard==(720 if smoke else 40800)
        assert m['id']==a.ref and m['is_private'] is True and m['machine_shape']=='NvidiaTeslaT4'
        assert m['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
        assert m['dataset_sources']==['thisarivson/nlp-hw02-26b-inputs','friend1/nlp-hw02-b5-inputs/1','friend1/nlp-hw02-l3-live/12']
        source_identity=json.loads((folder/'source-identity.json').read_text());modules=source_identity['modules']
        receipt_dir=ROOT.parent/'checks'/('b5-review-smoke-push-v' if smoke else 'b5-review-production-push-v')
        known_versions=[json.loads(p.read_text())['version'] for p in receipt_dir.glob('*.json')]
        try:
            remote=metadata(a.ref);assert remote.metadata.is_private
            remote_version=remote.metadata.current_version_number
        except Exception as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if status not in (403,404):raise
            if status==403:
                owned=set();fully_listed=False
                for page in range(1,101):
                    rows=api.kernels_list(mine=True,page=page,page_size=100) or []
                    owned.update(r.ref for r in rows)
                    if not rows:fully_listed=True;break
                require_absent_target(a.ref,owned,known_versions,fully_listed)
                rec['absent_target_verified_in_owned_inventory']=True
            remote_version=0
        q=None
        if a.retry_cancelled:
            assert not smoke and known_versions
            prior=json.loads((receipt_dir/(str(max(known_versions))+'.json')).read_text())
            local=json.loads((folder/m['code_file']).read_text());actual=json.loads(remote.blob.source)
            def codes(nb):return [''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in nb['cells'] if c['cell_type']=='code']
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.version_label='v'+str(remote_version);req.page_size=1
            with api.build_kaggle_client() as c:out=c.kernels.kernels_api_client.list_kernel_session_output(req)
            current_status=api.kernels_status(a.ref).status.name;q=quota()
            require_cancelled_retry(current_status,bool(out.files or out.next_page_token),prior['quota_before'],q,codes(local)==codes(actual))
            rec.update(retry_after_cancelled_version=remote_version,zero_GPU_spent_since_previous_push=True,full_notebook_code_exact=True)
        require_known_version(remote_version,known_versions,smoke,a.retry_cancelled)
        if not smoke:
            hand=json.loads((ROOT.parent/'results/next-b5.json').read_text())
            assert hand['ready'] and hand['smoke_passed'] and all(hand[k]==v for k,v in source_identity.items())
            evidence=json.loads((ROOT.parent/hand['GPU_smoke_receipt']).read_text())
            assert evidence['passed'] is True and all(evidence[k]==v for k,v in source_identity.items())
        q=q or quota();assert q['free_seconds']>=(840 if smoke else 42000),'Insufficient quota'
        rec.update(quota_before=q,request_started_at_utc=datetime.now(timezone.utc).isoformat())
        run=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(folder),
             '--timeout',str(hard),'--accelerator','NvidiaTeslaT4'],env=env,capture_output=True,text=True,timeout=150)
        assert run.returncode==0,'Push failed; raw response suppressed; inspect status before retry'
        match=re.search(r'Kernel version (\d+) successfully pushed',run.stdout);assert match,'Push receipt missing; do not retry blindly'
        rec.update(ref=pushed_ref(run.stdout,owner),version=int(match.group(1)),private=True,
                   backend_timeout_seconds=hard,pushed_at_utc=datetime.now(timezone.utc).isoformat(),**source_identity)
        handpath=ROOT.parent/'results/next-b5.json';hand=json.loads(handpath.read_text())
        hand['pushes']=hand.get('pushes',0)+1;hand['GPU_runs']=hand.get('GPU_runs',0)+1
        prefix='smoke' if smoke else 'production';hand[prefix+'_url']='https://www.kaggle.com/code/'+rec['ref']+'/versions?versionNumber='+str(rec['version'])
        if smoke:hand['smoke_pushed_at_utc']=rec['pushed_at_utc'];hand['smoke_status']='PUSHED'
        else:hand.update(production_version=rec['version'],pushed_at_utc=rec['pushed_at_utc'],quota_before_production=q,status='PUSHED')
        write(handpath,hand)
        a.path=ROOT.parent/'checks'/('b5-review-smoke-push-v' if smoke else 'b5-review-production-push-v')/(str(rec['version'])+'.json')
    else:
        assert len(version)==1 and version[0].isdecimal()
        if a.action=='metadata':
            r=metadata(a.ref);assert r.metadata.is_private
            assert r.metadata.enable_gpu and r.metadata.enable_internet and not r.metadata.enable_tpu
            assert r.metadata.machine_shape=='NvidiaTeslaT4'
            nb=json.loads(r.blob.source);v=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
            import ast
            first=next(c['source'] for c in nb['cells'] if c['cell_type']=='code')
            overrides=next(ast.literal_eval(n.value) for n in ast.parse(first).body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='C_OVERRIDES' for t in n.targets))
            port=hashlib.sha256(json.dumps(dict(SOURCES=v['SOURCES'],C_OVERRIDES=overrides),sort_keys=True).encode()).hexdigest()
            assert port==v['CONFIG']['port_sha256'] and v['CONFIG']['owner']==owner
            rec.update(private=True,machine_shape=r.metadata.machine_shape,enable_gpu=True,enable_internet=True,
                       server_current_version_number=r.metadata.current_version_number,
                       last_run_time_utc=str(r.metadata.last_run_time),config=v['CONFIG'],source_port_sha256=port,
                       modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['SOURCES'].items()},
                       C_overrides={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()})
        elif a.action=='status':
            pin_client(api,a.ref);rec['status']=api.kernels_status(a.ref).status.name
        else:
            # Kaggle's versioned output list can be empty. Latest output is subsequently
            # bound to this version by all source SHA and the downloaded manifest.
            a.path.mkdir(parents=True,exist_ok=True)
            original_build=api.build_kaggle_client;pin_client(api,a.ref)
            count=small_outputs(api,owner+'/'+slug,a.path)
            binding='version-label + review source SHA and manifest'
            if not count:
                api.build_kaggle_client=original_build
                count=small_outputs(api,owner+'/'+slug,a.path)
                binding='latest fallback + review source SHA and manifest'
            write(a.path/'review-download.json',dict(ref=a.ref,files=count,version_binding=binding))
            print(json.dumps(dict(ref=a.ref,files=count,version_binding=binding)));return
    write(a.path,rec)
    print(json.dumps({k:v for k,v in rec.items() if k not in ('config','modules','C_overrides')}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('B5 review API failed:',type(e).__name__,getattr(getattr(e,'response',None),'status_code',None));raise SystemExit(1)
