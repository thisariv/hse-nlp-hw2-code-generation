"""Reviewer-only private B7 pushes; read-only quota/status/compact output, SDK auth, no Secrets."""
import argparse,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
from b6_prepare import ReadClient
from b5_review_api import require_known_version,require_absent_target
from b3_review_api import pushed_ref
from build_l3 import ROOT,write


def require_single_push(remote_version, known_versions, smoke):
    """B7 authorizes exactly one reviewer push per target, with no automatic retry."""
    require_known_version(remote_version, known_versions, smoke)
    assert not known_versions and remote_version == 0, 'B7 target already pushed; monitor latest version, do not retry'


def reserve_push(path, rec):
    """Persist intent before the network mutation; ambiguous responses cannot enable a retry."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(rec,f,indent=2)


def launch_gate(hand,identity,smoke,receipt=None):
    assert hand['ready'] and hand['CPU_ready']
    assert all(hand[k]==identity[k] for k in ('modules','C_overrides'))
    assert identity['config_sha256']==hand['smoke_config_sha256' if smoke else 'production_config_sha256']
    if not smoke:
        assert hand['smoke_passed'] and not hand['GPU_smoke_pending'] and receipt
        assert receipt['passed'] is True and receipt['smoke'] is True
        assert all(receipt[k]==identity[k] for k in ('modules','C_overrides'))
        assert receipt['config_sha256']==hand['smoke_config_sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['quota','status','metadata','output','push'])
    p.add_argument('ref');p.add_argument('--path',type=Path,required=True);a=p.parse_args()
    owner,slug,*version=a.ref.split('/');assert owner=='friend1' and slug in ('nlp-hw02-b7','nlp-hw02-b7-gpu-smoke')
    assert not version or len(version)==1 and version[0].isdigit()
    c=ReadClient();api=c.api
    def metadata():
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug
        if version:req.version_label='v'+version[0]
        with api.build_kaggle_client() as client:return client.kernels.kernels_api_client.get_kernel(req)
    def quota():
        with api.build_kaggle_client() as client:q=client.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        return dict(free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),
            used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),reset_utc=str(q.quota_refresh_time))
    rec=dict(ref=a.ref,checked_at_utc=datetime.now(timezone.utc).isoformat())
    if a.action=='quota':rec.update(quota())
    elif a.action=='push':
        assert not version;smoke=slug.endswith('gpu-smoke');folder=a.path
        from b7_static_check import main as check
        check()
        handpath=ROOT.parent/'results/next-b7.json';hand=json.loads(handpath.read_text())
        assert folder.resolve()==(ROOT.parent/hand['smoke_bundle_path' if smoke else 'bundle_path']).resolve()
        identity=json.loads((folder/'source-identity.json').read_text());m=json.loads((folder/'kernel-metadata.json').read_text())
        hard=json.loads((folder/'session-limits.json').read_text())['session_timeout_seconds']
        assert m['id']==a.ref and m['title']==slug and m['is_private'] and m['enable_internet'] and m['enable_gpu']
        assert m['machine_shape']=='NvidiaTeslaT4' and hard==(720 if smoke else 40800)
        receipt=json.loads((ROOT.parent/hand['GPU_smoke_receipt']).read_text()) if not smoke and hand.get('GPU_smoke_receipt') else None
        launch_gate(hand,identity,smoke,receipt)
        receipt_dir=ROOT.parent/'checks'/('b7-review-smoke-push-v' if smoke else 'b7-review-production-push-v')
        known=[json.loads(p.read_text())['version'] for p in receipt_dir.glob('*.json')]
        absent=None
        try:
            r=metadata();assert r.metadata.is_private;remote=r.metadata.current_version_number
        except Exception as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if status not in (403,404):raise
            if status==403:
                owned=set();complete=False
                for page in range(1,101):
                    rows=api.kernels_list(mine=True,page=page,page_size=100) or [];owned.update(r.ref for r in rows)
                    if not rows:complete=True;break
                require_absent_target(a.ref,owned,known,complete)
                absent=dict(HTTP_status=status,owned_inventory_complete=complete,target_absent=a.ref not in owned)
            else:absent=dict(HTTP_status=status)
            remote=0
        require_single_push(remote,known,smoke)
        q=quota();assert q['free_seconds']>=hard+120
        reserve_push(receipt_dir.with_suffix('.intent.json'),dict(rec,remote_version=remote,
            absence_evidence=absent,quota_before=q,private=True,backend_timeout_seconds=hard,**identity))
        result=subprocess.run([c.cli,'kernels','push','-p',str(folder),'--accelerator','NvidiaTeslaT4','--timeout',str(hard)],
            env=c.env('friend1'),capture_output=True,text=True,timeout=150)
        assert result.returncode==0,'Push failed; inspect server state before retry; raw response suppressed'
        match=re.search(r'Kernel version (\d+) successfully pushed',result.stdout);assert match,'Ambiguous push receipt; do not retry blindly'
        rec.update(version=int(match.group(1)),ref=pushed_ref(result.stdout,owner),quota_before=q,**identity,
            private=True,backend_timeout_seconds=hard,pushed_at_utc=datetime.now(timezone.utc).isoformat())
        hand['pushes']=hand.get('pushes',0)+1;hand['GPU_runs']=hand.get('GPU_runs',0)+1
        prefix='smoke' if smoke else 'production';hand[prefix+'_url']='https://www.kaggle.com/code/'+a.ref+'/versions?versionNumber='+str(rec['version'])
        hand[prefix+'_status']='PUSHED';hand[prefix+'_version']=rec['version'];write(handpath,hand)
        a.path=receipt_dir/(str(rec['version'])+'.json')
    elif a.action=='metadata':
        r=metadata();assert r.metadata.is_private
        rec.update(private=True,current_version=r.metadata.current_version_number,machine_shape=r.metadata.machine_shape,
            enable_gpu=r.metadata.enable_gpu,enable_internet=r.metadata.enable_internet)
    elif a.action=='status':
        from kaggle_versioned import pin_client
        if version:pin_client(api,a.ref)
        rec['status']=api.kernels_status(a.ref).status.name
    else:
        assert version,'Pin output version'
        a.path.mkdir(parents=True,exist_ok=True)
        manifest=c.kernel_file(a.ref,'b1-output-manifest.json',a.path)
        inventory=json.loads(manifest.read_text())['files'];count=1
        for row in inventory:
            if Path(row['file']).suffix not in ('.json','.jsonl','.log'):continue
            path=c.kernel_file(a.ref,row['file'],a.path)
            from kaggle_io import sha
            assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];count+=1
        write(a.path/'review-download.json',dict(ref=a.ref,files=count,version_binding='version-label/latest fallback + source/config/producer SHA audit required'))
        print('B7 compact outputs downloaded',count);return
    write(a.path,rec);print(json.dumps({k:v for k,v in rec.items() if k not in ('modules','C_overrides')}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('B7 reviewer API failed:',type(e).__name__,'details suppressed');raise SystemExit(1)
