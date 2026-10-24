"""D1 private API launch and pinned audit. Never inspect or log credential files."""
import argparse, contextlib, io, json, os, re, subprocess, sys, hashlib
from pathlib import Path
from datetime import datetime, timezone
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_versioned import pin_client


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['push','status','metadata','output','inputs','quota'])
    p.add_argument('ref'); p.add_argument('--path',type=Path); a=p.parse_args()
    owner=a.ref.split('/')[0]; assert owner in ('thisarivson','friend1')
    config=os.environ.get('KAGGLE_CONFIG_DIR')
    assert config == str(Path('~/.kaggle-friend').expanduser()) if owner=='friend1' else config is None
    if a.command=='push':
        m=json.loads((a.path/'kernel-metadata.json').read_text()); limits=json.loads((a.path/'session-limits.json').read_text())
        assert m['id']==a.ref and m['is_private'] is True and m['enable_internet'] is True
        assert m['machine_shape']=='NvidiaTeslaT4' and m['enable_gpu'] is True
        assert limits['session_timeout_seconds'] <= (720 if owner=='thisarivson' else 8400)
        r=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(a.path),
            '--timeout',str(limits['session_timeout_seconds']),'--accelerator','NvidiaTeslaT4'],
            capture_output=True,text=True,timeout=120)
        assert r.returncode==0, 'CLI push failed; raw response suppressed'
        match=re.search(r'Kernel version (\d+) successfully pushed',r.stdout)
        assert match, 'Missing successful push receipt'
        receipt=dict(ref=a.ref,version=int(match.group(1)),private=True,backend_timeout_seconds=limits['session_timeout_seconds'],
                     pushed_at_utc=datetime.now(timezone.utc).isoformat(),url='https://www.kaggle.com/code/'+a.ref)
        (a.path/'push-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt)); return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api=KaggleApi(); api.authenticate()
    if a.command=='quota':
        with api.build_kaggle_client() as c: q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        receipt=dict(account=owner,used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),
                     allowed_seconds=g.total_time_allowed.total_seconds(),
                     free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),
                     refresh_utc=q.quota_refresh_time.isoformat(),checked_at_utc=datetime.now(timezone.utc).isoformat())
        if a.path: a.path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt)); return
    if a.command=='inputs':
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        result=[]
        m=json.loads((a.path/'kernel-metadata.json').read_text())
        for ref in m['dataset_sources']:
            user,slug=ref.split('/')[:2]; req=ApiGetDatasetRequest(); req.owner_slug=user; req.dataset_slug=slug
            with api.build_kaggle_client() as c: response=c.datasets.dataset_api_client.get_dataset(req)
            assert response.is_private, 'Public input refused'
            result.append(dict(ref=ref,accessible=True,private=response.is_private))
        print(json.dumps(result)); return
    assert len(a.ref.split('/'))==3
    pin_client(api,a.ref)
    if a.command=='status':
        status=api.kernels_status(a.ref).status.name
        if a.path:
            a.path.write_text(json.dumps(dict(ref=a.ref,status=status,checked_at_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
        print(status); return
    if a.command=='metadata':
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        user,slug,version=a.ref.split('/'); req=ApiGetKernelRequest(); req.user_name=user; req.kernel_slug=slug
        with api.build_kaggle_client() as c: response=c.kernels.kernels_api_client.get_kernel(req)
        m=response.metadata
        from build_l3 import literals
        nb=json.loads(response.blob.source); code=next(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        v=literals(''.join(code) if isinstance(code,list) else code); cfg=v['CONFIG']
        port=hashlib.sha256(json.dumps(v['SOURCES'],sort_keys=True).encode()).hexdigest()
        assert port==cfg['port_sha256'] and cfg['owner']==owner and cfg['session_kind']=='d1'
        assert m.is_private and m.enable_gpu and m.enable_internet and not cfg['require_secrets'] and not cfg['enable_live_sync']
        receipt=dict(ref=a.ref,is_private=m.is_private,machine_shape=m.machine_shape,enable_internet=m.enable_internet,
            dataset_sources=m.dataset_data_sources,kernel_sources=m.kernel_data_sources,config=cfg,source_port_sha256=port,
            modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['SOURCES'].items()})
        if a.path: a.path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt)); return
    a.path.mkdir(parents=True,exist_ok=True)
    files,token=api.kernels_output(a.ref,str(a.path),file_pattern=r'.*\.(json|jsonl|log|npz|npy)$',quiet=True,page_size=200)
    assert not token, 'Incomplete output pagination'
    print(json.dumps(dict(ref=a.ref,files=len(files),all_pages=True)))


if __name__=='__main__':
    try: main()
    except Exception as error:
        print('D1 API FAILED',type(error).__name__,getattr(getattr(error,'response',None),'status_code',None)); sys.exit(1)
