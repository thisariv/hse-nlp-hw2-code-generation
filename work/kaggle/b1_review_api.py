"""Private B1 reviewer API operations. Config directories are selected, never inspected."""
import argparse
import contextlib
import io
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from kaggle_versioned import pin_client

ROOT=Path(__file__).resolve().parent

def write(path,value):
    if path:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['push','status','output','inputs','quota','metadata','logs'])
    p.add_argument('ref');p.add_argument('--path',type=Path);a=p.parse_args()
    owner=a.ref.split('/')[0]
    assert owner == 'friend2', 'B1 review uses friend2 account only'
    for key in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(key,None)
    expected={'thisarivson':None,'friend1':str(Path('~/.kaggle-friend').expanduser()),
              'friend2':str(Path('~/.kaggle-friend2').expanduser())}
    assert owner in expected and os.environ.get('KAGGLE_CONFIG_DIR')==expected[owner]
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    if a.command=='logs':
        import signal
        def timeout(*_):raise TimeoutError('bounded live-log read')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(40);rows=[]
        log_error=None
        try:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest
            req=ApiGetKernelSessionLogsStreamRequest();req.user_name=owner;req.kernel_slug=a.ref.split('/')[1]
            req.version_label='v'+a.ref.split('/')[2];req.wait_for_logs_url_seconds=5
            with api.build_kaggle_client() as c:
                response=c.kernels.kernels_api_client.get_kernel_session_logs_stream(req)
                try:
                    sse=(response.headers.get('Content-Type') or '').startswith('text/event-stream')
                    for event in (api._iter_sse_events(response) if sse else api._iter_blob_lines(response)):
                        rows.append(event.get('data',''))
                        if len(rows)>=500:break
                finally:response.close()
        except TimeoutError:pass
        except Exception as e:log_error=type(e).__name__
        finally:signal.alarm(0)
        if a.path:
            a.path.parent.mkdir(parents=True,exist_ok=True);a.path.write_text(''.join(rows))
        print(''.join(rows)[-7000:] if rows else 'No live log events; '+str(log_error));return
    if a.command=='push':
        m=json.loads((a.path/'kernel-metadata.json').read_text());lim=json.loads((a.path/'session-limits.json').read_text())
        assert m['id']==a.ref and m['is_private'] is True and m['machine_shape']=='NvidiaTeslaT4'
        assert owner == 'friend2'
        assert a.ref in (owner+'/nlp-hw02-b1-gpu-smoke', owner+'/nlp-hw02-kaggle-b1')
        assert lim['session_timeout_seconds']==(900 if a.ref.endswith('gpu-smoke') else 40800)
        r=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(a.path),
            '--timeout',str(lim['session_timeout_seconds']),'--accelerator','NvidiaTeslaT4'],
            capture_output=True,text=True,timeout=150)
        assert r.returncode==0,'CLI push failed; raw response suppressed'
        match=re.search(r'Kernel version (\d+) successfully pushed',r.stdout);assert match,'No push receipt'
        rec=dict(ref=a.ref,version=int(match.group(1)),private=True,backend_timeout_seconds=lim['session_timeout_seconds'],
                 pushed_at_utc=datetime.now(timezone.utc).isoformat())
        write(a.path/'push-receipt.json',rec);print(json.dumps(rec));return
    if a.command=='quota':
        with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        rec=dict(owner=owner,used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),
                 free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),
                 refresh_utc=q.quota_refresh_time.isoformat())
    elif a.command=='inputs':
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        rec=dict(owner=owner,checked_at_utc=datetime.now(timezone.utc).isoformat(),inputs=[])
        for ref in ('thisarivson/nlp-hw02-26b-inputs','thisarivson/nlp-hw02-compact-cache-verified/2'):
            item=dict(ref=ref,accessible=False)
            try:
                if ref.endswith('/2'):
                    req=ApiGetKernelRequest();req.user_name='thisarivson';req.kernel_slug=ref.split('/')[1];req.version_label='v2'
                    with api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(req)
                    item.update(accessible=True,private=r.metadata.is_private)
                else:
                    req=ApiGetDatasetRequest();req.owner_slug='thisarivson';req.dataset_slug=ref.split('/')[1]
                    with api.build_kaggle_client() as c:r=c.datasets.dataset_api_client.get_dataset(req)
                    item.update(accessible=True,private=r.is_private)
            except Exception as e:item.update(error=type(e).__name__,http=getattr(getattr(e,'response',None),'status_code',None))
            rec['inputs'].append(item)
        rec['passed']=all(i.get('accessible') and i.get('private') for i in rec['inputs'])
    else:
        assert len(a.ref.split('/'))==3;pin_client(api,a.ref)
        if a.command=='status':rec=dict(ref=a.ref,status=api.kernels_status(a.ref).status.name,checked_at_utc=datetime.now(timezone.utc).isoformat())
        elif a.command=='metadata':
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
            from build_l3 import literals
            req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=a.ref.split('/')[1];req.version_label='v'+a.ref.split('/')[2]
            with api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(req)
            nb=json.loads(r.blob.source);v=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
            assert r.metadata.is_private and v['CONFIG']['owner']==owner
            port=hashlib.sha256(json.dumps(v['SOURCES'],sort_keys=True).encode()).hexdigest()
            assert port==v['CONFIG']['port_sha256']
            rec=dict(ref=a.ref,private=True,config=v['CONFIG'],dataset_sources=r.metadata.dataset_data_sources,
                     kernel_sources=r.metadata.kernel_data_sources,source_port_sha256=port,
                     modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['SOURCES'].items()})
        else:
            a.path.mkdir(parents=True,exist_ok=True)
            files,token=api.kernels_output(a.ref,str(a.path),file_pattern=r'.*\.(json|jsonl|log)$',quiet=True,page_size=200)
            assert not token,'Incomplete output pagination'
            rec=dict(ref=a.ref,small_files=len(files),all_pages=True)
            write(a.path/'review-download.json',rec);print(json.dumps(rec));return
    write(a.path,rec);print(json.dumps(rec))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('B1 API FAILED',type(e).__name__,getattr(getattr(e,'response',None),'status_code',None));sys.exit(1)
