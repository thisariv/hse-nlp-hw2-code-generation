"""Account-scoped metadata/quota + explicitly authorized EV push only. No B2 push command."""
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from b1_transfer import Client

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['quota','inputs','ev-push','ev-status','ev-metadata'])
    p.add_argument('--owner',choices=['thisarivson','friend1','friend2'],default='thisarivson');p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    env=Client().env(a.owner)
    for key in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(key,None)
    os.environ['KAGGLE_CONFIG_DIR']=env['KAGGLE_CONFIG_DIR']
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    rec=dict(owner=a.owner,checked_at_utc=datetime.now(timezone.utc).isoformat())
    if a.action=='quota':
        with api.build_kaggle_client() as c:q=c.kernels.kernels_api_client.get_accelerator_quota_statistics()
        g=q.gpu_quota
        rec.update(used_seconds=g.time_used.total_seconds(),reserved_seconds=g.time_reserved.total_seconds(),
                   free_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),refresh_utc=q.quota_refresh_time.isoformat())
    elif a.action=='inputs':
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        rows=[]
        for ref in ('thisarivson/nlp-hw02-26b-inputs','thisarivson/nlp-hw02-compact-cache-verified/2'):
            if ref.endswith('/2'):
                req=ApiGetKernelRequest();req.user_name='thisarivson';req.kernel_slug=ref.split('/')[1];req.version_label='v2'
                with api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(req)
                private=r.metadata.is_private
            else:
                req=ApiGetDatasetRequest();req.owner_slug='thisarivson';req.dataset_slug=ref.split('/')[1]
                with api.build_kaggle_client() as c:r=c.datasets.dataset_api_client.get_dataset(req)
                private=r.is_private
            rows.append(dict(ref=ref,accessible=True,private=private))
        rec['inputs']=rows
    else:
        assert a.owner=='thisarivson'
        ref='thisarivson/nlp-hw02-b1-ev';rec['ref']=ref
        if a.action=='ev-push':
            folder=ROOT/'bundles/b1_ev';m=json.loads((folder/'kernel-metadata.json').read_text())
            assert m['id']==ref and m['is_private'] is True and m['machine_shape']=='NvidiaTeslaT4'
            assert json.loads((folder/'session-limits.json').read_text())['session_timeout_seconds']==2700
            r=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(folder),
                              '--timeout','2700','--accelerator','NvidiaTeslaT4'],env=env,capture_output=True,text=True,timeout=150)
            assert r.returncode==0,'EV CLI push failed; raw response suppressed'
            match=re.search(r'Kernel version (\d+) successfully pushed',r.stdout);assert match,'No push receipt'
            rec.update(version=int(match.group(1)),private=True,backend_timeout_seconds=2700)
        elif a.action=='ev-status':
            raw=api.kernels_status(ref);rec['status']=getattr(raw.status,'name',str(raw.status)).upper()
        else:
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
            req=ApiGetKernelRequest();req.user_name='thisarivson';req.kernel_slug='nlp-hw02-b1-ev'
            with api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.get_kernel(req)
            from build_l3 import literals
            import hashlib
            nb=json.loads(r.blob.source);values=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
            port=hashlib.sha256(json.dumps(values['SOURCES'],sort_keys=True).encode()).hexdigest()
            assert port==values['CONFIG']['port_sha256'] and r.metadata.is_private
            rec.update(private=True,source_port_sha256=port,owner_in_config=values['CONFIG']['owner'],
                       session_ceiling_seconds=values['CONFIG']['session_ceiling_seconds'])
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('EV/B2 API failed:',type(error).__name__);sys.exit(1)
