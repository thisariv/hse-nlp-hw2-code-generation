"""Read-only B3 SHA-pinned output audit; download ONE stable, CPU check, delete.

SDK authenticates from default ~/.kaggle. No credential files/values inspected.
The versioned output endpoint currently returns an empty list for v1; latest API
bytes must match the archived version1 manifest AND pinned B3 latest exactly.
"""
import argparse,contextlib,io,json,os,subprocess,tempfile
from pathlib import Path
from datetime import datetime,timezone
from b1_transfer import Client
from build_l3 import ROOT,write
import sys
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha,safe_name


def prepare(cpu_python,cache):
    pins=json.loads((ROOT/'b4-anchor.json').read_text());pin=pins['anchor']
    c=Client();os.environ['KAGGLE_CONFIG_DIR']=c.env('thisarivson')['KAGGLE_CONFIG_DIR']
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(k,None)
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    import requests
    files={};token=None
    while True:
        req=ApiListKernelSessionOutputRequest();req.user_name='thisarivson';req.kernel_slug='nlp-hw02-b3';req.page_size=200;req.page_token=token
        with api.build_kaggle_client() as client:r=client.kernels.kernels_api_client.list_kernel_session_output(req)
        for f in r.files or []:
            assert f.file_name not in files;files[f.file_name]=f
        token=r.next_page_token
        if not token:break
    name='model-GPU1/'+str(safe_name(pin['file']));assert name in files,'Pinned stable absent'
    with tempfile.TemporaryDirectory(prefix='nlp-b4-cpu-') as folder:
        folder=Path(folder)
        def fetch(name,expected=None):
            target=folder/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True)
            with requests.get(files[name].url,stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                with target.open('wb') as out:
                    for block in r.iter_content(4*1024**2):out.write(block)
            if expected:
                assert target.stat().st_size==expected['bytes'] and sha(target)==expected['sha256']
            return target
        manifest=fetch('b1-output-manifest.json');assert sha(manifest)==pins['source_manifest_sha256']
        inventory={r['file']:r for r in json.loads(manifest.read_text())['files']}
        latest=fetch('model-GPU1/latest.json',inventory['model-GPU1/latest.json'])
        assert json.loads(latest.read_text())==pin
        assert all(inventory[name][k]==pin[k] for k in ('bytes','sha256'))
        order=fetch('weighted-order/order.npy',inventory['weighted-order/order.npy'])
        def check(*args):subprocess.run([str(cpu_python),str(ROOT/'b4_cpu_resume_check.py'),*map(str,args)],check=True)
        check('--capture-order',order,'--cache',cache,'--out',ROOT.parent/'checks/b4-order.json')
        full=fetch(name,pin)
        try:check('--file',full,'--gpu','0','--cache',cache,'--order-receipt',ROOT.parent/'checks/b4-order.json','--out',ROOT.parent/'checks/b4-real-resume.json')
        finally:full.unlink(missing_ok=True)
        assert not list(folder.rglob('*.pt'))
    with api.build_kaggle_client() as client:q=client.kernels.kernels_api_client.get_accelerator_quota_statistics()
    g=q.gpu_quota
    write(ROOT.parent/'checks/b4-inputs.json',dict(passed=True,checked_at_utc=datetime.now(timezone.utc).isoformat(),
          source=pins['source'],stable_file=name,stable_present_in_output=True,bytes=pin['bytes'],sha256=pin['sha256'],
          version_binding='latest endpoint; archived v1 manifest SHA + exact latest + full bytes/SHA',
          source_manifest_sha256=pins['source_manifest_sha256'],single_stable_downloaded=True,local_payload_deleted=True,
          free_quota_seconds=(g.total_time_allowed-g.time_used-g.time_reserved).total_seconds(),quota_reset_utc=str(q.quota_refresh_time),
          read_only=True,GPU_runs=0,pushes=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-python',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    try:prepare(a.cpu_python,a.cache)
    except Exception as e:print('B4 preparation failed:',type(e).__name__);raise SystemExit(1)
