#!/usr/bin/env python3
"""Pinned friend1 status/output via CLI 2.2.4 SDK; download only small review artifacts."""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_versioned import pin_client

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','output','metadata','push']);p.add_argument('ref')
    p.add_argument('--path',type=Path);a=p.parse_args()
    assert os.environ.get('KAGGLE_CONFIG_DIR')==str(Path('~/.kaggle-friend').expanduser()), 'Friend config required'
    if a.command=='push':
        assert a.path is not None and a.ref in ('friend1/nlp-hw02-p1-gpu-smoke','friend1/nlp-hw02-kaggle-p1')
        m=json.loads((a.path/'kernel-metadata.json').read_text())
        limits=json.loads((a.path/'session-limits.json').read_text())
        assert m['id']==a.ref and m['is_private'] is True and m['machine_shape']=='NvidiaTeslaT4'
        assert limits['session_timeout_seconds']==(1080 if 'gpu-smoke' in a.ref else 43200)
        r=subprocess.run([str(Path('~/.local/bin/kaggle').expanduser()),'kernels','push','-p',str(a.path),'--timeout',str(limits['session_timeout_seconds']),'--accelerator','NvidiaTeslaT4'],capture_output=True,text=True,timeout=120)
        assert r.returncode==0, 'CLI push failed; raw output suppressed'
        match=re.search(r'Kernel version (\d+) successfully pushed',r.stdout)
        assert match, 'No successful push receipt'
        print(json.dumps({'ref':a.ref,'version':int(match.group(1)),'successful_response':True,'private':True,'timeout':limits['session_timeout_seconds']}))
        return
    assert a.ref.startswith('friend1/') and len(a.ref.split('/'))==3, 'Explicit friend version required'
    api=KaggleApi();api.authenticate();pin_client(api,a.ref)
    if a.command=='status':
        print(api.kernels_status(a.ref).status.name)
    elif a.command=='metadata':
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        owner,slug,version=a.ref.split('/');request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=slug
        with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.get_kernel(request)
        m=response.metadata
        from build_l3 import literals
        notebook=json.loads(response.blob.source)
        code=next(c['source'] for c in notebook['cells'] if c['cell_type']=='code')
        values=literals(''.join(code) if isinstance(code,list) else code);cfg=values['CONFIG']
        port=hashlib.sha256(json.dumps(values['SOURCES'],sort_keys=True).encode()).hexdigest()
        assert port==cfg['port_sha256'] and cfg['owner']=='friend1' and cfg['session_kind']=='p1'
        receipt={'ref':m.ref,'version':version,'is_private':m.is_private,'enable_gpu':m.enable_gpu,
                 'enable_internet':m.enable_internet,'machine_shape':m.machine_shape,
                 'dataset_sources':m.dataset_data_sources,'kernel_sources':m.kernel_data_sources,
                 'source_SHA256':hashlib.sha256(response.blob.source.encode()).hexdigest(),
                 'source_port_sha256':port,'owner':cfg['owner'],'gpu_smoke':cfg['gpu_smoke']}
        assert receipt['is_private'], 'Public notebook refused'
        if a.path:
            a.path.parent.mkdir(parents=True,exist_ok=True);a.path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt))
    else:
        assert a.path is not None
        files,token=api.kernels_output(a.ref,str(a.path),file_pattern=r'.*\.(json|jsonl|log)$',quiet=True,page_size=100)
        assert not token, 'Incomplete output pagination'
        receipt={'ref':a.ref,'files_downloaded':len(files),'all_pages':True,'large_PT_downloaded':False}
        (a.path/'review-download.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    try:main()
    except Exception as error:
        # Signed download URLs and library credentials must never appear in reviewer logs.
        print('P1 API FAILED',type(error).__name__,getattr(getattr(error,'response',None),'status_code',None));sys.exit(1)
