#!/usr/bin/env python3
"""Pinned friend1 status/output via CLI 2.2.4 SDK; download only small review artifacts."""
import argparse, hashlib, json, sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_versioned import pin_client

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','output','metadata']);p.add_argument('ref')
    p.add_argument('--path',type=Path);a=p.parse_args()
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
        assert port==cfg['port_sha256'] and cfg['owner']=='friend1'
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
        print('L3 API FAILED',type(error).__name__,getattr(getattr(error,'response',None),'status_code',None));sys.exit(1)
