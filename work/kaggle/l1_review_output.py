#!/usr/bin/env python3
"""Download immutable smoke outputs; keep auth and signed URLs inside SDK/requests."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from kaggle_versioned import pin_client
from kaggle_io import safe_name,write

def main():
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    p=argparse.ArgumentParser();p.add_argument('ref');p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
    api=KaggleApi();api.authenticate();owner,slug,label=pin_client(api,a.ref)
    out=ROOT.parent/'results/kaggle-runs'/slug/label[1:];out.mkdir(parents=True,exist_ok=True)
    page=None;files=0;skipped=0;log=None
    while True:
        r=ApiListKernelSessionOutputRequest();r.user_name=owner;r.kernel_slug=slug;r.version_label=label;r.page_size=200
        if page:r.page_token=page
        with api.build_kaggle_client() as c:response=c.kernels.kernels_api_client.list_kernel_session_output(r)
        for item in response.files or []:
            target=out/safe_name(item.file_name)
            if a.metadata_only and target.suffix=='.pt':skipped+=1;continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                tmp=target.with_name(target.name+'.download-tmp')
                with requests.get(item.url,stream=True,timeout=(30,120)) as stream:
                    stream.raise_for_status()
                    with tmp.open('wb') as f:
                        for chunk in stream.iter_content(4*1024**2):f.write(chunk)
                tmp.replace(target)
            files+=1
        log=log or response.log;page=response.next_page_token
        if not page:break
    if log:(out/(slug+'.log')).write_text(log)
    write(out/'download-version.json',{'ref':a.ref,'files':files,'PT_skipped':skipped,'metadata_only':a.metadata_only})
    print(json.dumps({'ref':a.ref,'files':files,'PT_skipped':skipped,'out':str(out)}))
if __name__=='__main__':main()
