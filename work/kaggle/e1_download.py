"""Parallel read-only E1 artifact download; no auth data or signed URLs in logs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import time
import requests
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from kaggle_io import safe_name


def output(api,ref,destination):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner,slug,version=ref.split('/');label='v'+version.removeprefix('v')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    def listing(version_label):
        token=None;files=[];log=None
        while True:
            req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug;req.page_size=200
            if version_label:req.version_label=version_label
            if token:req.page_token=token
            with api.build_kaggle_client()as client:r=client.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(r.files or []);log=log or r.log;token=r.next_page_token
            if not token:return files,log
    files,log=listing(label);binding='explicit version label'
    if not files:
        # Caller checks exact current version and code before and after this fallback.
        files,log=listing(None);binding='latest listing; exact current version and code verified by caller'
    selected=[];excluded=[]
    for item in files:
        name=safe_name(item.file_name)
        if '__pycache__'in name.parts or name.suffix=='.pyc':excluded.append(str(name))
        else:selected.append((name,item.url))
    assert selected,'No E1 output files are visible yet; retry read-only download later'
    def priority(job):
        name=str(job[0])
        if name.startswith('e1-src/'):return 0,name
        if name.endswith('summary.json')or name=='e1/config.json':return 1,name
        if name.startswith('e1/full/tune2048/'):return 2,name
        if name.startswith('e1/full/eval2048/'):return 3,name
        return 4,name
    selected.sort(key=priority)
    (destination/'download-plan.json').write_text(json.dumps(dict(ref=ref,binding=binding,
        selected=[str(name)for name,_ in selected],excluded_bytecode=excluded),indent=2)+'\n')
    def fetch(job):
        name,url=job;path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file():return str(name),path.stat().st_size,True
        temp=path.with_name(path.name+'.part')
        error_type=None;http=None
        for attempt in range(3):
            try:
                with requests.get(url,stream=True,timeout=(20,90))as response:
                    http=response.status_code;response.raise_for_status()
                    with temp.open('wb')as f:
                        for chunk in response.iter_content(1024**2):f.write(chunk)
                temp.replace(path);return str(name),path.stat().st_size,False
            except requests.RequestException as error:
                error_type=type(error).__name__
                if attempt<2:time.sleep(1.+attempt)
        raise AssertionError(f'E1 artifact download failed: {name}; {error_type}; HTTP{http}; URL suppressed')
    started=time.monotonic();completed=[];failures=[];seen=0
    with ThreadPoolExecutor(max_workers=16)as executor:
        futures=[executor.submit(fetch,job)for job in selected]
        for future in as_completed(futures):
            seen+=1
            try:
                name,size,reused=future.result();completed.append(dict(file=name,bytes=size,reused=reused))
            except AssertionError as error:failures.append(str(error))
            if seen%16==0 or seen==len(selected):
                print(json.dumps(dict(downloaded=len(completed),failed=len(failures),total=len(selected),seconds=round(time.monotonic()-started,1))),flush=True)
    if log:(destination/(slug+'.log')).write_text(log)
    receipt=dict(ref=ref,version_label=label,files=len(completed),binding=binding,
                 excluded_bytecode=excluded,downloaded=completed,failures=failures,seconds=time.monotonic()-started)
    (destination/'download-version.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert not failures,'; '.join(failures)
    return receipt
