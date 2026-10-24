"""CLI2.2.4 version-label compatibility without editing installation. Read-only SDK calls."""
import argparse,contextlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import safe_name,write

def pin_client(api,ref):
    owner,slug,version=ref.split('/');label=version if version.startswith('v') else 'v'+version
    build=api.build_kaggle_client
    class Proxy:
        def __init__(self,target):self.target=target
        def __getattr__(self,name):
            method=getattr(self.target,name)
            def pinned(request,*args,**kwargs):
                if name in ('get_kernel','get_kernel_session_status','list_kernel_session_output','list_kernel_files'):
                    request.kernel_slug=slug;request.version_label=label
                return method(request,*args,**kwargs)
            return pinned
    @contextlib.contextmanager
    def client():
        with build() as c:
            target=c.kernels.kernels_api_client
            class Surface:
                kernels_api_client=Proxy(target)
            class Client:
                kernels=Surface()
            yield Client()
    api.build_kaggle_client=client
    return owner,slug,label

def output(api,ref,out):
    import requests
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner,slug,label=pin_client(api,ref);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    token=None;files=0;log=None
    while True:
        r=ApiListKernelSessionOutputRequest();r.user_name=owner;r.kernel_slug=slug;r.version_label=label;r.page_size=200
        if token:r.page_token=token
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.list_kernel_session_output(r)
        for item in response.files or []:
            target=out/safe_name(item.file_name);target.parent.mkdir(parents=True,exist_ok=True)
            # Immutable version-specific target. No URLs/auth values printed; signed download URL stays in requests.
            if not target.exists():
                tmp=target.with_name(target.name+'.download-tmp')
                with requests.get(item.url,stream=True,timeout=(30,120)) as stream:
                    stream.raise_for_status()
                    with tmp.open('wb') as f:
                        for chunk in stream.iter_content(4*1024**2):f.write(chunk)
                tmp.replace(target)
            files+=1
        log=log or response.log;token=response.next_page_token
        if not token:break
    if log:(out/(slug+'.log')).write_text(log)
    write(out/'download-version.json',{'ref':ref,'version_label':label,'files':files,'CLI_224_compatibility':True})
    print(f'{ref}: {files} output files saved to {out}')

if __name__=='__main__':
    from kaggle.api.kaggle_api_extended import KaggleApi
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['output','pull','status']);p.add_argument('ref');p.add_argument('--path')
    args=p.parse_args();api=KaggleApi();api.authenticate()
    if args.command=='output':output(api,args.ref,args.path)
    else:
        pin_client(api,args.ref)
        if args.command=='pull':api.kernels_pull(args.ref,args.path,metadata=True,quiet=False)
        else:
            result=api.kernels_status(args.ref);print(result.status.name)
