#!/usr/bin/env python3
"""Read-only pinned compact output download + independent full-array/order audit."""
import argparse, concurrent.futures as futures, datetime, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import write, sha, safe_name

def download(ref,out,workers=8,only_gate=False):
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    from kaggle_versioned import pin_client
    api=KaggleApi();api.authenticate();owner,slug,label=pin_client(api,ref)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);items=[];page=None;log=None
    while True:
        request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug;request.version_label=label;request.page_size=200
        if page:request.page_token=page
        with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        for item in response.files or []:
            name=item.file_name
            wanted=name.startswith('compact-cache/') if not only_gate else name in ('startup-gate.json','startup-live-gate.json','session-summary.json')
            if wanted:items.append((name,item.url))
        log=log or response.log;page=response.next_page_token
        if not page:break
    if log:(out/(slug+'.log')).write_text(log)
    def fetch(pair):
        name,url=pair;p=out/safe_name(name);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():return name
        for attempt in range(3):
            temporary=p.with_name(p.name+'.download-tmp')
            try:
                with requests.get(url,stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as f:
                        for block in response.iter_content(4*1024**2):f.write(block)
                temporary.replace(p);return name
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt==2:raise RuntimeError('Pinned output download failed for '+name) from None
    completed=0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for f in futures.as_completed([pool.submit(fetch,item) for item in items]):
            f.result();completed+=1
            if completed%60==0 or completed==len(items):print('PINNED DOWNLOAD',completed,'/',len(items),flush=True)
    write(out/'download-version.json',{'ref':ref,'version_label':label,'files':len(items),
          'selection':'startup gates + log' if only_gate else 'complete compact-cache tree + log',
          'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auth_and_signed_urls_saved':False})
    return out

def verify(out):
    import numpy as np
    from l1_cache import CompactCache,BASE_SHA,EXT_SHA,ROWS,ORDER_SEED
    started=time.monotonic();root=Path(out)/'compact-cache';cache=CompactCache(root,verify=True)
    manifest=cache.manifest;summary=json.loads((root/'cache-summary.json').read_text())
    assert len(cache)==ROWS and [r['shard'] for r in manifest['shards']]==list(range(60))
    assert manifest['original_manifest_sha256']==[BASE_SHA,EXT_SHA]
    assert sha(root/'compact-manifest.json')==summary['manifest_sha256']
    tokenizer=manifest['tokenizer'];assert sha(root/tokenizer['file'])==tokenizer['sha256']
    assert summary['roundtrip_rows']>=50000 and summary['answer_mismatches']==0
    assert summary['roundtrip']['matches']==summary['roundtrip']['denominator']==summary['roundtrip_rows']
    # Source/provenance copies independently agree with original producer receipts.
    for rec in manifest['shards']:
        original={f['file']:f for f in rec['original_files']};current={f['file']:f for f in rec['files']}
        for name in ('source.npy','source-offsets.npy','provenance-rows.npy','question-sha256.npy'):
            assert original[name]==current[name]
        a=cache.parts[rec['shard']]
        assert np.all((a['target.npy']>=3)&(a['target.npy']<8195))
        assert len(np.unique(a['provenance-rows.npy']))==rec['rows']
    order=np.load(root/'order.npy',mmap_mode='r',allow_pickle=False);order_meta=json.loads((root/'order.json').read_text())
    assert order.dtype==np.uint32 and len(order)==ROWS and order_meta['seed']==ORDER_SEED
    assert sha(root/'order.npy')==order_meta['file']['sha256']
    assert np.array_equal(np.sort(order),np.arange(ROWS,dtype=np.uint32))
    # Reconstruct independently from global permutation + stable length sorting and batch shuffling.
    from l1_cache import make_order
    assert np.array_equal(order,make_order(cache))
    report={'passed':True,'rows':ROWS,'shards':[0,59],'arrays_verified':60*6,
            'manifest_sha256':cache.manifest_sha,'order_sha256':order_meta['file']['sha256'],
            'tokenizer_sha256':tokenizer['sha256'],'representation_code_sha256':manifest['representation_code_sha256'],
            'roundtrip_rows':summary['roundtrip_rows'],'answer_mismatches':0,'order_seed':ORDER_SEED,
            'CPU_build_seconds':summary['seconds'],'audit_seconds':time.monotonic()-started,
            'source_provenance_unchanged':True,'order_bijection_and_reconstruction':True,
            'files':[{'file':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]}
    write(ROOT.parent/'checks/kaggle-l1-cache-output.json',report)
    print(json.dumps({k:report[k] for k in ('passed','rows','manifest_sha256','order_sha256','roundtrip_rows','answer_mismatches','CPU_build_seconds')},indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ref');p.add_argument('--out',required=True);p.add_argument('--gate-only',action='store_true');p.add_argument('--local-only',action='store_true');p.add_argument('--download-only',action='store_true');a=p.parse_args()
    if not a.local_only:download(a.ref,a.out,only_gate=a.gate_only)
    if not a.gate_only and not a.download_only:verify(a.out)
