"""Read-only, version-pinned compact L2 download; Kaggle SDK owns authentication.

Run with the interpreter from the Kaggle CLI launcher. Never downloads full states.
Signed URLs stay in memory; receipts contain only file names, bytes and SHA256.
"""
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

from kaggle_versioned import pin_client

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/kaggle-runs/nlp-hw02-kaggle-l2/1'
KERNEL = 'thisarivson/nlp-hw02-kaggle-l2/2'
LIVE = 'thisarivson/nlp-hw02-l2-live'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def safe(name):
    p = Path(name)
    assert not p.is_absolute() and '..' not in p.parts
    return p


def wanted(name, live=False):
    if not live and name in {'src/exp27_decode.py','src/l2_sampler.py','src/l2_eval.py','src/l1_eval.py'}:
        return True
    if name.endswith(('.json', '.jsonl', '.log', '.txt', '.csv')):
        return True
    return (('__weights__l2-28800-fp16.pt' in name) if live
            else ('/weights/l2-' in name and name.endswith('-fp16.pt')))


def transfer(response, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.download-tmp')
    response.raise_for_status()
    with tmp.open('wb') as f:
        for chunk in response.iter_content(4 * 1024**2):
            f.write(chunk)
    # Dataset single-file endpoints may return a one-file ZIP.
    if zipfile.is_zipfile(tmp) and not target.name.endswith('.pt'):
        with zipfile.ZipFile(tmp) as z:
            names = z.namelist()
            assert len(names) == 1 and Path(names[0]).name == target.name
            with z.open(names[0]) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        tmp.unlink()
    else:
        tmp.replace(target)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    # Use a separate unwrapped client for dataset operations.
    dataset_api = KaggleApi(); dataset_api.authenticate()
    version = None
    for page in range(1, 100):
        datasets = dataset_api.dataset_list(mine=True, page=page) or []
        found = next((d for d in datasets if d.ref == LIVE), None)
        if found:
            assert found.is_private
            version = found.current_version_number
            break
        if not datasets:
            break
    assert version is not None
    pin_client(api, KERNEL)
    assert api.kernels_status(KERNEL).status.name == 'COMPLETE'
    token = None; kernel_files = []; kernel_log = None
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = 'thisarivson'; req.kernel_slug = 'nlp-hw02-kaggle-l2'
        req.version_label = 'v2'; req.page_size = 200
        if token: req.page_token = token
        with api.build_kaggle_client() as c:
            response = c.kernels.kernels_api_client.list_kernel_session_output(req)
        kernel_files.extend(response.files or [])
        kernel_log = kernel_log or response.log
        token = response.next_page_token
        if not token: break
    (OUT/'nlp-hw02-kaggle-l2.log').write_text(kernel_log or '')
    token = None; live_files = []
    while True:
        response = dataset_api.dataset_list_files(LIVE+'/'+str(version), page_token=token, page_size=200)
        live_files.extend(response.files or [])
        token = response.next_page_token
        if not token: break
    live_dir = OUT/('live-v'+str(version))
    jobs = [('kernel', f.file_name, f.url) for f in kernel_files if wanted(f.file_name)]
    print(f'Pinned kernel v2: {len(jobs)}/{len(kernel_files)} selected; live v{version}: {len(live_files)} files', flush=True)

    def download(job):
        source, name, url = job
        target = (OUT if source == 'kernel' else live_dir)/safe(name)
        if not target.exists():
            for attempt in range(5):
                try:
                    if source == 'kernel':
                        with requests.get(url, stream=True, timeout=(30, 120)) as r:
                            transfer(r, target)
                    else:
                        req = ApiDownloadDatasetRequest()
                        req.owner_slug = 'thisarivson'; req.dataset_slug = 'nlp-hw02-l2-live'
                        req.dataset_version_number = version; req.file_name = name
                        with dataset_api.build_kaggle_client() as c:
                            r = c.datasets.dataset_api_client.download_dataset(req)
                            try: transfer(r, target)
                            finally: r.close()
                    break
                except Exception as e:
                    if attempt == 4:
                        status = getattr(getattr(e, 'response', None), 'status_code', None)
                        raise RuntimeError(f'{source}/{name}: {type(e).__name__}, HTTP {status}') from None
                    time.sleep(3 * 2**attempt)
        return {'source':source,'file':str(target.relative_to(OUT)),
                'bytes':target.stat().st_size,'sha256':sha(target)}

    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, r in enumerate(pool.map(download, jobs), 1):
            receipts.append(r)
            if i % 40 == 0: print('kernel files', i, flush=True)
    live_jobs = []
    for f in live_files:
        if not wanted(f.name, live=True): continue
        if '__weights__' in f.name:
            # Same producer artifact: hard link, then validate both producer receipts below.
            parts = f.name.split('__'); kernel = OUT/parts[0]/'weights'/parts[-1]
            assert kernel.is_file() and kernel.stat().st_size == f.total_bytes
            target = live_dir/safe(f.name); target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): os.link(kernel, target)
            receipts.append({'source':'live','file':str(target.relative_to(OUT)),
                             'bytes':target.stat().st_size,'sha256':sha(target),
                             'reused_kernel_artifact':str(kernel.relative_to(OUT))})
        else: live_jobs.append(('live', f.name, None))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i, r in enumerate(pool.map(download, live_jobs), 1):
            receipts.append(r)
            if i % 40 == 0: print('live files', i, flush=True)
    log = OUT/'nlp-hw02-kaggle-l2.log'
    receipts.append({'source':'kernel','file':log.name,'bytes':log.stat().st_size,'sha256':sha(log)})
    inventory = {'kernel':[f.file_name for f in kernel_files],
                 'live':[{'file':f.name,'bytes':f.total_bytes} for f in live_files]}
    (OUT/'remote-inventory.json').write_text(json.dumps(inventory,indent=2))
    # All FP16 weights must agree with point checkpoint and evaluation SHA.
    verified = []
    for p in sorted(OUT.glob('model-C-*/weights/*.pt')):
        model = p.parent.parent.name; tag = p.name.removesuffix('-fp16.pt')
        cp = json.loads((OUT/model/'points'/tag/'checkpoint.json').read_text())
        metrics = json.loads((OUT/model/'points'/tag/'evaluation/metrics.json').read_text())
        actual = sha(p)
        assert actual == cp['sha256'] == metrics['weights_sha256']
        assert p.stat().st_size == cp['bytes']
        if tag == 'l2-28800':
            live_cp = json.loads((live_dir/(model+'__points__'+tag+'__checkpoint.json')).read_text())
            assert live_cp['sha256'] == actual and live_cp['bytes'] == p.stat().st_size
        verified.append({'file':str(p.relative_to(OUT)), 'sha256':actual, 'bytes':p.stat().st_size})
    receipt = {'downloaded_at':datetime.now(timezone.utc).isoformat(), 'kernel_ref':KERNEL,
               'live_ref':LIVE+'/'+str(version), 'local_directory_label':'1 (requested; server kernel version is 2)',
               'full_states_downloaded':False,'files':receipts,'verified_weights':verified,
               'bytes_logical':sum(r['bytes'] for r in receipts),
               'bytes_excluding_hardlinks':sum(r['bytes'] for r in receipts if 'reused_kernel_artifact' not in r),
               'omitted':{'kernel':[f.file_name for f in kernel_files if not wanted(f.file_name)],
                          'live':[f.name for f in live_files if not wanted(f.name,live=True)]}}
    (OUT/'download-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('DONE',len(receipts),'files;',receipt['bytes_excluding_hardlinks'],'bytes; verified weights',len(verified),flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as e:
        print('Stopped:',type(e).__name__,str(e) if isinstance(e,RuntimeError) else '',file=sys.stderr)
        sys.exit(1)
