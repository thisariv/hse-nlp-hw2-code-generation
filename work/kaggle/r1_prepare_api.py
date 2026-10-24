#!/usr/bin/env python3
"""Private states: download one -> CPU verify -> upload blob -> delete -> next.

No kernels API. Never opens credential files. SDK authentication only.
The SDK's upload_files hook accumulates uploaded blob handles for ONE dataset
version, so both states are present remotely without coexisting on local disk.
"""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE='friend1/nlp-hw02-l3-live/12'
DEST='thisarivson/nlp-hw02-l3-states'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def download(api,name,folder):
    folder.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        api.dataset_download_file(SOURCE,name,path=str(folder),force=True,quiet=True)
    for p in folder.glob('*.zip'):
        with zipfile.ZipFile(p) as z:
            names=[n for n in z.namelist() if Path(n).name==name];assert len(names)==1
            with z.open(names[0]) as src,(folder/name).open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
        p.unlink()
    assert (folder/name).is_file()


def verify_states(api, retries=30):
    """Creation/listing is eventually consistent; validate without another upload."""
    pin=json.loads((ROOT/'r1-anchors.json').read_text())
    path=ROOT.parent/'checks/kaggle-r1-states-copy.json';receipt=json.loads(path.read_text())
    assert len(receipt['files'])==2 and all(r['local_deleted'] for r in receipt['files'])
    for attempt in range(retries):
        found=None
        for page in range(1,100):
            rows=api.dataset_list(mine=True,page=page) or []
            found=next((r for r in rows if r.ref==DEST),None)
            if found or not rows:break
        if found:
            assert found.is_private
            version=found.current_version_number
            if version:
                remote=api.dataset_list_files(DEST+'/'+str(version),page_size=100)
                sizes={f.name:f.total_bytes for f in remote.files}
                if all(sizes.get(p['file'])==p['bytes'] for p in pin['anchors'].values()):
                    receipt.update(passed=True,private=True,version=version,remote_file_bytes_verified=True,
                                   one_full_state_on_disk_at_a_time=True)
                    write(path,receipt);return receipt
        if attempt+1<retries:time.sleep(2)
    raise RuntimeError('Private copied state readiness pending; use verify-states, do not upload again')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['download','copy-states','verify-states','create-live'])
    p.add_argument('--file');p.add_argument('--path',type=Path,default=ROOT/'staging/r1-states')
    p.add_argument('--owner',choices=['friend1','thisarivson'],default='thisarivson')
    p.add_argument('--cpu-python',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--weighted-order',type=Path)
    a=p.parse_args()
    friend=str(Path('~/.kaggle-friend').expanduser())
    if a.command=='download':
        assert os.environ.get('KAGGLE_CONFIG_DIR')==friend
    elif a.owner=='thisarivson':assert os.environ.get('KAGGLE_CONFIG_DIR') in (None,str(Path('~/.kaggle').expanduser()))
    else:assert os.environ.get('KAGGLE_CONFIG_DIR')==friend
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    if a.command=='download':
        assert a.file and Path(a.file).name==a.file
        download(api,a.file,a.path);print('Downloaded',a.file);return
    if a.command=='verify-states':
        assert a.owner=='thisarivson'
        verify_states(api);print('Private states ready; remote bytes verified; no reupload');return
    if a.command=='create-live':
        ref=a.owner+'/nlp-hw02-r1-live';a.path.mkdir(parents=True,exist_ok=True)
        write(a.path/'dataset-metadata.json',dict(id=ref,title='NLP HW02 R1 live',licenses=[dict(name='other')]))
        write(a.path/'ready.json',dict(session='R1',owner=a.owner,status='ready; no training'))
        found=False
        for page in range(1,100):
            rows=api.dataset_list(mine=True,page=page) or []
            for row in rows:
                if row.ref==ref:assert row.is_private;found=True
            if found or not rows:break
        if not found:
            with contextlib.redirect_stdout(io.StringIO()):result=api.dataset_create_new(str(a.path),public=False,quiet=True,convert_to_csv=False)
            assert not result.error
        # This credential scope supports version-pinned file listings; unversioned
        # listing can return 403 even after successful private creation.
        files=api.dataset_list_files(ref+'/1',page_size=100)
        assert any(f.name=='ready.json' for f in files.files)
        write(ROOT.parent/f'checks/kaggle-r1-live-{a.owner}.json',dict(passed=True,ref=ref,private=True,already_existed=found,push=False))
        print('Private R1 live ready',a.owner);return
    assert a.owner=='thisarivson' and a.cpu_python and a.cache and a.weighted_order
    # Refuse to overwrite an existing prepared target on a retry.
    for page in range(1,100):
        rows=api.dataset_list(mine=True,page=page) or []
        for row in rows:
            if row.ref==DEST:
                assert row.is_private
                raise RuntimeError('Target exists; inspect receipt instead of reuploading')
        if not rows:break
    pins=json.loads((ROOT/'r1-anchors.json').read_text());stage=a.path;stage.mkdir(parents=True,exist_ok=True)
    assert not list(stage.glob('*.pt')), 'Remove only an inspected task-owned stale PT before retry'
    env=dict(os.environ,KAGGLE_CONFIG_DIR=friend)
    def get(name):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'download','--file',name,'--path',str(stage)],
                       env=env,check=True,timeout=600)
    get('recover-manifest.json');manifest=json.loads((stage/'recover-manifest.json').read_text())
    small=[e for e in manifest['files'] if e['original_path'].endswith('/latest.json') or e['original_path'].startswith('weighted-order/')]
    for entry in small:
        get(entry['file']);path=stage/entry['file']
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
    write(stage/'dataset-metadata.json',dict(id=DEST,title='NLP HW02 L3 full stable states',licenses=[dict(name='other')]))
    receipt=dict(passed=False,source=SOURCE,destination=DEST,private=True,files=[],GPU=False,push=False)
    # Override only this API object's upload callback. Authentication remains default.
    original_upload=api.upload_files
    def streaming_upload(request,resources,folder,blob_type,upload_context,quiet,dir_mode,ignore_patterns):
        original_upload(request,resources,folder,blob_type,upload_context,True,dir_mode,ignore_patterns)
        for name,pin in pins['anchors'].items():
            get(pin['file']);path=stage/pin['file']
            try:
                assert path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256']
                subprocess.run([str(a.cpu_python),str(ROOT/'r1_cpu_resume_check.py'),'--anchor-root',str(stage),
                                '--cache',str(a.cache),'--weighted-order',str(a.weighted_order),'--model',name,'--keep-anchor'],
                               check=True,timeout=900)
                one=stage/'one';one.mkdir(exist_ok=True);path.replace(one/path.name)
                try:original_upload(request,None,str(one),blob_type,upload_context,True,dir_mode,None)
                finally:(one/path.name).unlink(missing_ok=True);one.rmdir()
                receipt['files'].append(dict(model=name,file=pin['file'],bytes=pin['bytes'],sha256=pin['sha256'],local_deleted=True))
                write(ROOT.parent/'checks/kaggle-r1-states-copy.json',receipt)
                cpu_receipt=ROOT.parent/f'checks/kaggle-r1-real-resume-{name}.json'
                audit=json.loads(cpu_receipt.read_text());audit.update(downloaded_anchor_removed=True,deleted_after_stream_upload=True)
                write(cpu_receipt,audit)
            finally:path.unlink(missing_ok=True)
    api.upload_files=streaming_upload
    with contextlib.redirect_stdout(io.StringIO()):
        result=api.dataset_create_new(str(stage),public=False,quiet=True,convert_to_csv=False)
    assert not result.error
    verify_states(api)
    print('Private full L3 states copied; two CPU bit-exact resumes passed; local PT deleted')


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('R1 API FAILED',type(error).__name__,getattr(getattr(error,'response',None),'status_code',None));sys.exit(1)
