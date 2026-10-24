"""CLI-only access/transfer, real CPU replays, private remote SHA, automatic local cleanup.

No kernel pushes, Secrets or credential inspection. CLI/SDK alone authenticate.
"""
import argparse,json,re,shutil,subprocess,sys,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
from b1_transfer import Client
from build_l3 import ROOT,write
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha,safe_name


class B5Client(Client):
    def kernel_file(self,source,name,folder,owner=None):
        self.call(['kernels','output','/'.join(source.split('/')[:2]),'--file-pattern','^'+re.escape(name)+'$',
                   '--path',str(folder),'--force','--quiet','--page-size','200'],owner or source.split('/')[0])
        target=folder/safe_name(name)
        if not target.is_file():
            hits=[p for p in folder.rglob(Path(name).name) if p.is_file()];assert len(hits)==1
            target=hits[0]
        return target

    def dataset_file(self,ref,name,folder):
        import zipfile
        self.call(['datasets','download',ref,'-f',name,'-p',str(folder),'--force','--quiet'],'friend1')
        target=folder/safe_name(name)
        if target.is_file():return target
        hits=[p for p in folder.rglob('*') if p.is_file() and p.name in (Path(name).name,Path(name).name+'.zip')]
        assert len(hits)==1, 'Single dataset file missing or ambiguous'
        downloaded=hits[0];target.parent.mkdir(parents=True,exist_ok=True)
        if zipfile.is_zipfile(downloaded):
            with zipfile.ZipFile(downloaded) as z:
                assert len(z.namelist())==1 and Path(z.namelist()[0]).name==Path(name).name
                with z.open(z.namelist()[0]) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
            downloaded.unlink()
        else:downloaded.replace(target)
        return target


def prepare(cpu_python,cache):
    c=B5Client();pins=json.loads((ROOT/'b5-anchors.json').read_text());checks=ROOT.parent/'checks'
    receipts=[]
    with tempfile.TemporaryDirectory(prefix='b5-transfer-') as tmp:
        tmp=Path(tmp);download=tmp/'download';stage=tmp/'stage';download.mkdir();stage.mkdir()
        # Actual blob reads under friend config, not just a public metadata listing.
        path=c.kernel_file(pins['cache_source'],'compact-cache/compact-manifest.json',download,owner='friend1')
        assert sha(path)==pins['anchors']['0']['identity']['cache_manifest_sha256'];path.unlink()
        starter=c.dataset_file(pins['starter_source'],'26b-training-inputs-d0-initial-and-evaluation/manifest.json',download)
        print('B5 friend cache/starter CLI access verified',flush=True)
        receipts.append(dict(resource=pins['starter_source'],friend_CLI_download=True,manifest_sha256=sha(starter)))
        for p in list(download.rglob('*')):
            if p.is_file():p.unlink()
        ref=pins['transfer_dataset'];old=c.private_files(ref)
        if old.get('exists'):
            saved=json.loads((checks/'b5-transfer.json').read_text())
            assert saved['passed'] and saved['remote_SHA_verified'] and old['private']
            assert old['version']==saved['version'];print('B5 existing private transfer verified; no recreation',flush=True)
        else:
            print('B5 fetching pinned W manifest',flush=True)
            manifest=c.kernel_file(pins['sources']['0'],'b1-output-manifest.json',download)
            assert sha(manifest)==pins['W_source_manifest_sha256']
            inventory={r['file']:r for r in json.loads(manifest.read_text())['files']};manifest.unlink()
            entries=[];wpin=pins['anchors']['0']
            names=['weighted-order/order.json','weighted-order/order.npy','model-GPU0/latest.json','model-GPU0/'+wpin['file']]
            for name in names:
                print('B5 source fetch',name,flush=True)
                path=c.kernel_file(pins['sources']['0'],name,download);rec=inventory[name]
                assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
                if name.endswith('latest.json'):assert json.loads(path.read_text())==wpin
                flat=name.replace('/','__');target=stage/flat;path.replace(target)
                entries.append(dict(file=flat,original=name,bytes=rec['bytes'],sha256=rec['sha256']))
                if name.endswith('order.npy'):
                    order=tmp/'W-order';order.mkdir()
                    shutil.copyfile(target,order/'order.npy');shutil.copyfile(stage/'weighted-order__order.json',order/'order.json')
                    subprocess.run([str(cpu_python),str(ROOT/'b5_cpu_resume_check.py'),'--gpu','0','--capture-order',
                        '--cache',str(cache),'--order-root',str(order),'--out',str(checks/'b5-order-gpu0.json')],check=True)
                    shutil.rmtree(order)
                if name.endswith('.pt'):
                    assert rec['sha256']==wpin['sha256'] and rec['bytes']==wpin['bytes']
                    subprocess.run([str(cpu_python),str(ROOT/'b5_cpu_resume_check.py'),'--gpu','0','--file',str(target),
                        '--cache',str(cache),'--order-receipt',str(checks/'b5-order-gpu0.json'),
                        '--out',str(checks/'b5-real-resume-gpu0.json')],check=True)
            m=dict(format='nlp-hw02-b5-transfer-v1',source=pins['sources']['0'],source_manifest_sha256=pins['W_source_manifest_sha256'],
                   destination=ref,files=entries)
            write(stage/'b5-transfer.json',m);h=sha(stage/'b5-transfer.json')
            files=entries+[dict(file='b5-transfer.json',bytes=(stage/'b5-transfer.json').stat().st_size,sha256=h)]
            write(stage/'dataset-metadata.json',dict(id=ref,title='nlp-hw02-b5-inputs',isPrivate=True,licenses=[dict(name='CC0-1.0')]))
            c.create(stage,'friend1');shutil.rmtree(stage)
            remote=None
            for _ in range(90):
                r=c.private_files(ref)
                if r.get('exists'):
                    assert r['private'] is True
                    if r.get('status')=='ready' and all(r['files'].get(f['file'])==f['bytes'] for f in files):remote=r;break
                time.sleep(2)
            assert remote,'Private transfer not ready; do not recreate'
            for rec in files:
                path=c.dataset_file(ref+'/'+str(remote['version']),rec['file'],download)
                assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256'];path.unlink()
                print('B5 remote SHA verified',rec['file'],flush=True)
            write(checks/'b5-transfer.json',dict(passed=True,dataset=ref,version=remote['version'],private=True,
                transfer_sha256=h,source_manifest_sha256=pins['W_source_manifest_sha256'],files=files,
                remote_SHA_verified=True,local_payload_deleted=True,real_CPU_checks=True))
        # C-long2x uses the existing private L3 live, never P1 cooldown weights.
        source=pins['sources']['1'];pin=pins['anchors']['1']
        path=c.dataset_file(source,'recover-manifest.json',download);manifest=json.loads(path.read_text());mh=sha(path);path.unlink()
        entries={r['original_path']:r for r in manifest['files']}
        def live_fetch(original):
            rec=entries[original];path=c.dataset_file(source,rec['file'],download)
            assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
            return path,rec
        latest,lr=live_fetch('model-C-long2x/latest.json')
        import gzip
        raw=gzip.decompress(latest.read_bytes()) if lr['gzip'] else latest.read_bytes();actual=json.loads(raw);latest.unlink()
        assert dict(actual,file=pin['file'],original_file=actual['file'])==pin
        order=tmp/'C-order';order.mkdir()
        for name in ('order.json','order.npy'):
            path,rec=live_fetch('weighted-order/'+name)
            raw=gzip.decompress(path.read_bytes()) if rec['gzip'] else path.read_bytes()
            (order/name).write_bytes(raw);path.unlink()
        subprocess.run([str(cpu_python),str(ROOT/'b5_cpu_resume_check.py'),'--gpu','1','--capture-order',
            '--cache',str(cache),'--order-root',str(order),'--out',str(checks/'b5-order-gpu1.json')],check=True)
        shutil.rmtree(order)
        stable,rec=live_fetch('model-C-long2x/'+pin['original_file'])
        assert rec['sha256']==pin['sha256'] and rec['bytes']==pin['bytes']
        try:
            subprocess.run([str(cpu_python),str(ROOT/'b5_cpu_resume_check.py'),'--gpu','1','--file',str(stable),
                '--cache',str(cache),'--order-receipt',str(checks/'b5-order-gpu1.json'),
                '--out',str(checks/'b5-real-resume-gpu1.json')],check=True)
        finally:stable.unlink(missing_ok=True)
        receipts.append(dict(resource=source,friend_CLI_download=True,recover_manifest_sha256=mh,anchor_sha256=pin['sha256']))
        assert not list(tmp.rglob('*.pt'))
    # Machine-readable quota via CLI (no configuration file is opened by preparation code).
    import csv,io
    quota=subprocess.run([c.cli,'quota','--csv'],env=c.env('friend1'),capture_output=True,text=True,check=True).stdout
    rows=list(csv.DictReader(io.StringIO(quota)));gpu=next(r for r in rows if r['resource']=='GPU')
    free=float(gpu['remaining'].removesuffix('h'))*3600
    write(checks/'b5-inputs.json',dict(passed=True,checked_at_utc=datetime.now(timezone.utc).isoformat(),owner='friend1',
       receipts=receipts,shared_cache_CLI_blob_SHA_verified=True,cache_manifest_sha256=pins['anchors']['0']['identity']['cache_manifest_sha256'],
       GPU_runs=0,pushes=0,local_payload_deleted=True,quota_CLI=quota,quota_reset_utc='2026-10-24T00:00:00Z',
       free_quota_seconds=free))
    print('B5 private inputs, real resumes, remote SHA and cleanup passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu-python',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    try:prepare(a.cpu_python,a.cache)
    except Exception as e:
        import traceback
        print('B5 preparation failed:',type(e).__name__,[(Path(f.filename).name,f.lineno) for f in traceback.extract_tb(e.__traceback__)],flush=True)
        raise SystemExit(1)
