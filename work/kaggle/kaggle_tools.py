#!/usr/bin/env python3
"""Private-only Kaggle uploads/status/output. S1 executes ONLY via explicit push --run-gpu."""
import argparse,json,pathlib,shutil,subprocess,sys,tarfile,re,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha,write,place,extract,find_one,safe_name
WORK=ROOT.parent
CACHE_SHA='2c691fef64fea8b4186b8acae54561d5620080286196a4390a23108776e80add'
ORDER_SHA='737dac5a146ad15bb0ec70cb06fea59493277e4332dea440b2bdf2a6316c1637'
CP_SHA='8400c8e59da764c65fbcb574758acc72e8e034aedeb59b10bf979465f110ab5e'

def call(args, capture=False):
    r=subprocess.run(['kaggle',*map(str,args)],check=True,text=True,capture_output=capture,timeout=600)
    return r.stdout if capture else None

def kaggle_python():
    executable=pathlib.Path(shutil.which('kaggle'))
    # Only read the public launcher shebang, never configuration/credential files.
    with executable.open() as f: interpreter=f.readline().strip()[2:]
    return interpreter

def probe(ref):
    return json.loads(subprocess.check_output([kaggle_python(),str(ROOT/'api_probe.py'),ref],text=True,timeout=60))

def versioned(command,ref,path=None):
    args=[kaggle_python(),str(ROOT/'kaggle_versioned.py'),command,ref]
    if path is not None:args+=['--path',str(path)]
    subprocess.run(args,check=True,timeout=600)

def stage_drive(downloads,stage,unpacked_only=False):
    downloads,stage=pathlib.Path(downloads).expanduser(),pathlib.Path(stage)
    if not downloads.exists(): raise FileNotFoundError('Drive download folder not ready')
    merged=stage/'drive-merged';merged.mkdir(parents=True,exist_ok=True)
    if not unpacked_only:
        for p in sorted(downloads.rglob('*.zip')): extract(p,merged)
    for folder in ('26-token-cache','26-wsd-data-scaling'):
        for p in (downloads/folder).rglob('*'):
            if p.is_file() and p.suffix!='.zip': place(p,merged/p.relative_to(downloads))
    # Ignore archival experiment metadata cache-manifest in26b: find the actual cache's shard tree.
    candidates=[p.parent for p in merged.rglob('cache-manifest.json') if (p.parent/'shards').exists()]
    if len(candidates)!=1: raise ValueError('Cannot identify one complete token-cache folder')
    cache=candidates[0];run=find_one(merged,'session-stop-055625-8feec45530.pt').parent.parent
    # Colab stores order in a subfolder; the Kaggle adapter consumes the same
    # verified bytes at the run root. Normalize only our staging copy.
    for name in ('order.npy','order.json'):
        nested=run/'order'/name
        if nested.is_file(): place(nested,run/name)
    validate_drive(cache,run)
    write(stage/'drive-paths.json',{'cache':str(cache.resolve()),'run':str(run.resolve())})
    print('Verified Drive inputs:',stage/'drive-paths.json')
    return cache,run

def validate_drive(cache,run):
    if sha(cache/'cache-manifest.json')!=CACHE_SHA: raise ValueError('Cache manifest SHA')
    m=json.loads((cache/'cache-manifest.json').read_text())
    if not m['sealed'] or m['rows']!=7078471 or [s['shard'] for s in m['shards']]!=list(range(42)):
        raise ValueError('Wrong cache row/shard set')
    for s in m['shards']:
        for e in s['files']:
            p=cache/safe_name(s['folder'])/safe_name(e['file'])
            if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']: raise ValueError('Cache array SHA: '+str(p))
    if sha(run/'order.npy')!=ORDER_SHA: raise ValueError('Order SHA')
    order=json.loads((run/'order.json').read_text())
    if order['file']['sha256']!=ORDER_SHA or (run/'order.npy').stat().st_size!=order['file']['bytes']:
        raise ValueError('Order receipt SHA/bytes')
    cp=run/'checkpoints/session-stop-055625-8feec45530.pt'
    if cp.stat().st_size!=145323979 or sha(cp)!=CP_SHA: raise ValueError('Resume checkpoint SHA')
    r=json.loads((cp.with_suffix('.pt.json')).read_text())
    latest=json.loads((run/'latest.json').read_text())
    if r!=latest or r['main_updates']!=46875 or r['phase']!='main': raise ValueError('Latest receipt mismatch')
    if r['identity']['cache_manifest_sha256']!=CACHE_SHA or r['identity']['order_sha256']!=ORDER_SHA:
        raise ValueError('Checkpoint input identity')
    # Do not upload old large states accidentally returned by Drive.
    if list((run/'checkpoints').glob('*.pt'))!=[cp]: raise ValueError('Only the designated session-stop PT is allowed')
    print('SHA verified: cache arrays, manifest, order, resume PT/receipts')

def dataset(folder,ref,title,message,selected=None):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-z0-9-]+',ref): raise ValueError('Invalid dataset reference')
    folder=pathlib.Path(folder).expanduser().resolve()
    if not folder.is_dir(): raise ValueError('Dataset source must be a folder')
    info=probe(ref)
    if info.get('exists') and not info.get('private'): raise ValueError('Refusing a public dataset')
    stage=ROOT/'staging'/'datasets'/ref.split('/')[1];stage.mkdir(parents=True,exist_ok=True)
    write(stage/'dataset-metadata.json',{'id':ref,'title':title,'licenses':[{'name':'other'}],
        'description':'Private university NLP homework inputs. Original authors retain rights; no redistribution license granted.'})
    files=selected if selected is not None else [p for p in sorted(folder.rglob('*')) if p.is_file()]
    if not files: raise ValueError('Empty dataset')
    with tarfile.open(stage/'payload.tar','w') as t:
        for p in files:
            relative=p.relative_to(folder);safe_name(relative.as_posix())
            if p.is_symlink(): raise ValueError('No symlink uploads')
            t.add(p,arcname=relative.as_posix(),recursive=False)
    if info.get('exists'): call(['datasets','version','-p',stage,'-m',message,'--keep-tabular'])
    else: call(['datasets','create','-p',stage,'--keep-tabular']) # private by default; --public absent
    deadline=time.monotonic()+3600
    while True:
        info=probe(ref)
        if info.get('exists') and not info['private']: raise RuntimeError('Server privacy check failed')
        # New private datasets may return 403 from status until they appear
        # in --mine. Wait for visibility instead of retrying the upload.
        state=call(['datasets','status',ref],capture=True).strip().lower() if info.get('exists') else 'not visible in --mine'
        if state=='ready' and info.get('exists'):break
        if state=='error' or time.monotonic()>=deadline:raise RuntimeError('Dataset not ready: '+state)
        print('Dataset processing:',ref,state,flush=True);time.sleep(max(0,min(90,deadline-time.monotonic())))
    print('https://www.kaggle.com/datasets/'+ref)
    write(stage/'upload-receipt.json',dict(info,ref=ref,status=state,source_files=len(files),payload_sha256=sha(stage/'payload.tar')))

def push(folder,run_gpu=False):
    folder=pathlib.Path(folder);m=json.loads((folder/'kernel-metadata.json').read_text())
    if m['is_private'] is not True or m['enable_internet'] is not True: raise ValueError('Private/internet metadata required')
    limits=folder/'session-limits.json'
    timeout=json.loads(limits.read_text())['session_timeout_seconds'] if limits.exists() else 43200
    ceiling=43200 if limits.exists() and json.loads(limits.read_text()).get('kind')=='l1' else 13200
    if limits.exists() and not 0<timeout<=ceiling:raise ValueError('Session exceeds registered backend ceiling')
    args=['kernels','push','-p',folder,'--timeout',str(timeout)]
    if m['enable_gpu']:
        if not run_gpu: raise ValueError('GPU launch is Phase B: explicitly pass --run-gpu')
        if m['machine_shape']!='NvidiaTeslaT4': raise ValueError('Require T4 x2')
        args+=['--accelerator','NvidiaTeslaT4']
    elif m.get('enable_tpu') or m.get('machine_shape'): raise ValueError('CPU metadata must disable accelerators')
    for ref in m.get('dataset_sources',[]):
        d=probe('/'.join(ref.split('/')[:2]))
        if not d.get('exists') or not d.get('private'):raise ValueError('Require existing private input dataset: '+ref)
    for ref in m.get('kernel_sources',[]):
        path=ROOT/'staging'/'source-metadata'/ref.replace('/','-');path.mkdir(parents=True,exist_ok=True)
        versioned('pull',ref,path) if len(ref.split('/'))==3 else call(['kernels','pull',ref,'-p',path,'-m'])
        previous=json.loads((path/'kernel-metadata.json').read_text())
        if str(previous['is_private']).lower()!='true':raise ValueError('Previous kernel source must be private')
    call(args);print('https://www.kaggle.com/code/'+m['id'])

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('api-check')
    d=s.add_parser('prepare-drive');d.add_argument('--downloads',default='~/Downloads/hw02-drive');d.add_argument('--stage',type=pathlib.Path,default=ROOT/'staging');d.add_argument('--unpacked-only',action='store_true',help='Use extracted folders only; do not read ZIP files')
    d=s.add_parser('upload-drive');d.add_argument('--owner',default='thisarivson');d.add_argument('--stage',type=pathlib.Path,default=ROOT/'staging')
    d=s.add_parser('dataset');d.add_argument('--folder',required=True);d.add_argument('--id',required=True);d.add_argument('--title',required=True);d.add_argument('--message',default='Verified input update')
    d=s.add_parser('push');d.add_argument('--folder',required=True);d.add_argument('--run-gpu',action='store_true')
    d=s.add_parser('status');d.add_argument('ref');d.add_argument('--dataset',action='store_true')
    d=s.add_parser('output');d.add_argument('ref');d.add_argument('--version',required=True,type=int)
    a=p.parse_args()
    if a.command=='api-check':
        call(['datasets','list','--mine']);call(['kernels','list','--mine'])
    elif a.command=='prepare-drive':stage_drive(a.downloads,a.stage,a.unpacked_only)
    elif a.command=='upload-drive':
        d=json.loads((a.stage/'drive-paths.json').read_text());validate_drive(pathlib.Path(d['cache']),pathlib.Path(d['run']))
        cache=pathlib.Path(d['cache']);manifest=json.loads((cache/'cache-manifest.json').read_text())
        files=[cache/'cache-manifest.json']
        for row in manifest['shards']:
            files.extend(cache/row['folder']/e['file'] for e in row['files'])
            rp=cache/row['folder']/'manifest.json'
            if rp.exists():files.append(rp)
        dataset(cache,a.owner+'/nlp-hw02-26-token-cache','NLP HW02 26 token cache','Verified cache 0-41',files)
        dataset(d['run'],a.owner+'/nlp-hw02-26-resume','NLP HW02 26 resume','Verified main46875 resume')
    elif a.command=='dataset':dataset(a.folder,a.id,a.title,a.message)
    elif a.command=='push':push(a.folder,a.run_gpu)
    elif a.command=='status':
        if not a.dataset and len(a.ref.split('/'))==3:versioned('status',a.ref)
        else:call(['datasets' if a.dataset else 'kernels','status',a.ref])
    elif a.command=='output':
        if a.version<1 or not re.fullmatch(r'[\w-]+/[\w-]+',a.ref): raise ValueError('Invalid version/ref')
        out=WORK/'results/kaggle-runs'/a.ref.split('/')[1]/str(a.version);out.mkdir(parents=True,exist_ok=True)
        versioned('output',a.ref+'/'+str(a.version),out);print(out)
if __name__=='__main__':main()
