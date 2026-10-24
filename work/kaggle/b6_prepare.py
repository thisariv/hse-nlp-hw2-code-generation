"""Read-only B5 downloads, pinned provenance, real CPU replay and local PT cleanup."""
import argparse, contextlib, csv, io, json, os, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from b5_prepare import B5Client
from build_l3 import ROOT, write
from kaggle_io import sha

SOURCE = 'friend1/nlp-hw02-b5/6'
SNAPSHOT = ROOT.parent/'results/kaggle-runs/nlp-hw02-b5/1'


class ReadClient(B5Client):
    """One bounded output inventory per source, streamed blobs, no signed URLs in receipts."""
    def __init__(self):
        super().__init__(cli=shutil.which('kaggle'))
        for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(k,None)
        os.environ['KAGGLE_CONFIG_DIR']=self.env('friend1')['KAGGLE_CONFIG_DIR']
        from kaggle.api.kaggle_api_extended import KaggleApi
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.api=KaggleApi();self.api.authenticate()
        self.inventories={}

    def kernel_file(self,source,name,folder,owner=None):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        import requests
        if source not in self.inventories:
            user,slug,*version=source.split('/');files={}
            # Some completed kernel versions return an empty version-specific listing.
            # Latest fallback is always bound below to archived producer SHA/bytes.
            for label in ([('v'+version[0]),None] if version else [None]):
                token=None;seen=set()
                for _ in range(100):
                    req=ApiListKernelSessionOutputRequest();req.user_name=user;req.kernel_slug=slug;req.page_size=200
                    if label:req.version_label=label
                    if token:req.page_token=token
                    with self.api.build_kaggle_client() as c:r=c.kernels.kernels_api_client.list_kernel_session_output(req)
                    for item in r.files or []:files[item.file_name]=item.url
                    token=r.next_page_token
                    if not token:break
                    assert token not in seen,'Repeated output pagination token';seen.add(token)
                else:raise RuntimeError('Unbounded output inventory')
                if files:break
            self.inventories[source]=files
        assert name in self.inventories[source], 'Pinned output blob absent: '+name
        path=Path(folder)/name;path.parent.mkdir(parents=True,exist_ok=True)
        print('B6 download',name,flush=True)
        try:
            with requests.get(self.inventories[source][name],stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                with path.open('wb') as f:
                    for chunk in response.iter_content(4*1024**2):f.write(chunk)
        except Exception:
            path.unlink(missing_ok=True);raise RuntimeError('Pinned output transfer failed: '+name) from None
        return path


def planning():
    """Recompute rates from every actual B5 main/final update, not rounded prose."""
    pins = dict(source=SOURCE, source_manifest_sha256=sha(SNAPSHOT/'b1-output-manifest.json'),
                cache_source='thisarivson/nlp-hw02-compact-cache-verified/2',
                starter_source='thisarivson/nlp-hw02-26b-inputs', anchors={}, finals={}, budget={},predecessors={})
    inventory=json.loads((SNAPSHOT/'b1-output-manifest.json').read_text())['files']
    for g, tag in [('0', 'b5-final-125641'), ('1', 'b5-final-113690')]:
        root = SNAPSHOT/f'model-GPU{g}'
        pin = json.loads((root/'latest.json').read_text())
        final = root/'points'/tag/'complete.json'
        tr = json.loads(final.read_text())['training']
        assert all(tr['anchor'][k] == pin[k] for k in ('updates', 'cursor', 'main_seconds', 'identity', 'model_spec'))
        paths = [root/'updates.jsonl', root/'points'/tag/'cooldown-updates.jsonl']
        rates = {}
        for label, path in zip(('main', 'final50'), paths):
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            seconds = sum(r['seconds'] for r in rows)
            rates[label] = dict(updates=len(rows), seconds=seconds, seconds_per_update=seconds/len(rows),
                                updates_per_second=len(rows)/seconds, log_sha256=sha(path))
        s = max(r['seconds_per_update'] for r in rates.values())
        n = int(13200/s//100)*100
        pins['anchors'][g] = pin
        previous=next(r for r in inventory if r['file'].startswith(f'model-GPU{g}/checkpoints/') and r['sha256']!=pin['sha256'])
        pins['predecessors'][g]={**pin,**previous,'file':previous['file'].split('/',1)[1]}
        pins['finals'][g] = dict(tag=tag, anchor=tr['anchor'], complete_sha256=sha(final))
        pins['budget'][g] = dict(N=n, peak_lr=tr['start_lr'], rates=rates, reference_seconds_per_update=s,
            update_seconds=n*s, safety_factor=1.10, setup_seconds=480, eval_seconds=1800, io_seconds=600,
            expected_wall_seconds=1.10*n*s+2880, hard_wall_seconds=19200,
            basis='max(all B5 main mean, B5 final50 mean); 13200 update seconds; round down to 100; >=10% overhead')
        assert pins['budget'][g]['expected_wall_seconds'] < 18900
    write(ROOT/'b6-anchors.json', pins)
    return pins


def prepare(cpu_python, cache):
    pins = planning(); c = ReadClient(); checks = ROOT.parent/'checks'
    # Metadata is fetched by the SDK; credentials remain opaque to this program.
    code = '''import contextlib,io,json
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
with contextlib.redirect_stdout(io.StringIO()):
 a=KaggleApi();a.authenticate()
r=ApiGetKernelRequest();r.user_name='friend1';r.kernel_slug='nlp-hw02-b5';r.version_label='v6'
with a.build_kaggle_client() as c:d=c.kernels.kernels_api_client.get_kernel(r)
assert d.metadata.is_private and d.metadata.current_version_number==6
print(json.dumps(dict(ref='friend1/nlp-hw02-b5/6',private=True,current_version=6)))
'''
    metadata = subprocess.run([sys.executable, '-c', code], env=c.env('friend1'), capture_output=True, text=True)
    assert metadata.returncode==0, 'Read-only source metadata check failed; SDK diagnostic suppressed'
    remote = json.loads(metadata.stdout)
    downloads = []
    with tempfile.TemporaryDirectory(prefix='b6-real-inputs-') as tmp:
        tmp = Path(tmp)
        def fetch(name):
            # CLI latest output endpoint is bound to v6 by exact manifest SHA + per-file SHA.
            p = c.kernel_file(SOURCE, name, tmp, owner='friend1')
            if name == 'b1-output-manifest.json': assert sha(p) == pins['source_manifest_sha256']
            else:
                rec = inventory[name]
                assert p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256'], name
            downloads.append(dict(file=name, bytes=p.stat().st_size, sha256=sha(p)))
            return p
        manifest = fetch('b1-output-manifest.json')
        inventory = {r['file']:r for r in json.loads(manifest.read_text())['files']}
        manifest.unlink()
        for g in ('0', '1'):
            pin = pins['anchors'][g]; prefix = f'model-GPU{g}/'
            p = fetch(prefix+'latest.json'); assert json.loads(p.read_text()) == pin; p.unlink()
            tag = pins['finals'][g]['tag']
            p = fetch(prefix+'points/'+tag+'/complete.json')
            assert sha(p) == pins['finals'][g]['complete_sha256']; p.unlink()
            order = tmp/f'order-{g}'; order.mkdir()
            for name in ('order.json', 'order.npy'):
                fetch(f'weighted-order-GPU{g}/'+name).replace(order/name)
            subprocess.run([str(cpu_python), str(ROOT/'b6_cpu_resume_check.py'), '--gpu', g, '--capture-order',
                '--cache', str(cache), '--order-root', str(order), '--out', str(checks/f'b6-order-gpu{g}.json')], check=True)
            shutil.rmtree(order)
            # Final's pre-cooldown file was pruned by B5's two-generation rotation.
            # Check the retained post-restore predecessor, and exact final anchor metadata.
            previous=pins['predecessors'][g]
            predecessor=fetch(prefix+previous['file'])
            subprocess.run([str(cpu_python),str(ROOT/'b6_cpu_resume_check.py'),'--gpu',g,'--fingerprint-only',
                '--file',str(predecessor),'--out',str(checks/f'b6-final-anchor-gpu{g}.json')],check=True)
            predecessor.unlink()
            stable = fetch(prefix+pin['file'])
            try:
                subprocess.run([str(cpu_python), str(ROOT/'b6_cpu_resume_check.py'), '--gpu', g, '--file', str(stable),
                    '--cache', str(cache), '--order-receipt', str(checks/f'b6-order-gpu{g}.json'),
                    '--out', str(checks/f'b6-real-resume-gpu{g}.json')], check=True)
            finally: stable.unlink(missing_ok=True)
        # Actual friend blob access to both remaining mounted inputs.
        p = c.kernel_file(pins['cache_source'], 'compact-cache/compact-manifest.json', tmp, owner='friend1')
        assert sha(p) == pins['anchors']['0']['identity']['cache_manifest_sha256']; p.unlink()
        p = c.dataset_file(pins['starter_source'], '26b-training-inputs-d0-initial-and-evaluation/manifest.json', tmp)
        starter_sha = sha(p); p.unlink()
        assert not list(tmp.rglob('*.pt')) and not list(tmp.rglob('*.npy'))
    q = subprocess.run([c.cli, 'quota', '--csv'], env=c.env('friend1'), capture_output=True, text=True, check=True).stdout
    gpu = next(r for r in csv.DictReader(io.StringIO(q)) if r['resource'] == 'GPU')
    write(checks/'b6-inputs.json', dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        owner='friend1', source=remote, source_manifest_sha256=pins['source_manifest_sha256'], downloads=downloads,
        version_binding='current private v6 + exact archived manifest SHA + every downloaded blob SHA/bytes',
        local_payload_deleted=True, shared_cache_blob_verified=True, starter_manifest_sha256=starter_sha,
        quota_CLI=q, free_quota_seconds=float(gpu['remaining'].removesuffix('h'))*3600,
        quota_reset_utc=gpu['refreshAt'], GPU_runs=0, pushes=0))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cpu-python',type=Path); p.add_argument('--cache',type=Path)
    p.add_argument('--planning-only',action='store_true'); a=p.parse_args()
    if a.planning_only: planning()
    else:
        assert a.cpu_python and a.cache
        prepare(a.cpu_python,a.cache)
