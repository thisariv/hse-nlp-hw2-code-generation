#!/usr/bin/env python3
"""Pinned kernel output -> new private Dataset of another account; bounded CLI, no credential reads.

Only latest full states and the actual order are downloaded. Default mode deletes
staging after private readiness. --stream-states audits B2 on CPU and uploads/unlinks
each payload before the private commit; both modes verify remote files one at a time.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from kaggle_io import sha, write, safe_name

CONFIG_DIRS = dict(thisarivson='~/.kaggle', friend1='~/.kaggle-friend', friend2='~/.kaggle-friend2')


class Client:
    def __init__(self, cli=None): self.cli = cli or str(Path(sys.executable).with_name('kaggle'))

    def env(self, owner):
        env = dict(os.environ, KAGGLE_CONFIG_DIR=str(Path(CONFIG_DIRS[owner]).expanduser()))
        for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN'): env.pop(key, None)
        return env

    def call(self, args, owner, timeout=1800.):
        r = subprocess.run([self.cli, *args], env=self.env(owner), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        if r.returncode: raise RuntimeError('Kaggle CLI failed; return code '+str(r.returncode))

    def kernel_file(self, source, name, folder):
        # Official CLI2.2.4 ignores the parsed kernel version on this endpoint.
        # Our read-only CLI streams one file with an explicit SDK request.version_label.
        r = subprocess.run([sys.executable, str(Path(__file__).with_name('b1_output.py')),
                            '--source', source, '--name', name, '--path', str(folder)],
                           env=self.env(source.split('/')[0]), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=1800.)
        if r.returncode: raise RuntimeError('Pinned output download failed; return code '+str(r.returncode))
        path = folder/safe_name(name)
        # SDK/CLI releases may flatten filenames. Never guess among multiple candidates.
        if not path.is_file():
            hits = [p for p in folder.rglob(Path(name).name) if p.is_file()]
            assert len(hits) == 1, 'Requested output file missing or ambiguous'
            path = hits[0]
        return path

    def private_files(self, ref):
        # Read-only official SDK helper; auth is handled by SDK using this account's config.
        # Its only output is resource metadata. No credential value or CLI error is emitted.
        code = '''import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi();a.authenticate();ref=sys.argv[1];found=None
for page in range(1,101):
 rows=a.dataset_list(mine=True,page=page) or []
 found=next((r for r in rows if r.ref==ref),None)
 if found or not rows:break
if found:
 fs=a.dataset_list_files(ref,page_size=100).files
 print(json.dumps(dict(exists=True,private=bool(found.is_private),version=found.current_version_number,status=a.dataset_status(ref),
                      files={f.name:f.total_bytes for f in fs})))
else:print(json.dumps(dict(exists=False)))
'''
        r = subprocess.run([sys.executable, '-c', code, ref], env=self.env(ref.split('/')[0]),
                           capture_output=True, text=True, timeout=60.)
        if r.returncode: raise RuntimeError('Private Dataset metadata probe failed')
        return json.loads(r.stdout)

    def create(self, stage, owner):
        # No --public; create defaults to private. Never version an existing resource.
        self.call(['datasets', 'create', '-p', str(stage), '--quiet', '--keep-tabular', '--dir-mode', 'skip'], owner)

    def dataset_file(self, ref, name, folder):
        self.call(['datasets', 'download', '-d', ref, '-f', name, '-p', str(folder), '--unzip', '--force', '--quiet'],
                  ref.split('/')[0])
        path = folder/safe_name(name)
        # Kaggle's -f branch ignores --unzip and can return <name>.zip.
        if not path.is_file():
            archive = folder/(name+'.zip')
            assert archive.is_file(), 'Dataset single-file download missing'
            with zipfile.ZipFile(archive) as z:
                assert z.namelist() == [name], 'Unexpected Dataset download ZIP members'
                with z.open(name) as src, path.open('wb') as dst: shutil.copyfileobj(src, dst, 4*1024**2)
            archive.unlink()
        assert path.is_file(), 'Remote round-trip file missing'
        return path


def transfer(source, to_owner, dataset, receipt, client=None, temp_parent=None, manifest_sha=None, emergency_receipts=False):
    assert re.fullmatch(r'(thisarivson|friend1|friend2)/[a-z0-9-]+/[1-9][0-9]*', source), 'Pin kernel version'
    assert to_owner in CONFIG_DIRS and re.fullmatch(r'[a-z0-9-]+', dataset)
    client = client or Client(); ref = to_owner+'/'+dataset
    assert not client.private_files(ref)['exists'], 'Use a new Dataset slug for each transfer'
    with tempfile.TemporaryDirectory(prefix='b1-transfer-', dir=temp_parent) as folder:
        folder = Path(folder); stage = folder/'stage'; download = folder/'download'
        stage.mkdir(); download.mkdir(); entries = []
        if emergency_receipts:
            assert manifest_sha is None, 'Emergency mode uses pinned latest receipts, not a full manifest'
            # A hard watchdog exit can precede the supervisor's aggregate manifest.
            # Pinned version + atomic latest SHA/bytes still authenticate every full PT.
            source_entries = []
            identities = []
            for gpu in (0,1):
                name = f'model-GPU{gpu}/latest.json'
                path = client.kernel_file(source, name, download)
                latest = json.loads(path.read_text()); identities.append(latest['identity'])
                source_entries.append(dict(file=name, bytes=path.stat().st_size, sha256=sha(path)))
                source_entries.append(dict(file=f'model-GPU{gpu}/'+str(safe_name(latest['file'])),
                                           bytes=latest['bytes'], sha256=latest['sha256']))
            path = client.kernel_file(source, 'weighted-order/order.json', download)
            order = json.loads(path.read_text())
            assert order['format'] == 'nlp-hw02-b1-order-v1'
            assert all(i['cache_manifest_sha256'] == order['cache_manifest_sha256']
                       and i['order_sha256'] == order['file']['sha256'] for i in identities)
            source_entries += [dict(file='weighted-order/order.json', bytes=path.stat().st_size, sha256=sha(path)),
                               dict(order['file'], file='weighted-order/order.npy')]
            import hashlib
            source_hash = hashlib.sha256(json.dumps(source_entries, sort_keys=True).encode()).hexdigest()
        else:
            source_manifest = client.kernel_file(source, 'b1-output-manifest.json', download)
            if manifest_sha: assert sha(source_manifest) == manifest_sha, 'Output manifest SHA mismatch'
            source_hash = sha(source_manifest)
            source_entries = json.loads(source_manifest.read_text())['files']
        manifest = {r['file']:r for r in source_entries}
        assert len(manifest) == len(source_entries)
        def fetch(name):
            safe_name(name); expected = manifest[name]
            path = client.kernel_file(source, name, download)
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], 'Output SHA mismatch'
            flat = name.replace('/', '__'); target = stage/flat
            assert not target.exists()
            path.replace(target)
            entries.append(dict(file=flat, original=name, bytes=target.stat().st_size, sha256=sha(target)))
            return target
        for gpu in (0, 1):
            prefix = f'model-GPU{gpu}/'; latest = json.loads(fetch(prefix+'latest.json').read_text())
            assert latest['identity']['model'] in ((('E','W-lr1000'),('W','W-lr600'))[gpu])
            original = prefix+str(safe_name(latest['file']))
            assert original.startswith(prefix+'checkpoints/main-') and original.endswith('.pt')
            assert all(latest[k] == manifest[original][k] for k in ('bytes', 'sha256'))
            fetch(original)
        for name in ('weighted-order/order.json', 'weighted-order/order.npy'): fetch(name)
        transfer_manifest = dict(format='nlp-hw02-b1-transfer-v1', source=source, source_manifest_sha256=source_hash,
                                 destination=ref, files=entries, emergency_receipts=emergency_receipts)
        write(stage/'b1-transfer.json', transfer_manifest)
        transfer_hash = sha(stage/'b1-transfer.json')
        all_files = entries+[dict(file='b1-transfer.json', sha256=transfer_hash, bytes=(stage/'b1-transfer.json').stat().st_size)]
        write(stage/'dataset-metadata.json', dict(id=ref, title='NLP HW02 B1 stable transfer '+dataset,
              licenses=[dict(name='CC0-1.0')], isPrivate=True))
        client.create(stage, to_owner)
        remote = None
        for attempt in range(60):
            candidate = client.private_files(ref)
            if candidate.get('exists'):
                assert candidate.get('private') is True, 'Transferred Dataset is not private'
                if candidate.get('status') == 'ready' and all(candidate['files'].get(r['file']) == r['bytes'] for r in all_files):
                    remote = candidate; break
            if attempt < 59: time.sleep(2)
        assert remote is not None, 'Dataset not ready; retry verification, do not silently resume'
        # Remote payload is ready. Free the entire staging payload before the round-trip check.
        shutil.rmtree(stage); shutil.rmtree(download); download.mkdir()
        pinned = ref+'/'+str(remote['version'])
        for r in all_files:
            path = client.dataset_file(pinned, r['file'], download)
            assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'Remote SHA mismatch'
            path.unlink()
        result = dict(passed=True, source=source, dataset=ref, version=remote['version'], private=True,
                      transfer_sha256=transfer_hash, files=all_files, remote_SHA_verified=True,
                      local_payload_deleted=True, minimum_payload_bytes=sum(r['bytes'] for r in all_files))
        write(receipt, result)
        return result


def transfer_ev(source, to_owner, dataset, receipt, client=None, temp_parent=None):
    """One output payload at a time: verify, upload blob, delete; private commit, remote SHA.

    Accumulated SDK file descriptors remain in memory, never credentials or upload tokens on disk.
    Full-state transfer() remains unchanged for B1/B2 continuation.
    """
    assert re.fullmatch(r'friend2/[a-z0-9-]+/[1-9][0-9]*', source)
    assert to_owner == 'thisarivson' and re.fullmatch(r'[a-z0-9-]+', dataset)
    client = client or Client(); ref = to_owner+'/'+dataset
    assert not client.private_files(ref)['exists'], 'Use a new private Dataset slug'
    # Only the destination SDK runs in this process; source downloads use a separate config process.
    os.environ.update(KAGGLE_CONFIG_DIR=str(Path(CONFIG_DIRS[to_owner]).expanduser()))
    for key in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'): os.environ.pop(key, None)
    import contextlib, io
    from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadContext, ApiBlobType
    from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api = KaggleApi(); api.authenticate()
    request = ApiCreateDatasetRequest(); request.title = 'NLP HW02 B1 EV inputs'
    request.slug = dataset; request.owner_slug = to_owner; request.license_name = 'CC0-1.0'
    request.is_private = True; request.files = []
    with tempfile.TemporaryDirectory(prefix='b1-ev-transfer-', dir=temp_parent) as folder:
        folder = Path(folder); download = folder/'download'; stage = folder/'stage'
        download.mkdir(); stage.mkdir(); entries = []; latest = []
        manifest_path = client.kernel_file(source, 'b1-output-manifest.json', download)
        manifest_hash = sha(manifest_path)
        inventory = {r['file']:r for r in json.loads(manifest_path.read_text())['files']}
        def fetched(name):
            path = client.kernel_file(source, name, download); r = inventory[name]
            assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'Source SHA mismatch'
            return path
        for gpu in (0,1):
            r = json.loads(fetched(f'model-GPU{gpu}/latest.json').read_text())
            assert r['identity']['model'] == ('E','W')[gpu] and 34200 <= r['main_seconds'] < 34205
            latest.append(r)
        names = ['weighted-order/order.json','weighted-order/order.npy']
        for gpu in (0,1):
            prefix = f'model-GPU{gpu}/'
            names += [prefix+'latest.json', prefix+str(safe_name(latest[gpu]['file'])),
                      prefix+'weights/probe-028370-fp16.pt', prefix+'points/probe-028370/checkpoint.json']
        with ResumableUploadContext() as upload:
            def put(path):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    api.upload_files(request, None, str(stage), ApiBlobType.DATASET, upload, True, 'skip', None)
                path.unlink()
            for name in names:
                path = fetched(name); flat = name.replace('/', '__'); target = stage/flat
                path.replace(target); r = inventory[name]
                entries.append(dict(file=flat, original=name, bytes=r['bytes'], sha256=r['sha256']))
                put(target)
                print('EV transfer uploaded and removed', flat, flush=True)
            m = dict(format='nlp-hw02-b1-transfer-v1', purpose='EV prepared FP16 + final full stable',
                     source=source, source_manifest_sha256=manifest_hash, destination=ref, files=entries)
            path = stage/'b1-transfer.json'; write(path, m)
            h = sha(path); files = entries+[dict(file=path.name, bytes=path.stat().st_size, sha256=h)]
            put(path)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with api.build_kaggle_client() as c:
                    response = api.with_retry(c.datasets.dataset_api_client.create_dataset)(request)
                assert not response.error, 'Dataset creation failed'
        remote = None
        for _ in range(90):
            r = client.private_files(ref)
            if r.get('exists'):
                assert r['private'] is True
                if r.get('status') == 'ready' and all(r['files'].get(f['file']) == f['bytes'] for f in files):
                    remote = r; break
            time.sleep(2)
        assert remote is not None, 'Dataset readiness not confirmed'
        shutil.rmtree(download); download.mkdir()
        for r in files:
            path = client.dataset_file(ref+'/'+str(remote['version']), r['file'], download)
            assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'Remote SHA mismatch'
            path.unlink(); print('EV transfer remote SHA passed', r['file'], flush=True)
        result = dict(passed=True, source=source, dataset=ref, version=remote['version'], private=True,
                      transfer_sha256=h, files=files, remote_SHA_verified=True, local_payload_deleted=True,
                      streaming_one_payload=True)
        write(receipt, result); return result


def transfer_states(source, to_owner, dataset, receipt, cpu_python, cache, client=None, temp_parent=None):
    """B3: one B2 payload on disk; source SHA -> CPU audit -> upload -> unlink -> remote SHA.

    Existing B1/B2 transfer behavior is deliberately unchanged. Never prints SDK errors/auth.
    """
    assert source == 'friend2/nlp-hw02-b2/1' and to_owner == 'thisarivson'
    assert dataset == 'nlp-hw02-b2-states'
    client = client or Client(); ref = to_owner+'/'+dataset
    assert not client.private_files(ref)['exists'], 'Dataset already exists; verify the saved receipt, never overwrite'
    root = Path(__file__).resolve().parent
    pins = json.loads((root/'b3-anchors.json').read_text())
    checks = root.parent/'checks'; checks.mkdir(exist_ok=True)
    os.environ.update(KAGGLE_CONFIG_DIR=str(Path(CONFIG_DIRS[to_owner]).expanduser()))
    for key in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'): os.environ.pop(key, None)
    import contextlib, io
    from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadContext, ApiBlobType
    from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        api = KaggleApi(); api.authenticate()
    request = ApiCreateDatasetRequest(); request.title = 'NLP HW02 B2 states'
    request.slug = dataset; request.owner_slug = to_owner; request.license_name = 'CC0-1.0'
    request.is_private = True; request.files = []
    with tempfile.TemporaryDirectory(prefix='b3-transfer-', dir=temp_parent) as folder:
        folder = Path(folder); download = folder/'download'; stage = folder/'stage'
        download.mkdir(); stage.mkdir(); entries = []
        path = client.kernel_file(source, 'b1-output-manifest.json', download)
        assert sha(path) == pins['source_manifest_sha256'], 'Pinned B2 manifest mismatch'
        inventory = {r['file']:r for r in json.loads(path.read_text())['files']}; path.unlink()
        names = ['weighted-order/order.npy', 'weighted-order/order.json']
        for gpu in (0,1):
            pin = pins['anchors'][str(gpu)]; prefix = f'model-GPU{gpu}/'
            names += [prefix+'latest.json', prefix+str(safe_name(pin['file']))]
        with ResumableUploadContext() as upload:
            def put(path):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    api.upload_files(request, None, str(stage), ApiBlobType.DATASET, upload, True, 'skip', None)
                path.unlink()
                assert not any(stage.iterdir()), 'More than one staged payload'
            for name in names:
                assert not any(p.is_file() for p in download.rglob('*'))
                path = client.kernel_file(source, name, download); r = inventory[name]
                assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'Source SHA mismatch'
                if name.endswith('latest.json'):
                    gpu = name.split('/')[0][-1]
                    assert json.loads(path.read_text()) == pins['anchors'][gpu], 'Wrong B2 latest'
                if name.endswith('order.npy'):
                    subprocess.run([str(cpu_python), str(root/'b3_cpu_resume_check.py'), '--capture-order', str(path),
                                    '--cache', str(cache), '--out', str(checks/'b3-order.json')], check=True)
                if name.endswith('.pt'):
                    gpu = name.split('/')[0][-1]
                    subprocess.run([str(cpu_python), str(root/'b3_cpu_resume_check.py'), '--file', str(path),
                                    '--gpu', gpu, '--cache', str(cache), '--order-receipt', str(checks/'b3-order.json'),
                                    '--out', str(checks/f'b3-real-resume-gpu{gpu}.json')], check=True)
                flat = name.replace('/', '__'); target = stage/flat; path.replace(target)
                entries.append(dict(file=flat, original=name, bytes=r['bytes'], sha256=r['sha256']))
                put(target); print('B3 uploaded and deleted', flat, flush=True)
            m = dict(format='nlp-hw02-b1-transfer-v1', purpose='B3 strict B2 continuation, both W-QK branches',
                     source=source, source_manifest_sha256=pins['source_manifest_sha256'], destination=ref, files=entries)
            path = stage/'b1-transfer.json'; write(path, m); h = sha(path)
            files = entries+[dict(file=path.name, bytes=path.stat().st_size, sha256=h)]
            put(path)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with api.build_kaggle_client() as c:
                    response = api.with_retry(c.datasets.dataset_api_client.create_dataset)(request)
                assert not response.error, 'Private Dataset creation failed'
        remote = None
        for _ in range(90):
            candidate = client.private_files(ref)
            if candidate.get('exists'):
                assert candidate['private'] is True
                if candidate.get('status') == 'ready' and all(candidate['files'].get(f['file']) == f['bytes'] for f in files):
                    remote = candidate; break
            time.sleep(2)
        assert remote is not None, 'Dataset readiness not confirmed; do not recreate'
        for r in files:
            path = client.dataset_file(ref+'/'+str(remote['version']), r['file'], download)
            assert path.stat().st_size == r['bytes'] and sha(path) == r['sha256'], 'Remote SHA mismatch'
            path.unlink(); print('B3 remote SHA passed', r['file'], flush=True)
        result = dict(passed=True, source=source, dataset=ref, version=remote['version'], private=True,
                      transfer_sha256=h, files=files, remote_SHA_verified=True, local_payload_deleted=True,
                      streaming_one_payload=True, real_CPU_checks=True)
        write(receipt, result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--to', choices=CONFIG_DIRS, required=True)
    p.add_argument('--dataset', required=True, help='New slug per transfer; never updates an existing Dataset')
    p.add_argument('--receipt', type=Path, required=True); p.add_argument('--temp-parent', type=Path)
    p.add_argument('--manifest-sha');p.add_argument('--emergency-receipts',action='store_true');p.add_argument('--ev-inputs',action='store_true')
    p.add_argument('--stream-states',action='store_true');p.add_argument('--cpu-python',type=Path);p.add_argument('--cache',type=Path); a = p.parse_args()
    if a.stream_states: assert a.cpu_python and a.cache and not a.ev_inputs and not a.emergency_receipts
    if a.stream_states:
        try: result=transfer_states(a.source,a.to,a.dataset,a.receipt,a.cpu_python,a.cache,temp_parent=a.temp_parent)
        except Exception as error:
            # SDK upload exceptions can contain signed URLs. Emit only the type.
            print('B3 state transfer failed:',type(error).__name__); raise SystemExit(1)
    else:
        result = (transfer_ev(a.source, a.to, a.dataset, a.receipt, temp_parent=a.temp_parent) if a.ev_inputs else
                  transfer(a.source, a.to, a.dataset, a.receipt, temp_parent=a.temp_parent, manifest_sha=a.manifest_sha, emergency_receipts=a.emergency_receipts))
    print(json.dumps(dict(dataset=result['dataset'], version=result['version'], transfer_sha256=result['transfer_sha256'], passed=True)))
