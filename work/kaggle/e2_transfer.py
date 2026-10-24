"""Copy only the two pinned B6 FP16 payloads to a new private dataset."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time
from b1_transfer import Client
from src.kaggle_io import sha, write

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REF = 'thisarivson/nlp-hw02-b6-fp16'
RECEIPT = WORK/'checks/e2-b6-fp16-transfer.json'
SOURCE = 'friend1/nlp-hw02-b6/1'
MANIFEST_SHA = 'dd00395d133cd5c005261bf39b41c9a967dbd36e9523b838ecfa865aae218dfe'


def main():
    client = Client()
    assert not RECEIPT.exists(), 'Transfer receipt exists; never recreate dataset'
    assert not client.private_files(REF)['exists'], 'Dataset already exists; never recreate'
    config = json.loads((ROOT/'e1-config.json').read_text())
    specs = [s for s in config['models'] if s['name'] != 'pilot-b']
    entries = []
    # No existing local payload is removed; only this context's copies are temporary.
    with tempfile.TemporaryDirectory(prefix='e2-transfer-') as temp:
        temp = Path(temp); stage = temp/'stage'; download = temp/'download'
        stage.mkdir(); download.mkdir()
        manifest = client.kernel_file(SOURCE, 'b1-output-manifest.json', download)
        assert sha(manifest) == MANIFEST_SHA, 'Pinned B6 v1 manifest mismatch'
        inventory = {r['file']: r for r in json.loads(manifest.read_text())['files']}
        for s in specs:
            gpu = s['relative_path'].split('/')[0]
            point = client.kernel_file(SOURCE, gpu+'/points/b6-final/checkpoint.json', download)
            expected = inventory[gpu+'/points/b6-final/checkpoint.json']
            assert sha(point) == expected['sha256'] and point.stat().st_size == expected['bytes']
            checkpoint = json.loads(point.read_text())
            assert checkpoint['sha256'] == s['sha256'] and checkpoint['file'] == s['relative_path'].split('/', 1)[1]
            assert inventory[s['relative_path']]['sha256'] == s['sha256']
            source = WORK/'results/kaggle-runs/nlp-hw02-b6/v10-pinned-1'/s['relative_path']
            assert sha(source) == s['sha256'], 'Pinned local B6 SHA mismatch'
            name = s['relative_path'].replace('/', '__')
            shutil.copyfile(source, stage/name)
            assert sha(stage/name) == s['sha256']
            entries.append(dict(file=name, bytes=source.stat().st_size, sha256=s['sha256'],
                                source=SOURCE, original=s['relative_path'],
                                model=s['name'], point=gpu+'/points/b6-final'))
        write(stage/'b6-fp16-manifest.json', dict(format='nlp-hw02-b6-fp16-v1', files=entries,
              source=SOURCE, source_manifest_sha256=MANIFEST_SHA,
              purpose='Frozen B6 final model-only FP16 weights for E2 and final inference',
              full_states=False, destination=REF))
        (stage/'README.md').write_text('# NLP HW02 B6 final FP16 weights\n\n'
            'Private frozen model-only payloads from friend1/nlp-hw02-b6 version 1.\n'
            'GPU0 = W B6; GPU1 = C-long2x B6. Both are points/b6-final.\n'
            'Filenames flatten the original path using double underscores.\n'
            'No optimizer state, training data, credentials, or P1 weights.\n'
            'b6-fp16-manifest.json records source paths, sizes, and SHA256.\n'
            'Reusable by E2 experiments and the final inference package.\n')
        for name in ('b6-fp16-manifest.json', 'README.md'):
            p = stage/name; entries.append(dict(file=name, bytes=p.stat().st_size, sha256=sha(p)))
        write(stage/'dataset-metadata.json', dict(title='NLP HW02 B6 final FP16 weights', id=REF,
              subtitle='Frozen W B6 and C-long2x B6 model-only weights from B6 v1',
              description=(stage/'README.md').read_text(), isPrivate=True, licenses=[dict(name='CC0-1.0')]))
        ledger = dict(at_utc=datetime.now(timezone.utc).isoformat(), dataset=REF,
                      create_calls=1, files=entries, roundtrip_sha256_passed=False)
        write(RECEIPT, ledger)
        client.create(stage, 'thisarivson')
        shutil.rmtree(stage)
        remote = None
        for _ in range(90):
            r = client.private_files(REF)
            if r.get('exists'):
                assert r['private'], 'Dataset must be private'
                if r['status'] == 'ready' and all(r['files'].get(f['file']) == f['bytes'] for f in entries):
                    remote = r; break
            time.sleep(2)
        assert remote is not None, 'Dataset not ready; do not recreate'
        for entry in entries:
            p = client.dataset_file(REF+'/'+str(remote['version']), entry['file'], download)
            assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'Roundtrip SHA mismatch'
            p.unlink()
            print('E2 roundtrip SHA passed', entry['file'], flush=True)
        ledger.update(roundtrip_sha256_passed=True, dataset_version=remote['version'],
                      server_private=True, full_states=False)
    ledger.update(temporary_transfer_files_deleted=not temp.exists(),
                  verified_at_utc=datetime.now(timezone.utc).isoformat())
    write(RECEIPT, ledger)
    print(json.dumps(ledger))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__, reason=str(error)
              if isinstance(error, AssertionError) else 'API/CLI failed; raw response suppressed')))
        raise SystemExit(1)
