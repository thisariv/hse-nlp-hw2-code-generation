"""Manifest-checked flat private transfer mounts; strict state read remains in runtime."""
import json
import shutil
from pathlib import Path
from kaggle_io import sha, safe_name


def materialize(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/'b1-transfer.json').read_text())
    assert manifest['format'] == 'nlp-hw02-b1-transfer-v1'
    expected = {'weighted-order/order.npy', 'weighted-order/order.json'}
    for gpu in (0, 1): expected.add(f'model-GPU{gpu}/latest.json')
    originals = {r['original'] for r in manifest['files']}
    assert expected.issubset(originals) and len(originals) == len(manifest['files'])
    for rec in manifest['files']:
        src = source/safe_name(rec['file']); dst = destination/safe_name(rec['original'])
        assert not src.is_symlink() and src.stat().st_size == rec['bytes'] and sha(src) == rec['sha256']
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Read-only mount is retained for the session; symlinks avoid another ~2.5GB copy.
        dst.symlink_to(src.resolve())
    for gpu in (0, 1):
        root = destination/f'model-GPU{gpu}'; rec = json.loads((root/'latest.json').read_text())
        path = root/safe_name(rec['file'])
        assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256']
    return destination
