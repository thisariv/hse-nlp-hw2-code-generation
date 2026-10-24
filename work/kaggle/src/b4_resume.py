"""Read B3 kernel output in place; no transferred Dataset or full-PT copy."""
import json
from pathlib import Path
from kaggle_io import sha, safe_name


def verify_mount(source, pin, manifest_sha):
    source=Path(source)
    assert sha(source/'b1-output-manifest.json')==manifest_sha,'B3 output manifest changed'
    manifest=json.loads((source/'b1-output-manifest.json').read_text())
    inventory={r['file']:r for r in manifest['files']}
    root=source/'model-GPU1';assert json.loads((root/'latest.json').read_text())==pin
    name='model-GPU1/'+str(safe_name(pin['file']));path=source/name
    assert inventory[name]['sha256']==pin['sha256'] and inventory[name]['bytes']==pin['bytes']
    assert path.is_file() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256']
    for n in ('weighted-order/order.json','weighted-order/order.npy'):
        r=inventory[n];p=source/n
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    assert sha(source/'weighted-order/order.npy')==pin['identity']['order_sha256']
    return source
