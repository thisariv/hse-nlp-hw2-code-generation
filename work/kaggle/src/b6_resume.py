"""Pinned B5v6 kernel output mounts; never copy full states or rebuild weighted orders."""
import json, shutil
from pathlib import Path
from kaggle_io import sha, safe_name
from b5_resume import verify_order


def mount(source, working, cache, pins):
    source=Path(source);working=Path(working)
    assert sha(source/'b1-output-manifest.json')==pins['source_manifest_sha256']
    inventory={r['file']:r for r in json.loads((source/'b1-output-manifest.json').read_text())['files']}
    def checked(name):
        p=source/safe_name(name);r=inventory[name]
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],name
        return p
    roots={}
    for g,pin in pins['anchors'].items():
        prefix=f'model-GPU{g}/'
        latest=checked(prefix+'latest.json');assert json.loads(latest.read_text())==pin
        blob=checked(prefix+pin['file']);assert sha(blob)==pin['sha256'] and blob.stat().st_size==pin['bytes']
        final=checked(prefix+'points/'+pins['finals'][g]['tag']+'/complete.json')
        assert sha(final)==pins['finals'][g]['complete_sha256']
        assert json.loads(final.read_text())['training']['anchor']==pins['finals'][g]['anchor']
        dest=working/f'weighted-order-GPU{g}';dest.mkdir(parents=True,exist_ok=True)
        for name in ('order.npy','order.json'):shutil.copyfile(checked(f'weighted-order-GPU{g}/'+name),dest/name)
        verify_order(dest,cache,pin)
        roots[g]=str(latest.parent)
    return roots
