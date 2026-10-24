"""SHA-pinned private W transfer + existing C-long2x live v12, no full-state copies."""
import json
from pathlib import Path
from kaggle_io import sha,safe_name,write
from l3_resume import mounted_expected,restore_order
from l2_sampler import load_order
import numpy as np


def transfer_mount(source,destination,expected_sha,pin):
    source,destination=Path(source),Path(destination)
    assert sha(source/'b5-transfer.json')==expected_sha
    manifest=json.loads((source/'b5-transfer.json').read_text())
    assert manifest['format']=='nlp-hw02-b5-transfer-v1' and manifest['destination']=='friend1/nlp-hw02-b5-inputs'
    required={'model-GPU0/latest.json','model-GPU0/'+pin['file'],'weighted-order/order.npy','weighted-order/order.json'}
    assert {r['original'] for r in manifest['files']}==required
    destination.mkdir(parents=True,exist_ok=True)
    for r in manifest['files']:
        src=source/safe_name(r['file']);dst=destination/safe_name(r['original'])
        assert src.stat().st_size==r['bytes'] and sha(src)==r['sha256']
        dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(src.resolve())
    assert json.loads((destination/'model-GPU0/latest.json').read_text())==pin
    return destination


def c_mount(source,destination,pin):
    mounted_expected(source,'C-long2x',pin)
    src=Path(source)/safe_name(pin['file'])
    assert src.stat().st_size==pin['bytes'] and sha(src)==pin['sha256']
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    (destination/safe_name(pin['file'])).symlink_to(src.resolve())
    write(destination/'latest.json',pin)
    return destination


def verify_order(root,cache,pin):
    order,meta=load_order(root,cache,pin['identity']['order_sha256'])
    assert meta['cache_manifest_sha256']==cache.manifest_sha
    assert meta['threshold']==79 and meta['batch']==128 and meta['chunk']==4096
    assert meta['multiplicity']==dict(short=1,long=2)
    assert np.array_equal(np.bincount(order,minlength=len(cache)),1+(cache.target_lengths>=79))
    if pin['identity']['model']=='C-long2x':assert meta==pin['identity']['recipe']['sampler']
    else:assert meta['seed']==pin['identity']['order_seed']
    return order,meta
