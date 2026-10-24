"""Reuse every completed compact shard from v1; redo only independent executor audit."""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    from kaggle_io import sha,write,resolve_dataset,materialize_dataset
    from l1_cache import CompactCache,ROWS,FILES
    source=materialize_dataset(resolve_dataset(cfg['input_root'],cfg['recovery_dataset']),Path(cfg['temp_root'])/'v1','compact-manifest.json')
    root=Path(cfg['working_root'])/'compact-cache';started=time.monotonic();shutil.copytree(source,root)
    # The recovery upload contains only new target arrays/order/samples. Exact
    # unchanged source/provenance bytes come from the already verified original mounts.
    base=materialize_dataset(resolve_dataset(cfg['input_root'],'nlp-hw02-26-token-cache'),Path(cfg['temp_root'])/'base','cache-manifest.json')
    ext=materialize_dataset(resolve_dataset(cfg['input_root'],'nlp-hw02-26c-cpu-cache'),Path(cfg['temp_root'])/'ext','cache-manifest.json')
    manifest=json.loads((root/'compact-manifest.json').read_text())
    assert [sha(base/'cache-manifest.json'),sha(ext/'cache-manifest.json')]==manifest['original_manifest_sha256']
    for rec in manifest['shards']:
        original=(base if rec['shard']<42 else ext)/rec['folder']
        for name in FILES:
            if name not in ('target.npy','target-offsets.npy'):shutil.copyfile(original/name,root/rec['folder']/name)
    cache=CompactCache(root,verify=True);assert len(cache)==ROWS
    assert [r['shard'] for r in cache.manifest['shards']]==list(range(60))
    del cache
    # The executable main module is intentionally dependency-free, unlike l1_cache.py.
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('l1_roundtrip.py')),'--root',str(root)],
                          check=False,timeout=1200)
    assert result.returncode==0,'Round-trip failed; see saved diagnostics'
    report=json.loads((root/'roundtrip-report.json').read_text());assert report['rows']>=50000
    manifest=json.loads((root/'compact-manifest.json').read_text());order=json.loads((root/'order.json').read_text())
    summary={'status':'complete','rows':ROWS,'shards':list(range(60)),'manifest_sha256':sha(root/'compact-manifest.json'),
             'order':order,'roundtrip_rows':report['rows'],'answer_mismatches':report['answer_mismatches'],
             'roundtrip':report['score'],'seconds':time.monotonic()-started,'GPU':False,
             'representation_code_sha256':manifest['representation_code_sha256'],'tokenizer_sha256':manifest['tokenizer']['sha256'],
             'reused_source':'thisarivson/nlp-hw02-compact-cpu-cache/1','all_converted_arrays_reused':True,
             'first_run_wall_seconds':4579.6,'original_roundtrip_failure':'Original v1 audit failed; original/restored result journals not retained',
             'roundtrip_executor':report['executor_entrypoint']}
    write(root/'cache-summary.json',summary)
    for p in (root/'shards').glob('*/roundtrip-sample.json'):p.unlink()
    print('VERIFIED CACHE',json.dumps({k:summary[k] for k in ('rows','manifest_sha256','roundtrip_rows','answer_mismatches','seconds')}),flush=True)

if __name__=='__main__':main()
