"""Independent small-output audit. Large PTs stay on Kaggle; workers verify their SHA on reload."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def audit(root):
    root=Path(root)
    manifest=json.loads((root/'b1-output-manifest.json').read_text())['files']
    entries={v['file']:v for v in manifest};assert len(entries)==len(manifest)
    checked=[]
    for name,rec in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=root/name
        if p.is_file():
            assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],name
            checked.append(name)
    session=json.loads((root/'session-summary.json').read_text())
    assert session['status']=='complete' and session['exit_codes']==[0,0]
    assert session['wall_seconds']<840 and session['server_ceiling_seconds']==900
    inputs=json.loads((root/'input-preflight.json').read_text())
    assert inputs['passed'] and inputs['all_cache_SHA_verified'] and inputs['cache_rows']==9725737
    assert '/notebooks/thisarivson/nlp-hw02-compact-cache-verified' in inputs['cache_mount']
    workers=[]
    for gpu,name,enc,dec,params,micro in ((0,'E',8,4,108694019,64),(1,'W',6,6,113421827,32)):
        folder=root/f'model-GPU{gpu}';r=json.loads((folder/'gpu-smoke.json').read_text())
        assert r['passed'] and r['status']=='complete' and r['model']==name
        spec=r['spec'];assert spec==dict(d_model=768,heads=12,encoder_layers=enc,decoder_layers=dec,swiglu_dim=2048,vocab=8195)
        assert r['parameter_count']==params and r['main_attempts']==256 and r['nonfinite_skips']==0
        assert r['peak_lr_attempts']==16
        assert r['CUDA_next_update_bit_exact'] and r['eval_restore_next_12_updates_bit_exact'] and r['AMP_overflow_skip']
        probe=r['memory_probe'];assert probe['micro_batch']==micro
        assert probe['attempts'][-1]['Adam_step_verified']
        assert r['training_pairs_per_second']>140 and len(r['step_seconds'])==32
        assert len(r['shape_bench_replay']['batches'])==8
        # A wide guard catches an invalid timer; small differences remain visible in the report.
        assert .65<r['shape_bench_ratio']<1.35
        latest=json.loads((folder/'latest.json').read_text())
        blob=f'model-GPU{gpu}/'+latest['file'];assert blob in entries
        assert all(latest[k]==entries[blob][k] for k in ('sha256','bytes'))
        assert 11.9*params<latest['bytes']<12.1*params
        state=r['result']['state'];assert state['updates']==269 and state['cursor']==269*128
        assert r['result']['end_main_complete'] and r['result']['status']=='complete'
        assert latest['updates']==269 and latest['cursor']==269*128
        assert len(state['points'])==2 and all(p['complete'] for p in state['points'])
        for point in state['points']:
            train=point['training'];assert train['q']==0. and train['cooldown_seconds']>=3.
            for split,n in (('heldout4k',64),('old991',16)):
                metrics=point['metrics']['splits'][split]
                assert metrics['exec']['denominator']==n and math.isfinite(metrics['CE']['CE'])
        recipe=latest['identity']['recipe']
        assert recipe['dropout']==0. and recipe['peak_lr']==.001 and recipe['warmup_updates']==1000
        assert recipe['main_truncation_q']==0. and recipe['effective_batch']==128
        workers.append(dict(gpu=gpu,model=name,spec=spec,parameters=params,micro=micro,
            probe_allocated_GiB=probe['attempts'][-1]['peak_allocated_bytes']/1024**3,
            probe_reserved_GiB=probe['attempts'][-1]['peak_reserved_bytes']/1024**3,
            train_allocated_GiB=r['peak_allocated_bytes']/1024**3,train_reserved_GiB=r['peak_reserved_bytes']/1024**3,
            weighted_pairs_per_second=r['training_pairs_per_second'],
            shape_bench_pairs_per_second=r['shape_bench_replay']['pairs_per_s'],shape_bench_ratio=r['shape_bench_ratio'],
            AMP_skips=r['AMP_skips'],peak_lr_AMP_skips=r['peak_lr_AMP_skips'],stable_bytes=latest['bytes'],
            stable_sha256=latest['sha256'],CUDA_resume_bit_exact=True,eval_restore_continuation_bit_exact=True))
    return dict(passed=True,session_wall_seconds=session['wall_seconds'],small_files_SHA_verified=len(checked),
                manifest_files=len(entries),output_bytes=sum(r['bytes'] for r in manifest),workers=workers,
                large_PT_downloaded=False,quality_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();result=audit(args.root)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
