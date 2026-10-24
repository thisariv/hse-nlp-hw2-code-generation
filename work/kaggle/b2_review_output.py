"""Independent smoke gate: producer SHA, both LR workers, stable and mini-eval receipts."""
import argparse
import json
import math
from pathlib import Path
from b1_review_output import sha

def audit(root,metadata):
    root=Path(root);server=json.loads(Path(metadata).read_text())
    manifest=json.loads((root/'b1-output-manifest.json').read_text())['files']
    entries={r['file']:r for r in manifest};assert len(entries)==len(manifest)
    checked=[]
    for name,r in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=root/name
        if p.is_file():
            assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],name
            checked.append(name)
    session=json.loads((root/'session-summary.json').read_text())
    assert session['status']=='complete' and session['exit_codes']==[0,0]
    assert session['wall_seconds']<840 and session['server_ceiling_seconds']==900
    pre=json.loads((root/'input-preflight.json').read_text())
    assert pre['passed'] and pre['all_cache_SHA_verified'] and pre['cache_rows']==9725737
    workers=[]
    for gpu,name,peak in [(0,'W-lr1000',.001),(1,'W-lr600',.0006)]:
        folder=root/f'model-GPU{gpu}';r=json.loads((folder/'gpu-smoke.json').read_text())
        assert r['passed'] and r['model']==name and r['parameter_count']==113426435
        assert r['main_attempts']==256 and r['peak_lr_attempts']==16
        assert r['AMP_skips']==r['nonfinite_skips']==r['NaN_escalations']==r['peak_lr_AMP_skips']==0
        assert r['AMP_overflow_skip'] and r['CUDA_next_update_bit_exact'] and r['evaluation_restore_12_updates_bit_exact']
        assert math.isfinite(r['sampled_attention_abs_logit_max']) and len(r['attention_logit_samples'])==3
        assert all(math.isfinite(v) and v>0 for v in r['z_loss_range'])
        assert r['pairs_per_second']>80 and len(r['step_seconds'])==32
        probe=r['memory_probe'];assert probe['micro_batch']<=32 and probe['attempts'][-1]['Adam_step_verified']
        latest=json.loads((folder/'latest.json').read_text());blob=f'model-GPU{gpu}/'+latest['file']
        assert blob in entries and all(latest[k]==entries[blob][k] for k in ('sha256','bytes'))
        assert 11.9*r['parameter_count']<latest['bytes']<12.1*r['parameter_count']
        assert latest['updates']==256 and latest['cursor']==256*128
        recipe=latest['identity']['recipe']
        assert recipe['warmup_updates']==2000 and recipe['peak_lr']==peak and recipe['z_loss']==1e-4
        assert recipe['dropout']==recipe['main_truncation_q']==0.
        for n,h in recipe['implementation_sha256'].items():assert h==server['modules'][n]
        points=list((folder/'points').glob('*/complete.json'));assert len(points)==1
        point=json.loads(points[0].read_text());assert point['training']['q']==0 and point['training']['cooldown_seconds']>=3.
        for split,n in [('heldout4k',64),('old991',16)]:
            row=point['metrics']['splits'][split]
            assert row['exec']['denominator']==n and math.isfinite(row['CE']['CE'])
        assert r['v7_QK_generation']['rows']==32 and r['v7_QK_generation']['model_spec_qk_norm']=='layernorm'
        workers.append(dict(gpu=gpu,model=name,peak_lr=peak,pairs_per_second=r['pairs_per_second'],
            micro_batch=probe['micro_batch'],memory_probe=probe,main_attempts=256,peak_lr_attempts=16,
            AMP_skips=0,nonfinite_skips=0,NaN_escalations=0,
            peak_allocated_bytes=r['training_peak_allocated_bytes'],peak_reserved_bytes=r['training_peak_reserved_bytes'],
            sampled_attention_abs_logit_max=r['sampled_attention_abs_logit_max'],z_loss_range=r['z_loss_range'],
            stable_bytes=latest['bytes'],stable_sha256=latest['sha256'],CUDA_resume_bit_exact=True,
            eval_restore_continuation_bit_exact=True,v7_QK_generation=r['v7_QK_generation']))
    return dict(passed=True,session_wall_seconds=session['wall_seconds'],small_files_SHA_verified=len(checked),
        output_bytes=sum(r['bytes'] for r in manifest),large_PT_downloaded=False,quality_gate=False,
        input_preflight=pre,workers=workers,modules=server['modules'],source_port_sha256=server['source_port_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();result=audit(a.root,a.metadata)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('modules','input_preflight')},indent=2))
