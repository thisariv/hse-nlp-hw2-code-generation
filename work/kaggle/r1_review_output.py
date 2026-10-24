"""Audit all small R1 smoke artifacts against the producer manifest."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())

def audit(root):
    manifest=read(root/'r1-output-manifest.json');checked=[]
    for item in manifest['files']:
        if Path(item['file']).suffix not in ('.json','.jsonl','.log'):continue
        path=root/item['file'];assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['file']
        checked.append(item['file'])
    session=read(root/'session-summary.json')
    assert session['status']=='complete' and session['exit_codes']==[0,0] and not session['unreaped_worker_pids']
    assert session['wall_seconds']<900 and session['hard_shutdown_seconds']<=840
    assert read(root/'startup-gate.json')['live_enabled'] is False
    anchor=read(root/'anchor-preflight.json');inputs=read(root/'input-preflight.json')
    assert anchor['passed'] and inputs['passed'] and inputs['all_shard_SHA_verified']
    cfg=read(root/'session-config.json');assert cfg['owner']=='thisarivson' and not cfg['require_secrets']
    assert cfg['branches'][0]['anchor']==cfg['branches'][1]['anchor']=='C-long2x'
    assert all(b['data']=='ordinary' and b['q']==.1 for b in cfg['branches'])
    branches=[]
    for gpu in (0,1):
        p=root/f'model-GPU{gpu}';probe=read(root/f'smoke-probe-gpu{gpu}/probe.json')
        summary=read(p/'summary.json');latest=read(p/'latest.json');v7=read(p/'v7-load.json')
        assert probe['passed'] and summary['status']=='complete' and summary['stable_preserved'] and v7['passed']
        checks=probe['checks'];assert checks['GradScaler_overflow']['passed'] and checks['serialized_next_update_bit_exact']['passed']
        if gpu==0:
            for mode in ('AMP','FP16'):assert checks['growth_identity_'+mode]['max_abs']==0
        train=summary['training'];assert train['main_updates']==256 and train['cooldown_updates']>0
        assert latest['updates']==anchor['anchors']['C-long2x']['updates']+256
        assert latest['cursor']==anchor['anchors']['C-long2x']['cursor']+256*128
        assert latest['model_spec']['encoder_layers']==(12 if gpu==0 else 6)==v7['encoder_layers']
        vote=summary['metrics']['K8_tune2048'];assert vote['complete'] and vote['rows']==vote['target_rows']==36
        journals=[r for f in sorted((p/'points/final/evaluation/k8-tune2048').glob('batch-*.json')) for r in read(f)]
        assert len(journals)==36
        assert all(len(r['samples']['0.7'])==8 and all('execution' in c and c['mean_logprob'] is not None
                   and c['sampling_status']=='complete' for c in r['samples']['0.7']) for r in journals)
        progress=[json.loads(line) for line in (p/'truncation.jsonl').read_text().splitlines()]
        assert all(r['requested_q']==.1 and sum(r['excluded'])==13 for r in progress)
        timing=checks['representative_timing'];assert len(timing['samples'])==32
        branches.append(dict(gpu=gpu,encoder_layers=v7['encoder_layers'],seconds_per_step=timing['seconds_per_step'],
                             graph_OOM_fallbacks=timing['graph_OOM_fallbacks'],AMP_skips=timing['AMP_skips'],
                             benchmark_allocated_GiB=probe['peak_allocated_bytes']/1024**3,
                             benchmark_reserved_GiB=probe['peak_reserved_bytes']/1024**3,
                             session_reserved_GiB=summary['peak_reserved_bytes']/1024**3,
                             main_updates=256,cooldown_updates=train['cooldown_updates'],
                             stable_bytes=latest['bytes'],stable_sha256=latest['sha256'],
                             stable_preserved=True,v7=v7,K8_rows=36,K8_samples=288))
    ratio=branches[0]['seconds_per_step']/branches[1]['seconds_per_step']
    return dict(passed=True,checked_small_files=len(checked),wall_seconds=session['wall_seconds'],
                anchor_preflight_seconds=anchor['preflight_wall_seconds'],input_preflight_seconds=inputs['setup_wall_seconds'],
                branches=branches,deep_base_time_ratio=ratio,chosen_encoder_layers=9 if ratio>1.7 else 12,
                selection_rule='6->9 after old1/3/5 only if T4 deep/base >1.7',
                large_PT_downloaded=False,require_secrets=False,live_sync=False,
                quality_not_measured='smoke greedy/K8 cap8; full production uses cap512')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.root);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
