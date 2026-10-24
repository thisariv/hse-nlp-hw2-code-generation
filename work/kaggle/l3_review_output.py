#!/usr/bin/env python3
"""Independently audit downloaded L3 smoke JSON/log output, without retaining large weights."""
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-fallback',action='store_true');a=p.parse_args()
    root=a.output
    def read(name):return json.loads((root/name).read_text())
    pins=json.loads(Path(__file__).with_name('l3-anchors.json').read_text())
    smoke=read('cuda-smoke-summary.json');session=read('session-summary.json')
    assert smoke['passed'] and smoke['packaging_passed']
    assert session['exit_codes']==[1,0] and session['failed_workers']==[0] and not session['unreaped_worker_pids']
    assert not (root/'stop-request.json').exists()
    inp=read('input-preflight.json');assert inp['passed'] and inp['account']=='friend1'
    assert inp['cache_manifest_sha256']==pins['anchors']['C-main']['identity']['cache_manifest_sha256']
    assert inp['cache_rows']==9725737 and inp['cache_shards']==60 and inp['all_shard_SHA_verified']
    assert inp['main_order_sha256']==pins['anchors']['C-main']['identity']['order_sha256']
    assert inp['weighted_order_sha256']==pins['anchors']['C-long2x']['identity']['order_sha256']
    pre=read('anchor-preflight.json');assert pre['passed'] and pre['preflight_wall_seconds']<480
    fallback=read('fallback-preflight.json') if (root/'fallback-preflight.json').exists() else None
    if a.require_fallback:assert fallback is not None, 'Missing full fallback verification'
    if fallback:
        assert fallback['passed'] and fallback['all_shard_SHA_verified']
        assert fallback['manifest_sha256']==inp['cache_manifest_sha256'] and fallback['rows']==inp['cache_rows']
        assert fallback['shards']==60 and fallback['tokenizer_SHA_verified'] and fallback['answer_mismatches']==0
    summaries={}
    for gpu,name in enumerate(('C-main','C-long2x')):
        model=Path('model-'+name);r=read(model/'gpu-smoke.json');st=read(model/'summary.json')['state'];pin=pins['anchors'][name]
        assert r['passed'] and r['main_added_updates']>=300
        for label in ('bit_exact_L1_CUDA_update','GradScaler_overflow_skip','partial_backward_nonfinite_loss_skip',
                      'serialized_resume_bit_exact_CUDA','K8_executed_samples_and_greedy256'):
            assert label in r['checks']
        assert ('nonfinite_escalation_full_rollback_LR' if gpu==0 else 'constant_LR_weighted_fork') in r['checks']
        if gpu==1:assert r['checks']['healthy_worker_resumed_after_sibling_crash']['continued_updates']==64
        assert st['cursor']==pin['cursor']+128*(st['updates']-pin['updates'])
        assert st['last_lr']==.0015*(.5 if gpu==0 else 1.)
        assert [x['tag'] for x in st['l2_points'][:3]]==['l2-10800','l2-21600','l2-28800']
        assert len(st['l2_points'])==5 and all(x['complete'] for x in st['l2_points'])
        current=st['l2_points'][3:];samples=0
        for point in current:
            ev=model/'points'/point['tag']/'evaluation'
            assert len(read(ev/'heldout4k-per-example.json'))==64
            assert len(read(ev/'old991-per-example.json'))==16
            assert len(read(ev/'test-execution.json')['rows'])==256
            assert point['training']['cooldown_seconds']>=20 and point['metrics']['test']['rows']==256
        ev=model/'points'/current[-1]['tag']/'evaluation/k8-tune2048'
        vote=read(ev/'summary.json');assert vote['complete'] and vote['rows']==vote['target_rows']>0
        rows=[r for f in sorted((root/ev).glob('batch-*.json')) for r in json.loads(f.read_text())]
        assert len(rows)==vote['rows'] and len({r['id'] for r in rows})==len(rows)
        for row in rows:
            assert len(row['samples']['0.7'])==8
            for c in row['samples']['0.7']:
                assert c['sampling_status']=='complete' and 'execution' in c and c['mean_logprob'] is not None
                samples+=1
        probe_peak=max(x.get('peak_reserved_bytes',0) for x in r['probe']['attempts'])
        summaries[name]={'CUDA_checks':list(r['checks']),'main_added_updates':r['main_added_updates'],
                         'tail_updates':64 if gpu else 0,'K8_rows':vote['rows'],'K8_samples':samples,'K8_seconds':vote['seconds'],
                         'greedy256_seconds':[p['metrics']['test']['full_pipeline_seconds'] for p in current],
                         'training_eval_peak_allocated_bytes':r['peak_allocated_bytes'],
                         'training_eval_peak_reserved_bytes':r['peak_reserved_bytes'],
                         'longest_row_probe_peak_reserved_bytes':probe_peak,
                         'observed_max_reserved_bytes':max(probe_peak,r['peak_reserved_bytes']),
                         'stable_LR':st['last_lr'],'stable_updates':st['updates'],'stable_cursor':st['cursor']}
    verified=0;omitted=[]
    for rec in read('l2-output-manifest.json')['files']:
        path=root/rec['file']
        if path.is_file():
            assert path.stat().st_size==rec['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
            verified+=1
        else:
            assert path.suffix in ('.pt','.npy','.py','.ipynb'), 'Missing review output'
            omitted.append(rec['file'])
    receipt={'passed':True,'account':'friend1','wall_seconds':smoke['wall_seconds'],'input_preflight':inp,
             'anchor_preflight':pre,'models':summaries,'fault_exit_codes':session['exit_codes'],
             'fallback_preflight':fallback,
             'output_manifest_downloaded_files_SHA_verified':verified,'large_or_source_files_not_downloaded':omitted,
             'limitations':'Real CUDA loader verified full states on Kaggle; no redundant local PT download. Full 11h/live Secrets path requires UI run.'}
    a.out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'wall_seconds':smoke['wall_seconds'],'models':summaries},indent=2))

if __name__=='__main__':main()
