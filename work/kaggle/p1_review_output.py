#!/usr/bin/env python3
"""Independent small-output audit. No checkpoint/cache download is required."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))


def audit(root):
    def read(path): return json.loads((root/path).read_text())
    smoke=read('p1-smoke-summary.json'); session=read('session-summary.json')
    assert smoke['passed'] and session['exit_codes']==[1,0] and session['failed_workers']==[0]
    assert not session['unreaped_worker_pids'] and not (root/'stop-request.json').exists()
    pre=read('input-preflight.json'); anchors=read('anchor-preflight.json')
    assert pre['passed'] and pre['account']=='friend1' and pre['all_shard_SHA_verified']
    assert pre['cache_rows']==9725737 and pre['cache_shards']==60
    assert anchors['passed'] and pre['setup_wall_seconds']<480
    pin=json.loads(Path(__file__).with_name('p1-anchors.json').read_text())['anchors']['C-long2x']
    assert anchors['anchors']['C-long2x']['sha256']==pin['sha256']
    assert pre['weighted_order_sha256']==pin['identity']['order_sha256']
    if (root/'fallback-preflight.json').exists():
        fallback=read('fallback-preflight.json')
        assert fallback['passed'] and fallback['all_shard_SHA_verified'] and fallback['tokenizer_SHA_verified']
        assert fallback['manifest_sha256']==pre['cache_manifest_sha256'] and fallback['answer_mismatches']==0
    faults=read('pilot-smoke.json')
    assert faults['passed'] and faults['finite_update_bit_exact'] and faults['serialized_resume_bit_exact']
    assert set(faults['truncation_faults'])=={'graph','two-pass','partial_backward'}
    branches={}; streams=[]; starts=[]; journals=[]
    from l2_sampler import fixed_split
    expected_ids=[i for i in fixed_split()['tune2048'] if i<64]
    for label,tag in [('a','long'),('b','long-truncation')]:
        point=Path(f'model-pilot-{label}/points/{tag}')
        training=read(point/'training.json'); complete=read(point/'complete.json'); restore=read(point/'restore.json')
        assert training['cooldown_updates']==training['planned_cooldown_updates']==64
        assert training['anchor_sha256']==pin['sha256'] and training['anchor_cursor']==pin['cursor']
        assert training['anchor_updates']==pin['updates'] and training['final_lr']==0.
        assert training['cooldown_AMP_skips']==training['cooldown_nonfinite_skips']==0
        assert restore['passed'] and restore['fingerprints']==training['start_fingerprints']
        assert set(restore['fingerprints'])=={'model','optimizer','scaler','rng','cursor_history'}
        assert restore['cursor']==pin['cursor'] and restore['updates']==pin['updates']
        starts.append(training['start_fingerprints']);streams.append(training['stream_sha256'])
        ev=point/'evaluation'; metrics=read(ev/'metrics.json')
        assert metrics==complete['metrics'] and metrics['test']['rows']==256
        assert len(read(ev/'heldout4k-per-example.json'))==64 and len(read(ev/'old991-per-example.json'))==16
        for split in metrics['splits'].values(): assert math.isfinite(split['CE']['CE'])
        vote=read(ev/'k8-tune2048/summary.json')
        assert vote['complete'] and vote['rows']==vote['target_rows']==len(expected_ids)==36
        assert vote['K']==8 and vote['T']==.7 and vote['seed']==270042 and not vote['OOM_attempts']
        rows=[row for f in sorted((root/ev/'k8-tune2048').glob('batch-*.json')) for row in json.loads(f.read_text())]
        assert [r['id'] for r in rows]==expected_ids
        assert all([c['sample'] for c in r['samples']['0.7']]==list(range(8)) for r in rows)
        for row in rows:
            for c in row['samples']['0.7']:
                assert c['sampling_status']=='complete' and math.isfinite(c['mean_logprob'])
                assert isinstance(c['represented_code'],str) and 'execution' in c
        journals.append([(r['id'],r['question_sha256'],r['reference']) for r in rows])
        if label=='b':
            log=[json.loads(l) for l in (root/point/'truncation.jsonl').read_text().splitlines()]
            assert len(log)==64 and [r['step'] for r in log]==list(range(1,65))
            for r in log:
                assert r['presented']==128 and r['retained']==115 and sum(r['excluded'])==13
                assert r['retained_tokens']<r['presented_tokens'] and len(r['excluded_positions'])==13
                assert all(n==0 or drop<n for n,drop in zip(r['counts'],r['excluded']))
        branches[label]=dict(updates=64,K8_rows=36,K8_samples=288,K8_seconds=vote['seconds'],
                             peak_allocated_bytes=training['peak_allocated_bytes'],
                             peak_reserved_bytes=training['peak_reserved_bytes'],
                             updates_per_second=training['throughput_updates_per_second'],
                             greedy256_seconds=metrics['test']['full_pipeline_seconds'],
                             weights_sha256=complete['weights_sha256'])
    assert streams[0]==streams[1] and starts[0]==starts[1] and journals[0]==journals[1]
    bench=read('pilot-throughput.json')
    assert len(bench['a']['seconds'])==len(bench['b']['seconds'])==32
    # 15% is the explicit reviewer tolerance for material overhead; keep raw timings.
    assert bench['B_over_A_time']<=1.15, 'Truncation materially slows the paired protocol'
    assert bench['projected_seconds']['a']<13800 and bench['projected_seconds']['b']<17400
    reviewer=read('reviewer-smoke.json'); report=read('model-reviewer/summary.json')
    assert reviewer['passed'] and reviewer['updates']>=300 and reviewer['stable_reload_verified']
    assert reviewer['scorer_artifact_contract'] and report['status']=='complete'
    assert report['state']['cursor']==report['state']['updates']*128
    assert reviewer['stable_updates']<reviewer['updates'] and reviewer['reloaded_final_CE']['complete']
    final=read('model-reviewer/points/final/complete.json')
    assert final['weights_sha256']==reviewer['final_weights_sha256']
    assert final['metrics']['heldout4k']['rows']==64 and final['metrics']['old991']['rows']==16
    order=read('reviewer-order.json');assert order['rows']==9725736 and order['excluded_indices']==[8545870]
    stable=read('model-reviewer/latest.json')
    assert stable['cursor']==128*stable['updates']==128*reviewer['stable_updates']
    count=0; omitted=[]
    for rec in read('p1-output-manifest.json')['files']:
        path=root/rec['file']
        if path.is_file():
            assert path.stat().st_size==rec['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
            count+=1
        else:
            assert path.suffix in ('.pt','.npy','.py','.ipynb'), 'Missing small output '+str(path)
            omitted.append(rec['file'])
    return dict(passed=True,account='friend1',wall_seconds=session['wall_seconds'],branches=branches,
                throughput=bench,reviewer=reviewer,starts_bit_exact=True,data_stream_bit_exact=True,
                K8_seed_temperature_order_equal_L3=True,faults=faults['truncation_faults'],
                output_manifest_SHA_verified=count,large_or_source_files_not_downloaded=omitted,
                input_preflight=pre,anchor_preflight=anchors,
                limitation='32 sampled batch positions do not guarantee full-run throughput; live Secrets path awaits UI run')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit(a.output);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','wall_seconds','branches','reviewer')},indent=2))
