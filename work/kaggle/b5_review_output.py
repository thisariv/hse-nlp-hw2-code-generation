"""Independent B5 CUDA smoke gate; inspect small outputs, never download full PT."""
import argparse, json, math, sys
from pathlib import Path
from b1_review_output import sha
from build_l3 import ROOT
sys.path.insert(0,str(ROOT/'src'))
from l2_sampler import fixed_split


def audit(root, metadata):
    root=Path(root); server=json.loads(Path(metadata).read_text())
    assert server['private'] and server['config']['gpu_smoke']
    source_identity=json.loads((ROOT/'bundles/b5_friend1/source-identity.json').read_text())
    modules=source_identity['modules']
    assert all(server[k]==v for k,v in source_identity.items()), 'Production differs from the actual server smoke'
    config=json.loads((root/'session-config.json').read_text())
    assert config['port_sha256']==server['source_port_sha256']
    pins=json.loads((ROOT/'b5-anchors.json').read_text())['anchors']
    assert config['anchors']==pins and config['owner']=='friend1'
    assert config['smoke_updates']>=200
    manifest=json.loads((root/'b1-output-manifest.json').read_text())['files']
    inventory={r['file']:r for r in manifest}; assert len(inventory)==len(manifest)
    for name,h in modules.items():
        assert inventory['src-C/'+name]['sha256']==source_identity['C_overrides'].get(name,h),name
    checked=[]
    for name,rec in inventory.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=root/name
        if name.endswith(('.json','.jsonl','.log')):
            assert p.is_file(), name
        if p.is_file():
            assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'], name
            checked.append(name)
    session=json.loads((root/'session-summary.json').read_text())
    assert session['status']=='complete' and session['exit_codes']==[0,0]
    assert session['wall_seconds']<=720 and session['server_ceiling_seconds']==720
    assert not session['unreaped_worker_pids']
    pre=json.loads((root/'input-preflight.json').read_text())
    assert pre['passed'] and pre['all_cache_SHA_verified'] and pre['cache_rows']==9725737
    assert 'compact-cache-verified' in pre['cache_mount'] and 'nlp-hw02-b5-inputs' in pre['W_transfer_mount']
    assert 'nlp-hw02-l3-live' in pre['C_live_mount']
    workers=[];tune_pairs=[]
    for gpu,name,lr in ((0,'W600',.0006),(1,'C-long2x',.0015)):
        folder=root/f'model-GPU{gpu}'; q=0.;pin=pins[str(gpu)]
        process=(root/f'gpu{gpu}-process.log').read_text()
        assert not any(s in process for s in ('Traceback (most recent call last)', 'OOM RETRY', 'OutOfMemoryError', 'attn_bias is not correctly aligned'))
        report=json.loads((folder/'gpu-smoke.json').read_text())
        assert report['passed'] and report['model']==name and report['source_checkpoint_sha256']==pin['sha256']
        assert report['start_fingerprints']==json.loads((folder/'start-fingerprints.json').read_text())['fingerprints']
        assert report['main_updates']>=200 and report['q']==q
        audits=json.loads((folder/'restore-audits.json').read_text())
        assert audits['passed'] and audits['branches']==report['restore_audits'] and len(audits['branches'])==2
        required={'model','optimizer','scaler','scaler_enabled','state','training','grad_enabled','inference_mode',
                  'rng_python','rng_numpy','rng_torch_cpu','rng_torch_cuda'}
        for row in audits['branches']:
            assert row['passed'] and not row['different'] and set(row['before'])==required
            assert row['before']==row['restored']
        assert report['nonfinite_skips']==0
        for flag in ('AMP_overflow_skip','constant_lr','historical_state_preserved','mini_full_evaluation'):
            assert report[flag],flag
        assert report['truncation'] is False
        summary=json.loads((folder/'summary.json').read_text()); state=summary['state']
        assert summary['status']=='complete' and summary['final_complete']
        assert state['updates']==pin['updates']+report['main_updates'] and state['cursor']==pin['cursor']+report['main_updates']*128
        assert state['last_lr']==lr and state['stability']['escalations']==0
        if gpu==0:assert state['lr_factor']==1.
        else:assert state['cursor']==state['cursor_origin']+state['updates']*128
        latest=json.loads((folder/'latest.json').read_text()); blob=f'model-GPU{gpu}/'+latest['file']
        assert latest==summary['B6_uncooled_latest'] and latest['identity']==summary['identity']
        assert all(latest[k]==inventory[blob][k] for k in ('bytes','sha256'))
        assert latest['cursor']==state['cursor'] and latest['updates']==state['updates']
        logs=[json.loads(v) for v in (folder/'updates.jsonl').read_text().splitlines()]
        assert len(logs)==report['main_updates']
        for i,log in enumerate(logs,1):
            assert log['updates']==pin['updates']+i and log['main_pairs']==pin['cursor']+i*128
            assert log['lr']==lr and log['nonfinite_skip']==0
            assert log['AMP_skip'] or all(math.isfinite(log[k]) for k in (('CE','z_loss') if gpu==0 else ('CE',)))
        points=list((folder/'points').glob('*/complete.json')); assert len(points)==2
        assert {p.parent.name for p in points}=={r['tag'] for r in audits['branches']}
        for point_path in points:
            point=json.loads(point_path.read_text()); train=point['training']; metrics=point['metrics']
            assert train['cooldown']=='C timed linear' and train['q']==q and train['start_lr']==lr
            cooldown=[json.loads(v) for v in (point_path.parent/'cooldown-updates.jsonl').read_text().splitlines()]
            assert len(cooldown)==train['completed_updates'] and train['cooldown_seconds']>=train['cooldown_target_seconds']
            elapsed=0.
            for log in cooldown:
                assert math.isclose(log['lr'],lr*(1-min(1.,elapsed/train['cooldown_target_seconds'])),abs_tol=1e-15)
                assert log['nonfinite_skip']==0 and not log['OOM_retries'];elapsed+=log['seconds']
            assert train['cooldown_target_seconds']==(3. if train['final'] else 2.)
            assert train['anchor']['updates']==pin['updates']+(report['main_updates'] if train['final'] else 1)
            for split,n in (('heldout4k',64),('old991',16)):
                row=metrics['splits'][split]
                assert row['exec']['denominator']==n and math.isfinite(row['CE']['CE'])
            if train['final']:
                vote=metrics['K8_tune2048']; assert vote['complete'] and vote['rows']==vote['target_rows']==36
                assert vote['K']==8 and vote['T']==.7 and vote['seed']==270042 and metrics['test']['rows']==64
                assert not vote['OOM_attempts']
                batches=[r for p in (point_path.parent/'evaluation/k8-tune2048').glob('batch-*.json') for r in json.loads(p.read_text())]
                assert len(batches)==36 and len({r['id'] for r in batches})==36
                assert sorted(r['id'] for r in batches)==[i for i in fixed_split()['tune2048'] if i<64]
                assert all(len(r['samples']['0.7'])==8 and all('execution' in v and math.isfinite(v['mean_logprob']) and 'represented_code' in v for v in r['samples']['0.7']) for r in batches)
                tune_pairs.append(sorted((r['id'],r['question_sha256']) for r in batches))
            checkpoint=json.loads((point_path.parent/'checkpoint.json').read_text())
            weight_name=f'model-GPU{gpu}/'+checkpoint['file']
            assert checkpoint['dtype']=='FP16' and checkpoint['optimizer_included'] is False
            assert checkpoint['sha256']==point['weights_sha256']==inventory[weight_name]['sha256']
        bench=report['sampled_throughput']; assert len(bench['samples'])==9 and bench['warmups_excluded']==4
        # These independent architectures have no relative speed acceptance gate.
        assert all(r['micro_batch']>=1 and r['AMP_skip']==0 for r in bench['samples'])
        assert all(r['forward_mode']==('two-pass' if q else 'ordinary') for r in bench['samples'])
        assert all(not r['OOM_retries'] for r in bench['samples']+logs), 'OOM in smoke'
        stable=[r for n,r in inventory.items() if n.startswith(f'model-GPU{gpu}/checkpoints/') and n.endswith('.pt')]
        assert len(stable)==2 and all(r['bytes']==pin['bytes'] or abs(r['bytes']-pin['bytes'])<1000000 for r in stable)
        report['main_seconds_per_update']=sum(r['seconds'] for r in logs)/len(logs)
        report['main_pairs_per_second']=128/report['main_seconds_per_update']
        report['total_main_pairs']=state['updates']*128
        report['added_main_pairs']=(state['updates']-pin['updates'])*128
        report['order_cursor']=state['cursor']
        report['runtime_main_pairs_field']='order cursor; use updates*128 for lifetime pairs'
        workers.append(report)
    a,b=workers
    assert len(tune_pairs)==2 and tune_pairs[0]==tune_pairs[1]
    sampled_ratio=b['sampled_throughput']['seconds_per_update']/a['sampled_throughput']['seconds_per_update']
    main_ratio=b['main_seconds_per_update']/a['main_seconds_per_update']
    assert all(math.isfinite(r) and r>0 for r in (sampled_ratio,main_ratio))
    output_bytes=sum(r['bytes'] for r in manifest); assert output_bytes<10*1024**3
    return dict(passed=True,ref=server['ref'],session_status=session['status'],
                session_wall_seconds=session['wall_seconds'],small_files_SHA_verified=len(checked),
                output_bytes=output_bytes,large_PT_downloaded=False,workers=workers,modules=modules,
                C_overrides=source_identity['C_overrides'],independent_anchors_and_orders=True,no_OOM=True,
                sampled_time_ratio_GPU1_GPU0=sampled_ratio,main_time_ratio_GPU1_GPU0=main_ratio,
                speed_gate=False,input_preflight=pre,
                source_port_sha256=server['source_port_sha256'],quality_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--metadata',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();r=audit(a.root,a.metadata);Path(a.receipt).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('modules','input_preflight','workers')},indent=2))
