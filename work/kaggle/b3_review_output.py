"""Independent B3 CUDA smoke gate; inspect small outputs, never download full PT."""
import argparse, json, math
from pathlib import Path
from b1_review_output import sha
from build_l3 import ROOT


def audit(root, metadata):
    root=Path(root); server=json.loads(Path(metadata).read_text())
    assert server['private'] and server['config']['gpu_smoke']
    modules=json.loads((ROOT/'bundles/b3_thisarivson/source-identity.json').read_text())['modules']
    assert modules==server['modules'], 'Production differs from the actual server smoke'
    manifest=json.loads((root/'b1-output-manifest.json').read_text())['files']
    inventory={r['file']:r for r in manifest}; assert len(inventory)==len(manifest)
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
    assert 'compact-cache-verified' in pre['cache_mount'] and 'b2-states' in pre['resume_mount']
    pins=json.loads((ROOT/'b3-anchors.json').read_text())['anchors']; workers=[]
    for gpu,name,lr in ((0,'W-lr1000',.001),(1,'W-lr600',.0006)):
        folder=root/f'model-GPU{gpu}'; pin=pins[str(gpu)]
        report=json.loads((folder/'gpu-smoke.json').read_text())
        assert report['passed'] and report['model']==name and report['source_checkpoint_sha256']==pin['sha256']
        assert report['main_updates']==report['control_updates']>=200
        audits=json.loads((folder/'restore-audits.json').read_text())
        assert audits['passed'] and audits['branches']==report['restore_audits'] and len(audits['branches'])==2
        required={'model','optimizer','scaler','scaler_enabled','state','training','grad_enabled','inference_mode',
                  'rng_python','rng_numpy','rng_torch_cpu','rng_torch_cuda'}
        for row in audits['branches']:
            assert row['passed'] and not row['different'] and set(row['before'])==required
            assert row['before']==row['restored']
        replay=json.loads((folder/'noeval-replay.json').read_text())
        assert replay==report['noeval_replay'] and replay['updates']>=8
        assert replay['different']==[k for k in replay['control'] if replay['control'][k]!=replay['replay'][k]]
        assert replay['bit_exact']==(not replay['different'])
        trajectory=json.loads((folder/'trajectory-comparison.json').read_text())
        assert trajectory['bit_exact']==report['trajectory_bit_exact']
        assert trajectory['different']==report['trajectory_different']==[k for k in trajectory['control'] if trajectory['control'][k]!=trajectory['continuation'][k]]
        assert report['nonfinite_skips']==0
        for flag in ('probe_final_restore_bit_exact','AMP_overflow_skip','NaN_recovery','constant_lr',
                     'historical_points_sessions_preserved','mini_full_evaluation'):
            assert report[flag],flag
        summary=json.loads((folder/'summary.json').read_text()); state=summary['state']
        assert summary['status']=='complete' and summary['final_complete']
        assert state['updates']==pin['updates']+report['main_updates'] and state['cursor']==pin['cursor']+report['main_updates']*128
        assert state['last_lr']==lr and state['lr_factor']==1. and state['stability']['escalations']==0
        assert state['nonfinite_skips']==0
        latest=json.loads((folder/'latest.json').read_text()); blob=f'model-GPU{gpu}/'+latest['file']
        assert latest==summary['B4_uncooled_latest'] and latest['identity']==pin['identity']
        assert all(latest[k]==inventory[blob][k] for k in ('bytes','sha256'))
        assert latest['cursor']==state['cursor'] and latest['updates']==state['updates']
        logs=[json.loads(v) for v in (folder/'updates.jsonl').read_text().splitlines()]
        assert len(logs)==report['main_updates']
        for i,log in enumerate(logs,1):
            assert log['updates']==pin['updates']+i and log['main_pairs']==pin['cursor']+i*128
            assert log['lr']==lr and log['nonfinite_skip']==0
            assert log['AMP_skip'] or all(math.isfinite(log[k]) for k in ('CE','z_loss'))
        points=list((folder/'points').glob('*/complete.json')); assert len(points)==2
        assert {p.parent.name for p in points}=={r['tag'] for r in audits['branches']}
        for point_path in points:
            point=json.loads(point_path.read_text()); train=point['training']; metrics=point['metrics']
            assert train['cooldown']=='C timed linear' and train['q']==0 and train['start_lr']==lr
            cooldown=[json.loads(v) for v in (point_path.parent/'cooldown-updates.jsonl').read_text().splitlines()]
            assert len(cooldown)==train['completed_updates'] and train['cooldown_seconds']>=train['cooldown_target_seconds']
            elapsed=0.
            for log in cooldown:
                assert math.isclose(log['lr'],lr*(1-min(1.,elapsed/train['cooldown_target_seconds'])),abs_tol=1e-15)
                assert log['nonfinite_skip']==0;elapsed+=log['seconds']
            assert train['cooldown_target_seconds']==(3. if train['final'] else 2.)
            assert train['anchor']['updates']==pin['updates']+(report['main_updates'] if train['final'] else 1)
            for split,n in (('heldout4k',64),('old991',16)):
                row=metrics['splits'][split]
                assert row['exec']['denominator']==n and math.isfinite(row['CE']['CE'])
            if train['final']:
                vote=metrics['K8_tune2048']; assert vote['complete'] and vote['rows']==vote['target_rows']==36
                assert vote['K']==8 and vote['T']==.7 and vote['seed']==270042 and metrics['test']['rows']==64
                batches=[r for p in (point_path.parent/'evaluation/k8-tune2048').glob('batch-*.json') for r in json.loads(p.read_text())]
                assert len(batches)==36 and len({r['id'] for r in batches})==36
                assert all(len(r['samples']['0.7'])==8 and all('execution' in v and 'mean_logprob' in v for v in r['samples']['0.7']) for r in batches)
        workers.append(report)
    output_bytes=sum(r['bytes'] for r in manifest); assert output_bytes<10*1024**3
    return dict(passed=True,ref=server['ref'],session_status=session['status'],
                session_wall_seconds=session['wall_seconds'],small_files_SHA_verified=len(checked),
                output_bytes=output_bytes,large_PT_downloaded=False,workers=workers,modules=modules,
                input_preflight=pre,source_port_sha256=server['source_port_sha256'],quality_gate=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--metadata',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();r=audit(a.root,a.metadata);Path(a.receipt).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('modules','input_preflight')},indent=2))
