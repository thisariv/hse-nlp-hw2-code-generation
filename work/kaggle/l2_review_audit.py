#!/usr/bin/env python3
"""Verify downloaded immutable CUDA smoke output and source; no credentials or training."""
import argparse, gc, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import torch
import l2_runtime as R, l2_sampler as S, l2_eval as E
from l2_compare import mixed, compare
from kaggle_io import sha, write

def audit(out, source):
    out,source=Path(out),Path(source);manifest=json.loads((out/'l2-output-manifest.json').read_text());omitted=[]
    for rec in manifest['files']:
        p=out/rec['file']
        if not p.exists() and rec['file']=='__notebook__.ipynb':omitted.append(rec['file']);continue
        assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],rec['file']
    meta=json.loads((source/'kernel-metadata.json').read_text())
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
    published=json.loads((source/'nlp-hw02-l2-gpu-smoke.ipynb').read_text())
    local=json.loads((ROOT/'notebooks/nlp-hw02-l2-gpu-smoke.ipynb').read_text())
    assert [(c['cell_type'],c['source']) for c in local['cells']]==[(c['cell_type'],c['source']) for c in published['cells']]
    cfg=json.loads((out/'session-config.json').read_text());pin=json.loads((ROOT/'l2-anchor.json').read_text())
    assert cfg['anchor_pin']==pin and cfg['cache_source'].endswith('/2')
    assert json.loads((out/'anchor-preflight.json').read_text())['sha256']==pin['sha256']
    smoke=json.loads((out/'cuda-smoke-summary.json').read_text());summary=json.loads((out/'session-summary.json').read_text())
    assert smoke['passed'] and summary['exit_codes']==[1,0] and summary['failed_workers']==[0]
    assert not summary['unreaped_worker_pids'] and not (out/'stop-request.json').exists()
    assert (out/'smoke-worker-crash.json').exists()
    models={};journals={};greedy={}
    for gpu,name in enumerate(('C-main','C-long2x')):
        root=out/('model-'+name);identity=json.loads((root/'identity.json').read_text())
        report=json.loads((root/'gpu-smoke.json').read_text());assert report['passed'] and report['micro_batch']==64
        assert report['checks']['bit_exact_L1_CUDA_update'] and report['checks']['serialized_resume_bit_exact_CUDA']
        assert report['main_added_updates']>=300 and report['main_seconds']>=300
        assert len(list((root/'checkpoints').glob('*.pt')))==2
        for p in (root/'checkpoints').glob('*.pt'):
            payload=R.read_full(p,identity,check_environment=False)
            assert payload['scaler_enabled'] and len(payload['rng']['torch_cuda'])==1
            assert all(v.dtype==torch.float32 for v in payload['model'].values() if v.is_floating_point())
            del payload;gc.collect()
        receipt=json.loads((root/'latest.json').read_text());payload=R.read_full(root/receipt['file'],identity,check_environment=False)
        state=payload['state'];assert state['last_lr']==.0015*(.5 if gpu==0 else 1.)
        assert state['stability']['escalations']==(1 if gpu==0 else 0)
        assert state['cursor_origin']==(0 if gpu==0 else -pin['updates']*128)
        assert len(state['l2_points'])==2 and all(p['complete'] for p in state['l2_points'])
        if gpu==0:assert identity==pin['identity']
        else:assert identity['order_sha256']==cfg['weighted_order_sha256'] and identity['parent_recipe_sha256']==pin['identity']['recipe_sha256']
        weights=list((root/'weights').glob('*.pt'));assert len(weights)==2
        for p in weights:
            value=torch.load(p,map_location='cpu',weights_only=True)
            assert value['optimizer_included'] is False and 'optimizer' not in value
            assert all(v.dtype==torch.float16 and bool(torch.isfinite(v).all()) for v in value['model'].values() if v.is_floating_point())
            del value;gc.collect()
        greedy[name]={}
        for item in state['l2_points']:
            evaluation=root/'points'/item['tag']/'evaluation'
            rows=json.loads((evaluation/'heldout4k-per-example.json').read_text())
            assert len(rows)==64 and all(r['correct']==E.matches(r['reference'],r['execution']) for r in rows)
            greedy[name][item['tag']]=[(r['id'],r['question_sha256']) for r in rows]
            test=item['metrics']['test'];assert test['rows']==256 and not test['cache_reuse'] and test['accuracy'] is None
            assert test['generation']['precision']=='FP16' and test['generation']['SDPA_dimensions']==4
        final=root/'points'/state['l2_points'][-1]['tag']/'evaluation/k8-tune2048'
        rows=[r for p in sorted(final.glob('batch-*.json')) for r in json.loads(p.read_text())]
        assert [r['id'] for r in rows]==[i for i in S.fixed_split()['tune2048'] if i<64]
        assert all(len(r['samples']['0.7'])==8 for r in rows)
        assert all(c['execution']['status'] and math.isfinite(c['mean_logprob']) and c['sampling_status']=='complete'
                   for r in rows for c in r['samples']['0.7'])
        journals[name]=final
        log=(out/f'gpu{gpu}-process.log').read_text();assert 'Traceback' not in log
        assert ('WORKER FAILED RuntimeError' in log) if gpu==0 else ('WORKER FAILED' not in log)
        models[name]={'micro_batch':report['micro_batch'],'probe':report['probe'],'peak_reserved_bytes':report['peak_reserved_bytes'],
                      'peak_allocated_bytes':report['peak_allocated_bytes'],'main_added_updates':report['main_added_updates'],
                      'latest_updates':state['updates'],'latest_cursor':state['cursor'],'LR':state['last_lr'],
                      'test256_seconds':[p['metrics']['test']['full_pipeline_seconds'] for p in state['l2_points']],
                      'K8':report['checks']['K8_executed_samples_and_greedy256'],'checks':report['checks']}
        del payload;gc.collect()
    assert greedy['C-main']==greedy['C-long2x']
    offline=mixed(journals['C-main'],journals['C-long2x'])
    return {'passed':True,'server_status':'COMPLETE','internal_status':'partial expected injected fault',
            'expected_fault_exit_codes':[1,0],'wall_seconds':smoke['wall_seconds'],'models':models,
            'source_exact':True,'private':True,'manifest_files_verified':len(manifest['files'])-len(omitted),
            'manifest_service_files_unavailable':omitted,'output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            'full_states_and_FP16_weights_downloaded_and_verified':True,'paired':compare(out),'offline_mixed_vote':offline,
            'live_sync_tested':False,'live_sync_reason':'API smoke has no Secrets; production startup probe is mandatory'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source',required=True)
    p.add_argument('--report',required=True);a=p.parse_args();torch.set_num_threads(1)
    result=audit(a.output,a.source);write(a.report,result)
    print(json.dumps({k:result[k] for k in ('passed','wall_seconds','manifest_files_verified','output_bytes')}))
