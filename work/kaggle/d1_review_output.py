"""Independent D1 GPU output audit and conservative phase projections."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
from d1_data import verify_receipt, MODELS, WEIGHTS, ROWS, length_bins, ranked_candidates
from kaggle_io import write


def audit(root, destination, metadata):
    root=Path(root); data=root/'d1'
    read=lambda p: json.loads((root/p).read_text())
    summary=read('session-summary.json'); assert summary['status']=='complete' and summary['exit_codes']==[0,0]
    assert summary['wall_seconds']<=660 and not summary['unreaped_worker_pids']
    assert read('startup-gate.json')['passed']
    pre=read('input-preflight.json'); assert pre['all_shard_SHA_verified'] and pre['tokenizer_SHA_verified']
    protocol=read('d1-protocol.json'); cfg=protocol['config']
    assert cfg['owner']=='thisarivson' and cfg['gpu_smoke'] and not cfg['require_secrets'] and not cfg['enable_live_sync']
    server=json.loads(Path(metadata).read_text())
    assert server['is_private'] and server['ref'].endswith('/'+root.name)
    assert cfg['port_sha256']==server['source_port_sha256']
    for name, digest in protocol['modules'].items():
        assert server['modules'][name]==digest
    watchdog = read('watchdog-probe/probe.json') if (root/'watchdog-probe/probe.json').exists() else None
    if root.name == '2':
        assert watchdog['passed'] and watchdog['actual_forced_exit'] and watchdog['durable_artifact_preserved']
        assert read('watchdog-probe/session-summary.json')['watchdog_forced_exit']
        assert read('watchdog-probe/durable-batch.json')['rows']==17
    receipts=list(data.rglob('*.receipt.json'))
    for receipt in receipts: verify_receipt(receipt)
    ids=np.load(data/'scored-row-ids.npy'); assert len(ids)==cfg['smoke_rows'] and np.all(ids[1:]>ids[:-1])
    with np.load(data/'seed-evidence.npz') as z: lengths=z['target_length']
    reports=[]
    for gpu in (0,1):
        shard=data/f'loss-shards/gpu{gpu}'
        report=verify_receipt(shard/'complete.receipt.json'); reports.append(report)
        ce=np.load(shard/'ce.npy',mmap_mode='r'); lens=np.load(shard/'length.npy',mmap_mode='r')
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(ce)),ids[gpu::2])
        np.testing.assert_array_equal(lens[ids[gpu::2]],lengths[ids[gpu::2]]+1)
        parity=read(f'd1/{MODELS[gpu]}/loss-parity.json'); assert parity['passed'] and parity['absolute_delta']<1e-5
    merged=verify_receipt(data/'C-long2x/loss/complete.receipt.json'); assert merged['rows']==len(ids)
    assert not (data/'C-main/loss').exists()
    ce=np.load(data/'C-long2x/loss/ce.npy',mmap_mode='r')
    np.testing.assert_array_equal(np.flatnonzero(np.isfinite(ce)),ids)
    ranked,_=ranked_candidates(ce[ids],ce[ids],lengths[ids],.04)
    with np.load(data/'candidates.npz') as z:
        for b, rr in enumerate(ranked): np.testing.assert_array_equal(z[f'bin{b}'],ids[rr])
    with np.load(data/'selection.npz') as z: selected=z['row_id']
    generation={}; pilots=[]
    for model in MODELS:
        state=read(f'd1/{model}/summary.json'); assert state['status']=='complete' and state['generated_rows']==len(selected)
        ident=read(f'd1/{model}/identity.json'); assert ident['weight_sha256']==WEIGHTS[model]
        pilot=read(f'd1/{model}/pilot.json'); pilots.append(pilot)
        wave_receipts=[p for p in (data/model).glob('*.receipt.json') if p.name.startswith(('pilot-bin-','generation-'))]
        seen=[]; seconds=gpu_seconds=0.; reuse=executed=0; numeric=0
        for receipt in wave_receipts:
            r=verify_receipt(receipt); assert r['complete'] and not r['execution']['error']
            with np.load(receipt.with_name(receipt.name.replace('.receipt.json','.npz'))) as z:
                assert np.all(np.isin(z['row_id'],selected)); seen.extend(z['row_id'].tolist())
                numeric+=int(np.count_nonzero(z['status']=='numeric'))
                if model==MODELS[0]:
                    assert np.all(np.isfinite(z['reference'][z['reference_status']=='numeric']))
                    assert not np.any(z['reference_status']=='budget_not_evaluated')
            seconds+=r['seconds'];gpu_seconds+=r['GPU_seconds'];reuse+=r['reference_reused'];executed+=r['reference_executed']
        np.testing.assert_array_equal(np.sort(seen),np.sort(selected))
        if model==MODELS[0]: assert reuse>0 and executed>0
        generation[model]=dict(rows=len(seen),pipeline_seconds=seconds,GPU_seconds=gpu_seconds,
            pipeline_rows_per_second=len(seen)/seconds,GPU_rows_per_second=len(seen)/gpu_seconds,
            reference_reused=reuse,reference_executed=executed,numeric=numeric)
    counts=np.bincount(length_bins(lengths),minlength=4)
    maxima=np.floor(counts*.04+.5).astype(int)
    costs=[max(p['bins'][b]['seconds']/p['bins'][b]['rows'] for p in pilots) for b in range(4)]
    full_ce_seconds=max((ROWS+1)//2/r['rows_per_second'] for r in reports)
    full_generation_seconds=sum(k*c for k,c in zip(maxima,costs))
    report=dict(passed=True,ref='thisarivson/nlp-hw02-d1-gpu-smoke/'+root.name,session=summary,
        all_receipts_SHA_verified=len(receipts),CUDA_CE_parity=True,no_C_main_CE=True,
        CE_rows_per_second_per_GPU=[r['rows_per_second'] for r in reports],
        full_CE_projected_seconds=full_ce_seconds,CE_safety=1.15,full_CE_conservative_seconds=full_ce_seconds*1.15,
        generation=generation,pilot_seconds_per_row_by_bin=costs,full_generation_projected_seconds=full_generation_seconds,
        generation_safety=1.6,full_generation_conservative_seconds=full_generation_seconds*1.6,
        maximum_bin_quotas=maxima.tolist(),maximum_generation_rows=int(maxima.sum()),smoke_selected_rows=len(selected),
        downloaded_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        secrets_or_live_required=False,models_in_output=bool(list(root.rglob('*.pt'))))
    report['Kaggle_working_watchdog_probe']=watchdog
    assert not report['models_in_output']
    write(destination,report); print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--metadata',required=True,type=Path)
    a=p.parse_args();audit(a.root,a.out,a.metadata)
