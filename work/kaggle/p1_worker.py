"""One visible GPU and no credentials; GPU0 pilot, GPU1 reviewer."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(cfg['working_root'])
    sys.path.insert(0,str(root/'src'))
    import numpy as np
    import torch
    import exp26_core as C
    import l2_eval as E
    from l1_cache import CompactCache
    from l2_sampler import load_order
    from l3_resume import verify_anchor
    from exp26_input import validate_training_inputs
    from kaggle_io import sha,write
    from p1_math import ReverseCache, PROTECTED_CACHE_INDICES
    from p1_runtime import run_pilot,run_reviewer
    from p1_eval import PairCache,reverse_ce
    import s3_representation as S
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0')
    assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    assert C.fused_adamw_available(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=CompactCache(cfg['compact_cache'],verify=False)
    assert cache.manifest_sha==cfg['cache_manifest_sha256']
    inputs=Path(cfg['starter_dataset']);full,_,_=validate_training_inputs(inputs)
    refs={s:json.loads((inputs/(s+'-reference-execution.json')).read_text()) for s in full}
    eligible=[i for i,r in enumerate(refs['old1024']['rows']) if r['status']=='numeric'];assert len(eligible)==991
    sets={'heldout4k':full['heldout4k'],'old991':[full['old1024'][i] for i in eligible]}
    refs={'heldout4k':refs['heldout4k'],'old991':{'rows':[refs['old1024']['rows'][i] for i in eligible]}}
    if cfg['gpu_smoke']:
        sets={s:rows[:64 if s=='heldout4k' else 16] for s,rows in sets.items()}
        refs={s:{'rows':r['rows'][:len(sets[s])]} for s,r in refs.items()}
    if a.gpu==0:
        pin=cfg['resume_anchors']['C-long2x']
        path,anchor=verify_anchor(cfg['anchor_root'],pin,expected_points=5)
        order,meta=load_order(cfg['weighted_order_root'],cache,pin['identity']['order_sha256'])
        assert meta==anchor['identity']['recipe']['sampler'];del anchor
        evaluate=lambda point,out,deadline:E.evaluate(point,out,inputs,sets,refs,device,deadline,True,
                                vote_cap_seconds=cfg['vote_cap_seconds'],test_rows=256 if cfg['gpu_smoke'] else None)
        if cfg['gpu_smoke']:
            from p1_gpu_smoke import check_anchor_update
            check_anchor_update(path,pin,cache,order,device,root,cfg)
        result=run_pilot(path,pin,cache,order,root/'model-pilot',device,cfg,evaluate)
        if cfg['gpu_smoke']:
            assert result['status']=='complete' and result['same_stream']
            write(root/'smoke-worker-crash.json',dict(GPU=0,expected_exit=1,primary_checks_passed=True))
            raise RuntimeError('Smoke sibling isolation injection after pilot completed')
    else:
        reverse=ReverseCache(cache)
        order,meta=load_order(cfg['compact_cache'],cache,cfg['main_order_sha256'])
        # Preserve all cache exclusions; strengthen the known normalized test overlap.
        order=np.asarray(order)[~np.isin(order,PROTECTED_CACHE_INDICES)].copy()
        digest=__import__('hashlib').sha256(order.astype('<u4').tobytes()).hexdigest()
        write(root/'reviewer-order.json',dict(order_sha256=digest,original_order_sha256=meta['file']['sha256'],
                    rows=len(order),excluded_indices=list(PROTECTED_CACHE_INDICES),bucketing='inherited ordinary'))
        tok,_=S.tokenizer(inputs/'tokenizer.json',30)
        val={s:ReverseCache(PairCache(rows,tok)) for s,rows in sets.items()}
        def validate(model,deadline,phase):
            metrics={s:reverse_ce(model,c,device,deadline) for s,c in val.items()}
            write(root/'model-reviewer'/('CE-'+phase+'.json'),metrics)
            return metrics
        result=run_reviewer(reverse,order,digest,root/'model-reviewer',device,cfg,validate)
        if cfg['gpu_smoke']:
            marker=root/'smoke-worker-crash.json'
            assert marker.exists() and result['status']=='complete'
            assert result['state']['updates']>=300, 'Reviewer smoke must train hundreds of steps'
            # Validate the exact artifact contract used by research/score-reviewer.py.
            import l2_runtime as R
            import exp26_train as T
            model_root=root/'model-reviewer'
            m,o,s,stable=R.reload_latest(model_root,json.loads((model_root/'identity.json').read_text()),device)
            assert stable['updates']>0 and stable['updates']<result['state']['updates']
            assert stable['cursor']==stable['updates']*128 and stable['last_lr']==.0015
            del m,o,s;R.release(device)
            weight=model_root/'weights/final-fp16.pt'
            artifact=torch.load(weight,map_location='cpu',weights_only=True)
            assert artifact['format']=='nlp-hw02-model-only-v1' and artifact['identity']['model']=='reviewer'
            assert artifact['model_spec']['vocab']==8195 and artifact['optimizer_included'] is False
            m=C.SimpleTransformer(**artifact['model_spec'],dropout=0.).eval().to(device)
            m.load_state_dict(artifact['model'],strict=True);del artifact
            ce=reverse_ce(m,val['heldout4k'],device,cfg['deadline_monotonic']-15)
            assert ce['complete'] and ce['rows']==64
            del m;R.release(device)
            # Parent sees the pilot's nonzero exit while this independent worker trains.
            write(root/'reviewer-smoke.json',dict(passed=True,healthy_worker_completed_after_sibling_crash=True,
                        updates=result['state']['updates'],GPU=True,
                        stable_reload_verified=True,stable_updates=stable['updates'],
                        final_weights_sha256=sha(weight),scorer_artifact_contract=True,reloaded_final_CE=ce,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print('P1 WORKER FAILED',type(error).__name__,flush=True);sys.exit(1)
