"""C-main and sampler-only C-long2x forks; one visible GPU each, no worker credentials."""
import argparse, json, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());working=Path(cfg['working_root']);sys.path.insert(0,str(working/'src'))
    import numpy as np, torch
    import l1_runtime as L, l2_runtime as Run, l2_eval as E
    from l2_resume import verify_anchor, branch_identity
    from l1_cache import CompactCache
    from exp26_input import validate_training_inputs
    from kaggle_io import sha, write
    from l2_sampler import load_order
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    smoke=cfg.get('cpu_smoke',False);device=torch.device('cpu' if smoke else 'cuda:0')
    if not smoke:
        assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
        import exp26_core as C
        assert C.fused_adamw_available(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    continuation=cfg.get('session_kind') in ('l3','l4')
    name='C-main' if a.gpu==0 else 'C-long2x'
    if continuation:
        from l3_resume import verify_anchor as verify_l3, run_continuation
        path,anchor=verify_l3(cfg['anchor_root'],cfg['resume_anchors'][name],check_environment=not smoke,
                              expected_points=cfg.get('inherited_points',3))
    else:path,anchor=verify_anchor(cfg['anchor_root'],cfg['anchor_expected'],check_environment=not smoke)
    cache=CompactCache(cfg['compact_cache'],verify=False)
    assert cache.manifest_sha==anchor['identity']['cache_manifest_sha256']==cfg['cache_manifest_sha256']
    order_root=Path(cfg['compact_cache'] if a.gpu==0 else cfg['weighted_order_root'])
    expected_order=anchor['identity']['order_sha256'] if continuation or a.gpu==0 else cfg['weighted_order_sha256']
    order,meta=load_order(order_root,cache,expected_order)
    if continuation:
        identity=anchor['identity']
        if a.gpu==1:assert identity['recipe']['sampler']==meta, 'Strict inherited weighted sampler mismatch'
    elif a.gpu==0:
        assert meta['file']['sha256']==anchor['identity']['order_sha256']
        identity=anchor['identity']
        assert L.identity_for('C',anchor['model_spec'],cache,meta['file']['sha256'])==identity, 'Strict L1 recipe mismatch'
    else:identity=branch_identity(anchor['identity'],meta)
    inputs=Path(cfg['starter_dataset']);full,_,_=validate_training_inputs(inputs)
    refs={s:json.loads((inputs/(s+'-reference-execution.json')).read_text()) for s in full}
    eligible=[i for i,r in enumerate(refs['old1024']['rows']) if r['status']=='numeric'];assert len(eligible)==991
    sets={'heldout4k':full['heldout4k'],'old991':[full['old1024'][i] for i in eligible]}
    refs={'heldout4k':refs['heldout4k'],'old991':{'rows':[refs['old1024']['rows'][i] for i in eligible]}}
    write(working/f'gpu{a.gpu}-old991-index.json',{'original_indices':eligible})
    if cfg.get('gpu_smoke'):
        # Same anchor/full C/batch128/fused/AMP. Only phase durations and eval row counts reduced.
        sets={s:r[:64 if s=='heldout4k' else 16] for s,r in sets.items()}
        refs={s:{'rows':r['rows'][:len(sets[s])]} for s,r in refs.items()}
    if cfg.get('gpu_smoke'):
        if continuation:
            # Reuse L2 CUDA faults, with a wrapper that preserves this L2 anchor's cursor/history.
            from l3_resume import exercise_smoke
            exercise_smoke(a.gpu,name,anchor,identity,cache,order,inputs,sets,refs,device,cfg)
            return
        from l2_gpu_smoke import exercise
        exercise(a.gpu,name,anchor,identity,cache,order,inputs,sets,refs,device,cfg)
        return
    evaluate=lambda point,out,deadline,final:E.evaluate(point,out,inputs,sets,refs,device,deadline,final,
                                                        vote_cap_seconds=cfg.get('vote_cap_seconds',900.),
                                                        test_rows=256 if cfg.get('gpu_smoke') else None)
    if continuation:
        run_continuation(name,anchor,identity,cache,order,working/f'model-{name}',device,cfg['deadline_monotonic'],evaluate,cfg)
    else:
        Run.run(name,anchor,identity,cache,order,working/f'model-{name}',device,cfg['deadline_monotonic'],evaluate,
                **cfg.get('phase_recipe',{}))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        # No raw exception text/env/CLI output. Stable receipt survives even unexpected failure.
        print('WORKER FAILED',type(error).__name__,flush=True);sys.exit(1)
