"""Reviewer-only, same production cooldown path: 200 steps/branch, mini K8, FP16, full restore."""
import json,time
from pathlib import Path
import torch
import b5_state as S
import b6_state as D
import b6_runtime as G
import l2_runtime as R
from b3_gpu_smoke import fingerprints
from l2_gpu_smoke import digest
from kaggle_io import write


def exercise(name,cache,order,meta,root,device,cfg,evaluate,resume,pin,cpu=False):
    begin=time.monotonic();root=Path(root)
    assert cfg['N_by_gpu'][cfg['gpu']]>=200
    result=G.run(name,cache,order,meta,root,device,cfg,evaluate,resume,pin,cpu=cpu)
    passed=result['status']=='complete';checks={}
    if result['latest']:
        p=D.read_full(root,pin['identity'],environment=not cpu);m,o,s,state,cd=D.restore(p,device)
        # Exact serialized checkpoint restoration, rather than independent CUDA reruns.
        actual=dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),state=digest(state))
        checks['full_restore_exact']=all(actual[k]==digest(p[k]) for k in actual)
        import exp26_train as T
        checks['all_RNG_exact']=all(digest(T.rng_state()[k])==digest(p['rng'][k]) for k in p['rng'])
        checks['exact_N_zero_LR']=cd['updates']==cd['N'] and cd['last_lr']==0.
        checks['cursor']=cd['cursor']==pin['cursor']+128*cd['N']
        checks['no_main']=state['updates']==pin['updates'] and state['cursor']==pin['cursor'] and state['main_seconds']==pin['main_seconds']
        point=root/(result['FP16_final'] or 'missing')
        if point.is_file():
            fp=torch.load(point,map_location='cpu',weights_only=True)
            checks['FP16_exact']=all(torch.equal(v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu(),fp['model'][k]) for k,v in m.state_dict().items())
        else:checks['FP16_exact']=False
        checks['nonfinite_zero']=cd['nonfinite_skips']==0
        del p,m,o,s;R.release(device)
    passed=passed and bool(checks) and all(checks.values())
    write(root/'gpu-smoke.json',dict(passed=passed,model=name,q=0.,main_added_updates=0,
        source_checkpoint_sha256=pin['sha256'],N=cfg['N_by_gpu'][cfg['gpu']],checks=checks,
        mini_full_evaluation=result['final_complete'],full_test_timer=False,seconds=time.monotonic()-begin))
    if not passed:
        result['status']='partial';result['reason']='GPU smoke checks incomplete';write(root/'summary.json',result)
    return result
