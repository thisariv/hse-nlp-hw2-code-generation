"""Real B2 model/Adam/scaler, CPU control vs probes+restore on short synthetic rows.

This diagnostic creates a separate CPU identity in a temporary directory. It does
not relax B3's production Torch/CUDA/SHA gate or claim to use the next train rows.
"""
import argparse, contextlib, copy, gc, json, random, sys, tempfile, time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import b2_runtime as B
import b3_runtime as G
import exp26_core as C
from exp26_common import json_hash
from b3_gpu_smoke import BranchAudit, control_step, fingerprints
from kaggle_io import sha, write


class ShortCache:
    manifest_sha='B3-real-state-CPU-synthetic-short-batch'
    def __getitem__(self,i):
        return np.asarray([1,3+int(i)%7,2]),np.asarray([8,9])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); start=time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    pin=json.loads((ROOT/'b3-anchors.json').read_text())['anchors']['0']
    assert args.file.stat().st_size==pin['bytes'] and sha(args.file)==pin['sha256']
    for name,h in pin['identity']['recipe']['implementation_sha256'].items(): assert sha(ROOT/'src'/name)==h
    p=torch.load(args.file,map_location='cpu',weights_only=True); G.validate_anchor(p,pin)
    original={k:fingerprints_value(p[k]) for k in ('model','optimizer','scaler','rng','state')}
    # Same complete numerical payload, explicit CPU-only execution settings.
    identity=copy.deepcopy(p['identity']); recipe=identity['recipe']
    identity.update(cpu_test=True,cache_manifest_sha256=ShortCache.manifest_sha)
    recipe.update(torch=str(torch.__version__),CUDA=torch.version.cuda,fused_AdamW=False,AMP='CPU diagnostic')
    identity['recipe_sha256']=json_hash(recipe); p['identity']=identity
    for group in p['optimizer']['param_groups']: group['fused']=False
    device=torch.device('cpu'); cache=ShortCache(); order=np.arange(128,dtype=np.uint32)
    m,o,s,state=G.restore(p,device); del p; gc.collect()
    origin=state['main_seconds']; original_update=B.O.safe_update
    def fixed_seconds(*a,**kw):
        log,micro=original_update(*a,**kw); log['seconds']=1.; return log,micro
    with tempfile.TemporaryDirectory(prefix='b3-real-branch-') as tmp, \
         patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()), \
         patch.object(B.O,'safe_update',side_effect=fixed_seconds):
        root=Path(tmp); anchor=B.save_full(root/'initial',m,o,s,state,pin['model_spec'],identity)
        for _ in range(3):
            log,escalate=control_step(m,o,s,state,cache,order,device)
            assert not escalate and not log['AMP_skip'] and not log['nonfinite_skip']
        control=fingerprints(m,o,s); counters={k:copy.deepcopy(state[k]) for k in ('updates','cursor','stability','last_lr')}
        del m,o,s; gc.collect()
        audit=BranchAudit(root); evaluations=[]
        def evaluate(point,*_):
            q=torch.load(point,map_location='cpu',weights_only=True)
            model=C.SimpleTransformer(**q['model_spec'],dropout=0.).eval(); model.load_state_dict(q['model'])
            with torch.no_grad(): output=model(torch.tensor([[1,3,2]]),torch.tensor([[1,8]]))
            assert torch.isfinite(output).all(); evaluations.append(fingerprints_value(output))
            random.random(); np.random.random(); torch.rand(3)
            return dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3))
                               for k,n in (('heldout4k',4096),('old991',991))},
                        K8_tune2048=dict(complete=True,rows=2048,target_rows=2048),test=dict(rows=1919))
        cfg=dict(owner='CPU diagnostic',main_seconds=3.,probe_main_marks=[origin+1.,origin+2.],
                 deadline_monotonic=time.monotonic()+1200.,io_reserve_seconds=0.,
                 probe_cooldown_seconds=.001,probe_eval_seconds=60.,final_cooldown_seconds=.001,final_eval_seconds=60.)
        result=G.run('W-lr1000',cache,order,{},root/'run',device,cfg,evaluate,root/'initial',anchor,
                     cpu=True,branch_audit=audit)
        assert result['status']=='complete' and len(audit.rows)==3
        m,o,s,state=G.restore(B.read_full(root/'run',identity),device)
        actual=fingerprints(m,o,s); assert actual==control
        assert all(state[k]==v for k,v in counters.items())
        write(args.out,dict(passed=True,GPU=False,source_checkpoint_sha256=pin['sha256'],
                            source_initial_digests=original,model_spec=pin['model_spec'],
                            main_updates_per_route=3,effective_batch=128,control=control,continuation=actual,
                            restore_audits=audit.rows,disposable_eval_output_digests=evaluations,
                            seconds=time.monotonic()-start,
                            limitation='Real B2 full model/Adam/CPU GradScaler; synthetic short rows, FP32/no-autocast, fused=False; evaluation metrics are fixtures, not quality measurements'))
    print('Real B2 CPU: control3 vs two probes+final, all fields bit-exact',flush=True)


def fingerprints_value(value):
    from l2_gpu_smoke import digest
    return digest(value)


if __name__=='__main__': main()
