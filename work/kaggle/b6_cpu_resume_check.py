"""Real full-state CPU continuation against P1-style uninterrupted q0 control, batch128."""
import argparse, contextlib, copy, gc, json, sys, tempfile, time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
_namespace=None
if '--gpu' in sys.argv and sys.argv[sys.argv.index('--gpu')+1]=='1':
    from b5_sources import c_namespace
    _namespace=tempfile.TemporaryDirectory(prefix='b6-c-modules-')
    sys.path.insert(0,str(c_namespace(_namespace.name)))
else:sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import b5_state as S
import b5_resume as M
import b6_state as D
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
import b2_runtime as B
from p1_math import sqrt_lr
from l1_cache import CompactCache
from l2_gpu_smoke import digest
from kaggle_io import sha,write


def cpu_optimizer(o):
    for g in o.param_groups:g['fused']=False
    o._step_supports_amp_scaling=False


def provenance(p):
    st=copy.deepcopy(p['state'])
    # The final disposable branch only adds these four B5 reporting fields.
    report={k:st.pop(k,None) for k in ('b5_points','b5_sessions','b5_branch_seconds','b5_branch_attempted_updates')}
    return dict(fields={k:digest(p[k]) for k in ('model','optimizer','scaler','rng','scheduler','identity','model_spec')},
                main_state=digest(st),b5_points=len(report['b5_points'] or []),b5_sessions=len(report['b5_sessions'] or []),
                b5_branch_seconds=report['b5_branch_seconds'],b5_branch_attempted_updates=report['b5_branch_attempted_updates'])


def main():
    ap=argparse.ArgumentParser()
    for n in ('file','cache','order-root','order-receipt','out'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--gpu',choices=['0','1'],required=True);ap.add_argument('--capture-order',action='store_true')
    ap.add_argument('--fingerprint-only',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);device=torch.device('cpu');start=time.monotonic()
    pins=json.loads((ROOT/'b6-anchors.json').read_text());pin=pins['anchors'][a.gpu];identity=pin['identity']
    if a.fingerprint_only:
        previous=pins['predecessors'][a.gpu];final=pins['finals'][a.gpu]['anchor']
        assert a.file.stat().st_size==previous['bytes'] and sha(a.file)==previous['sha256']
        p=torch.load(a.file,map_location='cpu',weights_only=True);S.validate(p,previous,environment=False)
        assert p['state']['b5_points'][-1]['training']['anchor']==final
        assert all(previous[k]==final[k] for k in ('updates','cursor','main_seconds','identity','model_spec'))
        write(a.out,dict(passed=True,sha256=previous['sha256'],final_anchor=final,final_anchor_metadata_exact=True,
            pre_final_blob_available=False,provenance=provenance(p),
            limitation='B5 pruned pre-final anchor PT. Exact final receipt/counters/identity + both retained post-restore numeric states checked; no claim of direct pre-final tensor comparison.'))
        print('B6 post-restore predecessor and final anchor metadata verified',a.gpu,flush=True);return
    cache=CompactCache(a.cache,verify=a.capture_order);assert cache.manifest_sha==identity['cache_manifest_sha256']
    if a.capture_order:
        order,meta=M.verify_order(a.order_root,cache,pin)
        batches=[L.take(order,pin['cursor']+128*i,128).tolist() for i in range(3)]
        circular=L.take(order,len(order)-63,256)
        assert np.array_equal(circular,np.concatenate((order[-63:],order[:193])))
        # Verify the WHOLE proposed cooldown stream, including a possible wrap.
        n=pins['budget'][a.gpu]['N'];stream=__import__('hashlib').sha256()
        for j in range(n):stream.update(np.asarray(L.take(order,pin['cursor']+128*j,128),dtype='<u4').tobytes())
        write(a.out,dict(passed=True,gpu=a.gpu,order_sha256=meta['file']['sha256'],cache_manifest_sha256=cache.manifest_sha,
            all_cache_SHA_verified=True,next_batches=batches,origin_cursor=pin['cursor'],batch=128,circular_tail_verified=True,
            N=n,full_cooldown_stream_sha256=stream.hexdigest(),final_cursor=pin['cursor']+128*n));return
    assert a.file.stat().st_size==pin['bytes'] and sha(a.file)==pin['sha256']
    r=json.loads(a.order_receipt.read_text());assert r['passed'] and r['all_cache_SHA_verified']
    assert r['order_sha256']==identity['order_sha256'] and r['origin_cursor']==pin['cursor']
    batches=np.asarray(r['next_batches'],dtype=np.uint32);reports=[];restores=[];initial=None
    with tempfile.TemporaryDirectory(prefix='b6-real-replay-') as scratch:
        for route in ('P1 q0 uninterrupted control','B6 adapter with disk resume'):
            p=torch.load(a.file,map_location='cpu',weights_only=True);S.validate(p,pin,environment=False)
            actual=provenance(p);final=json.loads((ROOT.parent/f'checks/b6-final-anchor-gpu{a.gpu}.json').read_text())['provenance']
            assert actual['fields']==final['fields'] and actual['main_state']==final['main_state'], 'B5 retained states differ numerically'
            assert actual['b5_points']==final['b5_points'] and actual['b5_sessions']==final['b5_sessions']+1
            training=json.loads((ROOT.parent/'results/kaggle-runs/nlp-hw02-b5/1'/f'model-GPU{a.gpu}'/'points'/pins['finals'][a.gpu]['tag']/'complete.json').read_text())['training']
            assert actual['b5_branch_attempted_updates']==final['b5_branch_attempted_updates']
            assert actual['b5_branch_seconds']==final['b5_branch_seconds']
            assert p['state']['b5_points'][-1]['training']==training
            sealed={k:digest(p[k]) for k in ('model','optimizer','scaler','rng','state','scheduler')}
            if initial is None:initial=sealed
            assert initial==sealed
            if route.startswith('P1'):
                m=C.SimpleTransformer(**p['model_spec'],dropout=identity['recipe']['dropout'])
                o=L.optimizer(m,p['state']['last_lr'],False);s=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
                state=L.restore_full(p,m,o,s,device)
            else:m,o,s,state=S.restore(p,device)
            audit={k:digest(v) for k,v in dict(model=m.state_dict(),optimizer=o.state_dict(),scaler=s.state_dict(),state=state).items()}
            assert all(audit[k]==digest(p[k]) for k in audit)
            assert all(digest(T.rng_state()[k])==digest(p['rng'][k]) for k in ('python','numpy','torch_cpu'))
            opaque_cuda=[v.clone() for v in p['rng']['torch_cuda']];assert opaque_cuda
            restores.append(dict(route=route,fields_exact=True,CUDA_rng_bytes_sha256=digest(opaque_cuda)))
            cpu_optimizer(o);del p;gc.collect();steps=[]
            cd=D.new_cooldown(state,pins['budget'][a.gpu]['N'],pin['sha256'])
            with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
                for j in range(1,3):
                    ids=batches[j-1];before=digest(state)
                    if route.startswith('P1'):
                        update=B.O.safe_update if S.is_w(identity) else R.safe_update
                        lr=sqrt_lr(j,cd['N'],cd['peak_lr'])
                        log,micro=update(m,o,s,cache,ids,device,cd['micro_batch'],lr)
                        cd.update(updates=j,cursor=cd['cursor']+128,last_lr=lr,micro_batch=micro)
                        cd['seconds']+=log['seconds'];cd['AMP_skips']+=log['AMP_skip'];cd['nonfinite_skips']+=log['nonfinite_skip']
                        R.observe(cd['stability'],log['nonfinite_skip'],amp_skip=log['AMP_skip'])
                    else:
                        order=np.roll(batches[:2].reshape(-1),pin['cursor']%256)
                        assert np.array_equal(L.take(order,cd['cursor'],128),ids)
                        log,unsafe=D.step(m,o,s,state,cd,cache,order,device,identity);assert not unsafe
                    assert not log['AMP_skip'] and not log['nonfinite_skip'] and 'truncation' not in log
                    assert digest(state)==before and cd['cursor']==pin['cursor']+128*j
                    normalized=copy.deepcopy(cd);normalized.pop('seconds')
                    steps.append(dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),
                        rng=digest(T.rng_state()),main_state=digest(state),cooldown=digest(normalized),CE=log['CE'],
                        z_loss=log.get('z_loss'),lr=log['lr'],batch_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest()))
                    print('B6 real CPU',a.gpu,route,j,round(log['seconds'],2),'s',flush=True)
                    if j==1 and not route.startswith('P1'):
                        expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),state,cd,T.rng_state()))
                        original_rng=T.rng_state
                        def full_rng():
                            rng=original_rng();rng['torch_cuda']=opaque_cuda;return rng
                        with patch.object(T,'rng_state',side_effect=full_rng):D.save_full(scratch,m,o,s,state,pin['model_spec'],identity,cd)
                        del m,o,s;gc.collect();p=D.read_full(scratch,identity,environment=False)
                        assert digest(p['rng']['torch_cuda'])==digest(opaque_cuda)
                        m,o,s,state,cd=D.restore(p,device);cpu_optimizer(o);del p
                        assert expected==digest((m.state_dict(),o.state_dict(),s.state_dict(),state,cd,T.rng_state()))
            reports.append(steps);del m,o,s;gc.collect()
    assert reports[0]==reports[1], 'Uninterrupted P1 q0 control differs from B6 serialized continuation'
    write(a.out,dict(passed=True,gpu=a.gpu,checkpoint_sha256=pin['sha256'],checkpoint_bytes=pin['bytes'],
        final_anchor_sha256=pins['finals'][a.gpu]['anchor']['sha256'],final_anchor_metadata_exact=True,
        post_restore_generations_numerically_exact=True,pre_final_blob_available=False,
        initial_full_state_digests=initial,restores=restores,steps=reports[1],steps_bit_exact=2,effective_batch=128,
        N=cd['N'],main_added_updates=0,inherited_main_state_unchanged=True,q=0.,disk_resume_between_steps=True,
        production_serializer=True,opaque_CUDA_rng_preserved=True,GPU=False,seconds=time.monotonic()-start,
        modules={n:sha(ROOT/'src'/n) for n in ('b5_state.py','b5_resume.py',*D.MODULES)},
        C_overrides={p.name:sha(p) for p in (ROOT/'legacy-c').glob('*.py')} if a.gpu=='1' else {},
        script_sha256=sha(Path(__file__)),limitation='CPU FP32/no autocast; original enabled scaler; fused disabled after exact load; CUDA execution belongs to reviewer'))


if __name__=='__main__':main()
