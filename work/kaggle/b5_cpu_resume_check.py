"""Real W/C batch128 replay: full restore, stream, two updates and an intervening disk resume.

CPU FP32/no autocast; historical enabled GradScaler, fused disabled only after
checking complete restoration. CUDA RNG bytes checked; GPU gate belongs to reviewer.
"""
import argparse,contextlib,copy,gc,hashlib,json,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
# Import every C dependency through one directory so historical relative SHA gates
# see the original L3 core/decoder as well as the unchanged L1/L2 modules.
_namespace=None
if '--gpu' in sys.argv and sys.argv[sys.argv.index('--gpu')+1]=='1':
    from b5_sources import c_namespace
    _namespace=tempfile.TemporaryDirectory(prefix='b5-c-modules-')
    sys.path.insert(0,str(c_namespace(_namespace.name)))
else:sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import b5_state as S
import b5_resume as M
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
import b2_runtime as B
from l1_cache import CompactCache
from l2_gpu_smoke import digest
from kaggle_io import sha,write


def cpu_optimizer(o):
    for g in o.param_groups:g['fused']=False
    o._step_supports_amp_scaling=False


def normalized(state):
    st=copy.deepcopy(state)
    for k in ('main_seconds','l2_main_seconds'):st.pop(k,None)
    return st


def main():
    p=argparse.ArgumentParser()
    for n in ('file','cache','order-root','order-receipt','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--gpu',choices=['0','1'],required=True);p.add_argument('--capture-order',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);device=torch.device('cpu');start=time.monotonic()
    pin=json.loads((ROOT/'b5-anchors.json').read_text())['anchors'][a.gpu];identity=pin['identity']
    cache=CompactCache(a.cache,verify=a.capture_order);assert cache.manifest_sha==identity['cache_manifest_sha256']
    if a.capture_order:
        order,meta=M.verify_order(a.order_root,cache,pin)
        batches=[L.take(order,pin['cursor']+128*i,128).tolist() for i in range(3)]
        # Include the circular tail, not merely an aligned non-wrapping prefix.
        circular=L.take(order,len(order)-63,256)
        assert np.array_equal(circular,np.concatenate((order[-63:],order[:193])))
        write(a.out,dict(passed=True,gpu=a.gpu,order_sha256=meta['file']['sha256'],cache_manifest_sha256=cache.manifest_sha,
             all_cache_SHA_verified=True,next_batches=batches,origin_cursor=pin['cursor'],batch=128,circular_tail_verified=True,
             modules={n:sha(ROOT/'src'/n) for n in ('b5_state.py','b5_resume.py')},
       C_overrides={p.name:sha(p) for p in (ROOT/'legacy-c').glob('*.py')} if a.gpu=='1' else {}))
        print('B5 order/cache verified',a.gpu,flush=True);return
    assert a.file.stat().st_size==pin['bytes'] and sha(a.file)==pin['sha256']
    r=json.loads(a.order_receipt.read_text());assert r['passed'] and r['all_cache_SHA_verified']
    assert r['order_sha256']==identity['order_sha256'] and r['origin_cursor']==pin['cursor']
    batches=np.asarray(r['next_batches'],dtype=np.uint32);fields=('model','optimizer','scaler','rng','state','scheduler')
    reports=[];restores=[];initial=None
    with tempfile.TemporaryDirectory(prefix='b5-real-replay-') as scratch:
        for route in ('historical uninterrupted control','B5 adapter with disk resume'):
            payload=torch.load(a.file,map_location='cpu',weights_only=True);S.validate(payload,pin,environment=False)
            sealed={k:digest(payload[k]) for k in fields}
            if initial is None:initial=sealed
            assert initial==sealed
            if route.startswith('historical'):
                m=C.SimpleTransformer(**payload['model_spec'],dropout=identity['recipe']['dropout'])
                o=L.optimizer(m,payload['state']['last_lr'],False)
                s=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
                state=L.restore_full(payload,m,o,s,device)
            else:m,o,s,state=S.restore(payload,device)
            def audit(p):
                actual=dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),state=digest(state))
                assert all(actual[k]==digest(p[k]) for k in actual)
                assert all(digest(T.rng_state()[k])==digest(p['rng'][k]) for k in ('python','numpy','torch_cpu'))
                assert p['rng']['torch_cuda'] and all(v.dtype==torch.uint8 for v in p['rng']['torch_cuda'])
                restores.append(dict(route=route,fields_exact=True,CUDA_rng_bytes_sha256=digest(p['rng']['torch_cuda'])))
            audit(payload);opaque_cuda=[v.clone() for v in payload['rng']['torch_cuda']]
            cpu_optimizer(o);del payload;gc.collect();steps=[]
            with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
                for j in range(2):
                    before=copy.deepcopy(state);ids=batches[j]
                    if route.startswith('historical'):
                        update=B.O.safe_update if S.is_w(identity) else R.safe_update
                        log,micro=update(m,o,s,cache,ids,device,state['micro_batch'],state['last_lr'])
                        state.update(updates=state['updates']+1,cursor=state['cursor']+128,micro_batch=micro,last_lr=log['lr'])
                        state['main_seconds']+=log['seconds']
                        if not S.is_w(identity):state['l2_main_seconds']+=log['seconds']
                        state['AMP_skips']+=log['AMP_skip'];state['nonfinite_skips']+=log['nonfinite_skip']
                        R.observe(state['stability'],log['nonfinite_skip'],amp_skip=log['AMP_skip'])
                    else:
                        # Rotate the verified two-batch prefix to feed the production cursor unchanged.
                        order=np.roll(batches[:2].reshape(-1),pin['cursor']%256)
                        assert np.array_equal(L.take(order,state['cursor'],128),ids)
                        log,_=S.main_step(m,o,s,state,cache,order,device,identity)
                    assert not log['AMP_skip'] and not log['nonfinite_skip']
                    assert state['cursor']==before['cursor']+128 and state['updates']==before['updates']+1
                    assert state['last_lr']==(.0006 if a.gpu=='0' else .0015)
                    steps.append(dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),
                       rng=digest(T.rng_state()),state=digest(normalized(state)),CE=log['CE'],z_loss=log.get('z_loss'),
                       batch_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
                    print('B5 real CPU',a.gpu,route,j+1,round(log['seconds'],2),'s',flush=True)
                    if j==0 and not route.startswith('historical'):
                        # Save/read the production format, preserving opaque CUDA RNG
                        # bytes on this CPU-only diagnostic (no CUDA RNG execution).
                        expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),state,T.rng_state()))
                        original_rng=T.rng_state
                        def full_rng():
                            result=original_rng();result['torch_cuda']=opaque_cuda;return result
                        with patch.object(T,'rng_state',side_effect=full_rng):
                            S.save_full(scratch,m,o,s,state,pin['model_spec'],identity)
                        del m,o,s;gc.collect();p=S.read_full(scratch,identity,environment=False)
                        assert digest(p['rng']['torch_cuda'])==digest(opaque_cuda)
                        m,o,s,state=S.restore(p,device);cpu_optimizer(o);del p
                        assert expected==digest((m.state_dict(),o.state_dict(),s.state_dict(),state,T.rng_state()))
            reports.append(steps);del m,o,s;gc.collect()
    assert reports[0]==reports[1], 'Historical continuation differs from serialized B5 replay'
    write(a.out,dict(passed=True,gpu=a.gpu,checkpoint_sha256=pin['sha256'],checkpoint_bytes=pin['bytes'],
       updates=pin['updates'],cursor=pin['cursor'],main_seconds=pin['main_seconds'],initial_full_state_digests=initial,
       restores=restores,steps=reports[1],steps_bit_exact=2,effective_batch=128,disk_resume_between_steps=True,production_serializer=True,opaque_CUDA_rng_preserved=True,
       historical_math=True,constant_lr=.0006 if a.gpu=='0' else .0015,GPU=False,seconds=time.monotonic()-start,
       modules={n:sha(ROOT/'src'/n) for n in ('b5_state.py','b5_resume.py')},
       C_overrides={p.name:sha(p) for p in (ROOT/'legacy-c').glob('*.py')} if a.gpu=='1' else {},
       script_sha256=sha(Path(__file__)),limitation='CPU FP32/no autocast; enabled CPU GradScaler from original state; fused disabled after exact load; CUDA execution pending reviewer'))


if __name__=='__main__':main()
