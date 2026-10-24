"""One real B2 checkpoint at a time; full restore and exact next batch128 on CPU.

CPU FP32 diagnostic, enabled GradScaler loaded from CUDA, fused Adam disabled after
checking its complete loaded state. CUDA RNG bytes are audited; CUDA execution is
exclusively the reviewer's smoke. No production environment gate is weakened.
"""
import argparse, contextlib, copy, gc, hashlib, json, sys, time
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
import b2_runtime as B
import b4_runtime as G
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
from l1_cache import CompactCache
from l2_gpu_smoke import digest
from kaggle_io import sha, write


def cpu_adapt_optimizer(opt):
    for group in opt.param_groups: group['fused']=False
    # Constructor fused=True enables GradScaler's fused protocol independently
    # of param_groups. Clear it too when doing this CPU-only diagnostic.
    opt._step_supports_amp_scaling=False


def main():
    p = argparse.ArgumentParser()
    for n in ('capture-order','file','cache','order-receipt','out'): p.add_argument('--'+n, type=Path)
    p.add_argument('--gpu',type=int,choices=(0,1)); a = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    entry=json.loads((ROOT/'b4-anchor.json').read_text()); pins=dict(source=entry['source'],anchors={'0':entry['anchor'],'1':entry['anchor']}); start=time.monotonic()
    cache = CompactCache(a.cache, verify=bool(a.capture_order))
    assert cache.manifest_sha == pins['anchors']['0']['identity']['cache_manifest_sha256']
    if a.capture_order:
        order = np.load(a.capture_order, allow_pickle=False, mmap_mode='r')
        assert sha(a.capture_order) == pins['anchors']['0']['identity']['order_sha256']
        assert np.array_equal(np.bincount(order,minlength=len(cache)),1+(cache.target_lengths>=79))
        indices = {str(i):L.take(order,pins['anchors'][str(i)]['cursor'],128).tolist() for i in (0,1)}
        write(a.out,dict(passed=True,order_sha256=sha(a.capture_order),cache_manifest_sha256=cache.manifest_sha,
                         all_cache_SHA_verified=True,next_indices=indices,batch=128))
        print('B4 order/cache SHA and both next batches verified',flush=True); return
    pin = pins['anchors'][str(a.gpu)]; assert a.file.stat().st_size==pin['bytes'] and sha(a.file)==pin['sha256']
    for n,h in pin['identity']['recipe']['implementation_sha256'].items(): assert sha(ROOT/'src'/n)==h,n
    selected = json.loads(a.order_receipt.read_text())
    assert selected['passed'] and selected['all_cache_SHA_verified']
    assert selected['cache_manifest_sha256']==cache.manifest_sha
    assert selected['order_sha256']==pin['identity']['order_sha256']
    ids = np.asarray(selected['next_indices'][str(a.gpu)],dtype=np.uint32)
    device = torch.device('cpu'); fields = ('model','optimizer','scaler','rng','state','scheduler')
    original = None; reports = []
    for route in ('direct full-state control','B4/B2 restore'):
        payload = torch.load(a.file,map_location='cpu',weights_only=True)
        G.validate_anchor(payload,pin)
        initial = {k:digest(payload[k]) for k in fields}
        if original is None: original = initial
        assert initial == original
        if route.startswith('direct'):
            m = C.SimpleTransformer(**payload['model_spec'],dropout=0.)
            o = L.optimizer(m,payload['state']['last_lr'],False)
            s = torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
            state = L.restore_full(payload,m,o,s,device)
        else:
            m,o,s,state = G.restore(payload,device)
        restored = dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),
                        scaler=digest(s.state_dict()),state=digest(state))
        assert all(restored[k]==original[k] for k in restored)
        rng = T.rng_state()
        assert all(digest(rng[k])==digest(payload['rng'][k]) for k in ('torch_cpu','python','numpy'))
        assert payload['rng']['torch_cuda'] and all(t.dtype==torch.uint8 for t in payload['rng']['torch_cuda'])
        # Diagnostic adaptation is identical on both routes, after the complete restore audit.
        cpu_adapt_optimizer(o)
        prior = copy.deepcopy(state); expected_lr = pin['identity']['recipe']['peak_lr']*state['lr_factor']
        assert B.learning_rate(state)==expected_lr and state['warmup_updates']==2000
        del payload; gc.collect()
        with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
            if route.startswith('direct'):
                log,micro = B.O.safe_update(m,o,s,cache,ids,device,state['micro_batch'],expected_lr)
                state.update(updates=state['updates']+1,cursor=state['cursor']+128,micro_batch=micro,last_lr=log['lr'])
                state['main_seconds']+=log['seconds']; state['AMP_skips']+=log['AMP_skip']; state['nonfinite_skips']+=log['nonfinite_skip']
                R.observe(state['stability'],log['nonfinite_skip'],amp_skip=log['AMP_skip'])
            else:
                # Feed the verified real next order batch through the production main_step.
                log,_ = G.main_step(m,o,s,state,cache,ids,device,0.)
        assert not log['AMP_skip'] and not log['nonfinite_skip'] and log['lr']==expected_lr
        assert state['cursor']==prior['cursor']+128 and state['updates']==prior['updates']+1
        state.pop('main_seconds'); prior.pop('main_seconds')
        record = dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),
                      rng=digest(T.rng_state()),state=digest(state),CE=log['CE'],z_loss=log['z_loss'])
        reports.append(record)
        print('B4 CPU',a.gpu,route,round(log['seconds'],2),'s',flush=True)
        del m,o,s; gc.collect()
    assert reports[0]==reports[1], 'Real checkpoint next update differs'
    write(a.out,dict(passed=True,source=pins['source'],model=pin['identity']['model'],checkpoint_sha256=pin['sha256'],
                     checkpoint_bytes=pin['bytes'],updates=pin['updates'],cursor=pin['cursor'],main_seconds=pin['main_seconds'],
                     full_initial_digests=original,restored_fields=list(fields),next_update=reports[1],
                     next_batch_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),steps_bit_exact=1,effective_batch=128,
                     constant_lr=expected_lr,seconds=time.monotonic()-start,GPU=False,
                     limitation='CPU FP32, CPU GradScaler from B2, fused disabled after load audit; CUDA RNG bytes checked, CUDA replay pending reviewer'))


if __name__=='__main__': main()
