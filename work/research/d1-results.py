"""Train-only CPU audit, precision proxies and reproducible D1 manual sample.

Run analyze-d1-labels.py first. Large artifacts stay under ignored results/d1-labels.
CPU replay recovers inspectable code, NOT the missing original T4 token sequences.
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
RUN = WORK/'results/kaggle-runs/nlp-hw02-kaggle-d1/1'
OUT = WORK/'results/d1-labels'
sys.path.insert(0, str(WORK/'kaggle/src'))
from d1_data import close, verify_receipt, ranked_candidates, CACHE_SHA, WEIGHTS, EDGES
from kaggle_io import sha, write


def load():
    spec = importlib.util.spec_from_file_location('d1_analysis', Path(__file__).with_name('analyze-d1-labels.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = RUN/'d1'
    with np.load(root/'seed-evidence.npz', allow_pickle=False) as z:
        seed = {k:z[k] for k in z.files}
    with np.load(OUT/'flags.npz', allow_pickle=False) as z:
        flags = {k:z[k] for k in z.files}
    ids = flags['row_id']
    data = {}
    for name in WEIGHTS:
        rows, _ = module.read_generation(root/name)
        data[name] = {k:np.asarray([rows[int(i)][k] for i in ids]) for k in next(iter(rows.values()))}
    return seed, flags, data


def wilson(k, n):
    if not n: return None
    z = 1.959963984540054; p = k/n; d = 1+z*z/n
    mid = (p+z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(mid-half), float(mid+half)]


def analyze():
    started = time.monotonic(); seed, f, data = load(); ids = f['row_id']; root = RUN/'d1'
    receipts = list(root.rglob('*.receipt.json'))
    for p in receipts: verify_receipt(p)
    ce = np.load(root/'C-long2x/loss/ce.npy', mmap_mode='r', allow_pickle=False)
    seen = np.zeros(len(ce), bool); nll = 0.; tokens = 0
    for p in sorted((root/'loss-shards').rglob('block-*.receipt.json')):
        meta = json.loads(p.read_text())
        with np.load(p.parent/meta['files'][0]['file'], allow_pickle=False) as z:
            rr = z['row_id']; assert not seen[rr].any(); seen[rr] = True
            assert np.array_equal(ce[rr], z['ce'])
            assert np.array_equal(z['length'], seed['target_length'][rr]+1)
        nll += meta['sum_NLL']; tokens += meta['tokens']
    assert seen.all()
    loss = json.loads((root/'C-long2x/loss/complete.receipt.json').read_text())
    assert tokens == loss['tokens'] and abs(nll-loss['sum_NLL']) < .001
    bins = np.searchsorted(EDGES, seed['target_length'], side='right')
    def distribution(mask):
        a = ce[mask].astype(np.float64)
        q = np.quantile(a, [0,.01,.1,.25,.5,.75,.9,.95,.96,.98,.99,.995,.999,1])
        return dict(rows=len(a), mean=float(a.mean()), std=float(a.std()),
                    quantiles=dict(zip(['min','p1','p10','p25','p50','p75','p90','p95','p96','p98','p99','p99.5','p99.9','max'],q.tolist())))
    a,b = data['C-main'],data['C-long2x']; eligible = f['eligible']
    both_ref = eligible & close(a['answer'],a['reference']) & close(b['answer'],a['reference'])
    categories = dict(both_against_reference=int(f['A'].sum()), both_match_reference=int(both_ref.sum()),
                      other=int((~(f['A']|both_ref)).sum()), eligible_other=int((eligible & ~(f['A']|both_ref)).sum()),
                      ineligible=int((~eligible).sum()))
    pos = np.searchsorted(seed['row_id'], ids); found = pos < len(seed['row_id'])
    jj = np.flatnonzero(found); found[jj] = seed['row_id'][pos[jj]] == ids[jj]
    majority = np.full(len(ids), np.nan); group = np.full(len(ids), -1, np.int64)
    minority = np.zeros(len(ids), bool)
    for out,key in [(majority,'majority'),(group,'group'),(minority,'minority')]: out[found] = seed[key][pos[found]]
    strict = np.isfinite(majority)
    support = strict & close(a['answer'],majority) & close(b['answer'],majority)
    ref_mode = strict & close(a['reference'],majority)
    def precision(mask):
        use = mask & strict; k = int((use & support).sum()); n = int(use.sum())
        unique,inverse=np.unique(group[use],return_inverse=True)
        group_n=np.bincount(inverse,minlength=len(unique))
        group_k=np.bincount(inverse,weights=support[use],minlength=len(unique))
        interval=None
        if n:
            samples=np.random.default_rng(2026101503).integers(0,len(unique),(2000,len(unique)))
            boot=group_k[samples].sum(1)/group_n[samples].sum(1)
            interval=np.quantile(boot,[.025,.975]).tolist()
        return dict(candidates=int(mask.sum()), strict_mode_candidates=n, model_matches_mode=k,
                    reference_matches_mode=int((use & ref_mode).sum()), neither_matches_mode=int((use & ~(support|ref_mode)).sum()),
                    precision=k/n if n else None, row_wilson95=wilson(k,n),
                    group_bootstrap95=interval,groups=len(unique))
    thresholds = {'all_A':precision(f['A'])}
    rank_boundaries = {}
    for fraction in [.001,.0025,.005,.01,.02,.04]:
        ranked,_ = ranked_candidates(ce,ce,seed['target_length'],fraction)
        keep = np.zeros(len(ce),bool)
        cuts = []
        for ii in ranked:
            keep[ii] = True; cuts.append(dict(rows=len(ii),ce_min=float(ce[ii[-1]]),last_row=int(ii[-1])))
        key = f'top_{100*fraction:g}_percent_by_bin'
        thresholds[key] = precision(f['A'] & keep[ids]); rank_boundaries[key] = cuts
    for t in [.5,1,1.5,2,2.5,3,4]: thresholds[f'CE_ge_{t:g}'] = precision(f['A'] & (ce[ids]>=t))
    # C here is A + mode support. Historical B+support equals it on this subset.
    c = f['A'] & support & minority
    assert np.array_equal(c, f['C'])
    thresholds['C'] = precision(c)
    thresholds['C'].update(precision_by_definition=True,row_wilson95=None,group_bootstrap95=None)
    full_minority = seed['minority'] & np.isfinite(seed['majority'])
    selected_minority = minority & strict
    rng = np.random.default_rng(2026101501)
    sample_ids = np.sort(rng.choice(ids[f['A']],30,replace=False))
    np.save(OUT/'manual-row-ids.npy',sample_ids,allow_pickle=False)
    generation = {}
    for name in WEIGHTS:
        metas = [json.loads(p.read_text()) for p in (root/name).glob('*.receipt.json')]
        metas = [m for m in metas if m.get('phase')=='generation']
        sec = sum(m['seconds'] for m in metas); gpu = sum(m['GPU_seconds'] for m in metas)
        generation[name] = dict(rows=sum(m['rows'] for m in metas),seconds=sec,gpu_seconds=gpu,
                               rows_per_second=len(ids)/sec,gpu_rows_per_second=len(ids)/gpu,
                               reference_reused=sum(m['reference_reused'] for m in metas),
                               reference_executed=sum(m['reference_executed'] for m in metas))
    minority_ce_coverage={}
    for percent in [4,10,20,50,100]:
        counts=[]
        for bin_id in range(4):
            # Include all FP16 boundary ties: these are approximate percentile pools.
            boundary=float(np.quantile(ce[bins==bin_id],1-percent/100))
            counts.append(int((full_minority & (bins[seed['row_id']]==bin_id) &
                               (ce[seed['row_id']]>=boundary)).sum()))
        minority_ce_coverage[str(percent)]=dict(rows=sum(counts),by_bin=counts,include_boundary_ties=True)
    slow_rate=min(d['rows_per_second'] for d in generation.values())
    expansion={}
    for name,total in [('finish_top4',389029),('top10',round(len(ce)*.1)),
                       ('top20',round(len(ce)*.2)),('all_train',len(ce))]:
        remaining=total-len(ids)
        expansion[name]=dict(additional_rows=remaining,measured_rate_scenario_hours=remaining/slow_rate/3600,
                             safety1_6_plus10min_hours=remaining/slow_rate*1.6/3600+1/6)
    expansion['unscanned_strict_minority']=dict(additional_rows=int(full_minority.sum()-selected_minority.sum()),
        measured_rate_scenario_minutes=float((full_minority.sum()-selected_minority.sum())/slow_rate/60),
        practical_budget_minutes=[10,20])
    report = dict(format='nlp-hw02-d1-results-v1',GPU=False,cache_manifest_sha256=CACHE_SHA,
                  audit=dict(receipts=len(receipts),all_payload_SHA_verified=True,loss_shards_bijection=True,
                             sum_NLL=nll,tokens=tokens,token_CE=nll/tokens),
                  ce=dict(all=distribution(np.ones(len(ce),bool)),by_compact_bin={str(i):distribution(bins==i) for i in range(4)}),
                  paired_rows=len(ids),eligible_rows=int(eligible.sum()),categories=categories,
                  statuses={name:{key:dict(Counter(map(str,d[key]))) for key in ('status','generation_status')} for name,d in data.items()},
                  reference_statuses=dict(Counter(map(str,a['reference_status']))),
                  thresholds=thresholds,rank_boundaries=rank_boundaries,
                  by_bin={str(i):dict(paired=int((bins[ids]==i).sum()),A=precision(f['A'] & (bins[ids]==i))) for i in range(4)},
                  coverage=dict(train_rows=len(ce),paired_fraction=len(ids)/len(ce),top4_rows=389029,
                                fraction_of_top4=len(ids)/389029,strict_minority_total=int(full_minority.sum()),
                                strict_minority_scanned=int(selected_minority.sum()),strict_minority_confirmed=int(c.sum()),
                                strict_minority_unscanned=int(full_minority.sum()-selected_minority.sum()),
                                A_in_strict_minority=int((f['A'] & selected_minority).sum()),
                                minority_ce_coverage=minority_ce_coverage),
                  expansion_budget=expansion,
                  generation=generation,manual_sample=dict(seed=2026101501,rule='A',rows=sample_ids.tolist(),
                  design='simple random sample without replacement from all A; sorted only for display'),
                  limitations=['Duplicate mode is a conditional proxy, not independent gold; correlated rows.',
                               'Original T4 generated code/tokens absent; CPU code replay must be labelled.',
                               'No random low-CE greedy; total train noise and recall are not identified.'],
                  seconds=time.monotonic()-started)
    write(OUT/'diagnostics.json',report)
    write(WORK/'results/d1-results.json',report)
    print(json.dumps(dict(categories=categories,thresholds=thresholds,coverage=report['coverage'],generation=generation),indent=2))


def review():
    # Prefer exact run sources for model/decoder/restore/executor replay.
    sys.path.insert(0,str(RUN/'src'))
    import torch
    from torch import nn
    import final_core as Core
    import final_fast as Fast
    import final_speed as Speed
    from final_representation import tokenizer,restore_compact
    from final_executor_v6 import Executor
    from l1_cache import CompactCache
    seed,f,data = load(); ids = np.load(OUT/'manual-row-ids.npy',allow_pickle=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cache = CompactCache(WORK/'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache',verify=False)
    tokpath = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
    assert sha(tokpath)==cache.manifest['original_tokenizer_sha256']
    _,table = tokenizer(tokpath,30)
    rows = []
    for row in ids:
        i = int(np.searchsorted(f['row_id'],row)); src,tgt=cache[int(row)]
        rows.append(dict(row_id=int(row),CE=float(f['mean_CE'][i]),
                         question=Core.decode_bytes(table,src).decode(),
                         reference_code=restore_compact(Core.decode_bytes(table,tgt).decode()),
                         reference_answer=float(data['C-main']['reference'][i]),
                         D1_answers={name:float(d['answer'][i]) for name,d in data.items()},cpu_replay={}))
    write(OUT/'manual-review-data.json',dict(rows=rows,provenance='original reference; model code pending CPU replay'))
    executor = Executor(2)
    try:
        refs=[dict(code=r['reference_code']) for r in rows]; executor.submit(refs)
        for name,expected in WEIGHTS.items():
            started=time.monotonic()
            path=WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'/f'model-{name}__weights__l2-62280-fp16.pt'
            assert sha(path)==expected
            payload=torch.load(path,map_location='cpu',weights_only=True)
            model=Core.SimpleTransformer(**payload['model_spec'],dropout=0.).eval()
            model.load_state_dict(payload['model'],strict=True); del payload
            jobs=[]
            with torch.inference_mode():
                for lo in range(0,len(ids),10):
                    batch=ids[lo:lo+10]
                    src=nn.utils.rnn.pad_sequence([torch.tensor(np.asarray(cache[int(i)][0]).copy(),dtype=torch.long) for i in batch],batch_first=True,padding_value=0)
                    ctx=Fast.Context(model,src,512)
                    seq,_,_=Speed.decode(ctx,compute_logprob=False,options=Speed.Options(share_cross=True,async_eos=True,eos_every=32,compact_done=True,race_sampler=True,lean_greedy=True))
                    for r,tokens in zip(rows[lo:lo+10],seq[:,0].tolist()):
                        represented=Core.decode_bytes(table,tokens).decode('utf8',errors='replace')
                        code=restore_compact(represented)
                        job=dict(code=code); jobs.append(job)
                        r['cpu_replay'][name]=dict(code=code,EOS=2 in tokens)
                    executor.submit(jobs[-len(batch):])
                    print(name,'generated',min(lo+10,len(ids)),flush=True)
                    del ctx,src,seq
            executor.drain(time.monotonic()+180); executor.attach(jobs)
            for r,job in zip(rows,jobs):
                execution=job['execution']; result=r['cpu_replay'][name]
                result['execution']=execution
                result['matches_D1_answer']=bool(execution.get('status')=='numeric' and close(execution['value'],r['D1_answers'][name]))
            del model
            write(OUT/'manual-review-data.json',dict(rows=rows,model_code_provenance='CPU FP32 replay of pinned FP16 weights; original T4 code absent',last_model_seconds=time.monotonic()-started))
        executor.drain(time.monotonic()+180);executor.attach(refs)
        for r,job in zip(rows,refs):
            r['reference_reexecution']=job['execution']
            assert job['execution']['status']=='numeric' and close(job['execution']['value'],r['reference_answer'])
        write(OUT/'manual-review-data.json',dict(rows=rows,model_code_provenance='CPU FP32 replay of pinned FP16 weights; original T4 code absent'))
    finally: executor.close()
    print('manual replay complete',flush=True)


def orders(rule='C'):
    """Filter pinned inherited permutations, preserving their 1x/2x counts."""
    import hashlib
    from l1_cache import CompactCache
    from l2_sampler import load_order
    seed,f,_=load()
    cache_root=WORK/'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'
    cache=CompactCache(cache_root,verify=False)
    assert cache.manifest_sha==CACHE_SHA
    pins=json.loads((WORK/'kaggle/r1-anchors.json').read_text())
    sources={'C-main':cache_root,'C-long2x':WORK/'kaggle/staging/r1-states/weighted-order__order.npy'}
    keep=np.ones(len(cache),bool);keep[seed['protected']]=False
    if rule=='C':keep[f['row_id'][f['C']]]=False
    kept=np.flatnonzero(keep).astype('<u4'); selected_sha=hashlib.sha256(kept.tobytes()).hexdigest()
    result=dict(rule='C: A + strict n>=3 unique >=2/3 duplicate mode; unknown-mode rows retained' if rule=='C' else 'protected-only control; no D1 exclusions',
                excluded_by_C=int(f['C'].sum()) if rule=='C' else 0,protected=seed['protected'].tolist(),
                kept_unique_rows=len(kept),selection_sha256=selected_sha,
                evidence_sha256=sha(OUT/'analysis.json'),orders={},private=True,uploaded=False)
    for name,source in sources.items():
        expected=pins['anchors'][name]['identity']['order_sha256']
        if source.is_file():
            base=json.loads(source.with_suffix('.json').read_text())
            assert base==pins['weighted_order']
            assert sha(source)==base['file']['sha256']==expected and source.stat().st_size==base['file']['bytes']
            original=np.load(source,mmap_mode='r',allow_pickle=False)
            assert original.dtype==np.uint32 and original.ndim==1 and len(original)==base['slots']
        else:
            original,base=load_order(source,cache,expected)
        counts=np.bincount(original,minlength=len(cache))
        expected_counts=np.ones(len(cache),np.int64)
        if name=='C-long2x':expected_counts+=(cache.target_lengths>=79)
        assert np.array_equal(counts,expected_counts)
        order=np.asarray(original[keep[original]]).copy()
        assert np.array_equal(np.bincount(order,minlength=len(cache)),counts*keep)
        dest=OUT/f'r1-{rule}'/name;dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'order.npy',order,allow_pickle=False)
        metadata=dict(rows=len(cache),slots=len(order),batch=128,chunk=4096,rule='D1-C' if rule=='C' else 'protected-control',
                      cache_manifest_sha256=CACHE_SHA,selected_cache_indices_sha256=selected_sha,
                      seed=base.get('seed',2026101201),parent_order_sha256=expected,
                      order_policy='stable filtering of inherited order; R1 reshuffles at seed 2026101701',
                      selected_unique_rows=len(kept),excluded_unique_rows=int((~keep).sum()),
                      sampling='uniform' if name=='C-main' else 'long2x',
                      evidence_sha256=result['evidence_sha256'],
                      file=dict(file='order.npy',bytes=(dest/'order.npy').stat().st_size,sha256=sha(dest/'order.npy')))
        if name=='C-long2x':
            metadata['filtered_multiplicity']=dict(format='nlp-hw02-filtered-long2x-v1',threshold=79,short=1,long=2)
        write(dest/'order.json',metadata)
        (dest/'split.json').write_bytes((OUT/'C/split.json').read_bytes())
        loaded,_=load_order(dest,cache,metadata['file']['sha256'])
        assert np.array_equal(loaded,order)
        result['orders'][name]=dict(path=str(dest.relative_to(WORK)),**metadata,
                                   excluded_slots=len(original)-len(order))
    write(OUT/f'r1-{rule}/manifest.json',result)
    write(WORK/('checks/d1-filtered-orders.json' if rule=='C' else 'checks/d1-control-orders.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['analyze','review','orders']);args=p.parse_args()
    if args.command=='orders':
        orders();orders('control')
    else:dict(analyze=analyze,review=review)[args.command]()
