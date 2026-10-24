"""Greedy paired journals and bounded K8, preserving every executed sample."""
import gc, json, math, time
from pathlib import Path
import torch
from torch import nn
from tokenizers import Tokenizer
import exp26_core as C, exp26_exec as N, exp27_decode as D, exp27b_fast as Fast
import l1_eval as E, s3_representation as R
from kaggle_io import sha, write
from l2_sampler import fixed_split

def scores(records, ids):
    rows = [records[i] for i in ids]; n = len(rows)
    return {'matches': sum(r['correct'] for r in rows), 'denominator': n,
            'accuracy': sum(r['correct'] for r in rows)/n if n else None,
            'length_bins': {label: {'matches':sum(r['correct'] for r in rows if lo <= r['original_bpe_length'] < hi),
                                   'denominator':sum(lo <= r['original_bpe_length'] < hi for r in rows)}
                            for label,lo,hi in [('<60',0,60),('60-90',60,90),('90-120',90,120),('>=120',120,10**9)]}}

def matches(reference, execution):
    return reference['status'] == execution['status'] == 'numeric' and math.isclose(
        reference['value'], execution['value'], rel_tol=1e-6, abs_tol=1e-6)

def greedy_journals(out, inputs, sets, refs):
    original = Tokenizer.from_file(str(Path(inputs)/'tokenizer.json')); records = {}
    for split, rows in sets.items():
        if not (out/(split+'-execution.json')).exists(): continue
        generated = json.loads((out/(split+'-generations.json')).read_text())
        executed = json.loads((out/(split+'-execution.json')).read_text())['rows']
        score = N.compare_exec(refs[split], {'rows':executed}); correct = set(score['match_indices'])
        records[split] = [dict(id=i, question_sha256=__import__('hashlib').sha256(rows[i]['question'].encode()).hexdigest(),
                               correct=i in correct, execution=executed[i], reference=refs[split]['rows'][i],
                               original_bpe_length=len(original.encode(rows[i]['code'],add_special_tokens=False).ids),
                               generated_code=generated[i]['generated_code']) for i in range(len(rows))]
        write(out/(split+'-per-example.json'), records[split])
    return records

@torch.no_grad()
def vote_k8(point, out, inputs, rows, greedy, device, deadline, seed=270042, batch=32, cap=512):
    out = Path(out); out.mkdir(parents=True,exist_ok=True); ids = fixed_split()['tune2048']
    ids = [i for i in ids if i < len(rows)]
    tok, table = R.tokenizer(Path(inputs)/'tokenizer.json',30); model = E.load(point,device)
    completed = []; started = time.monotonic(); lo = 0; attempts = []
    try:
        torch.manual_seed(seed)
        if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
        while lo < len(ids):
            C.check_deadline(deadline); selected = ids[lo:lo+batch]; src=ctx=seq=lps=None
            # Restore the RNG if a CUDA OOM discards a partially sampled batch.
            import exp26_train as T
            rng = T.cpu(T.rng_state())
            try:
                encoded = [[C.BOS]+C.encode(tok,rows[i]['question'])+[C.EOS] for i in selected]
                src = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in encoded],batch_first=True,padding_value=C.PAD).to(device)
                ctx = Fast.Context(model,src,cap); seq,lps,_ = Fast.decode(ctx,k=8,temperature=.7,deadline=deadline)
                candidates = []
                for i,group,means in zip(selected,seq.tolist(),lps.tolist()):
                    samples = []
                    for j,(tokens,lp) in enumerate(zip(group,means)):
                        active = C.active_tokens(tokens); represented = C.decode_bytes(table,active).decode('utf8',errors='replace')
                        try: code = R.restore_compact(represented); error = None
                        except (ValueError,IndexError): code = ''; error = 'invalid representation'
                        samples.append({'sample':j,'code':code,'represented_code':represented,'restore_error':error,
                                        'mean_logprob':float(lp) if math.isfinite(lp) else None,'length':len(active)})
                    candidates.append({'id':i,'question_sha256':greedy[i]['question_sha256'],
                                       'greedy':{'code':greedy[i]['generated_code'],'execution':greedy[i]['execution'],'mean_logprob':0.},
                                       'reference':greedy[i]['reference'],'samples':{'0.7':samples}})
                flat = [c for r in candidates for c in r['samples']['0.7']]
                executions = E.execute([{'generated_code':c['code']} for c in flat],device.type=='cpu',deadline)['rows']
                for candidate,execution in zip(flat,executions):
                    candidate['execution'] = execution
                    candidate['sampling_status'] = 'nonfinite_logprob' if candidate['mean_logprob'] is None else 'complete'
                for r in candidates:
                    valid = dict(r,samples={'0.7':[c for c in r['samples']['0.7'] if c['mean_logprob'] is not None]})
                    winner = D.choose(valid,{'mode':'vote','temperature':.7,'k':8,'greedy_vote':False})
                    r['correct'] = matches(r['reference'],winner['execution']); r['vote_execution'] = winner['execution']
                write(out/f'batch-{lo:05d}.json',candidates); completed.extend(candidates); lo += len(selected)
            except torch.cuda.OutOfMemoryError:
                T.restore_rng(rng,device); attempts.append({'offset':lo,'batch':batch}); batch //= 2
                if batch < 1: raise
            finally:
                del src,ctx,seq,lps; gc.collect()
                if device.type == 'cuda': torch.cuda.empty_cache()
    except C.BudgetExpired:
        pass
    finally:
        del model; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        summary = {'complete':len(completed)==len(ids),'rows':len(completed),'target_rows':len(ids),'K':8,'T':.7,
                   'seed':seed,'matches':sum(r['correct'] for r in completed),'seconds':time.monotonic()-started,
                   'weights_sha256':sha(point),'OOM_attempts':attempts,
                   'logprob':'untempered logsoftmax; includes EOS; PAD/BOS masked, as existing sampler',
                   'vote':'exp27_decode.choose; numeric connected components isclose1e-6, ties count/greedy/mean LP/order',
                   'offline_mixed':'first4 from each worker; C-main greedy tie anchor; same K8, no gold in selection'}
        write(out/'summary.json',summary)
    return summary

def evaluate(point,out,inputs,sets,refs,device,deadline,final=False,smoke=False,vote_cap_seconds=900.,test_rows=None):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    # At most 900s for greedy including CE and complete test; independently bounded vote.
    try:
        metrics = E.evaluate(point,out,inputs,sets,refs,device,min(deadline,time.monotonic()+900.),smoke,test_rows=test_rows)
    finally:
        # CE/test timeout must not discard already executed greedy correctness.
        records = greedy_journals(out,inputs,sets,refs)
    splits = fixed_split()
    metrics['paired_subsets'] = {name:scores(records['heldout4k'],[i for i in ids if i < len(records['heldout4k'])])
                               for name,ids in splits.items()}
    if final:
        metrics['K8_tune2048'] = vote_k8(point,out/'k8-tune2048',inputs,sets['heldout4k'],records['heldout4k'],device,
                                        min(deadline,time.monotonic()+vote_cap_seconds),cap=8 if smoke else 512,
                                        batch=2 if smoke else 32)
    write(out/'metrics.json',metrics); return metrics
