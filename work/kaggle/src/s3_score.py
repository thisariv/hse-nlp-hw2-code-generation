"""Inference-only B teacher forcing of all saved A candidates, including EOS."""
import gc, json, time
from pathlib import Path
import torch
from torch import nn
import exp26_core as C
from exp26_common import read_json, write_json, sha_file, json_hash

@torch.inference_mode()
def score_batch(model, sources, targets, device):
    assert all(target and target[-1] == C.EOS for target in targets)
    src = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in sources],batch_first=True,padding_value=C.PAD).to(device)
    tgt = nn.utils.rnn.pad_sequence([torch.tensor([C.BOS]+t[:-1]) for t in targets],batch_first=True,padding_value=C.PAD).to(device)
    labels = nn.utils.rnn.pad_sequence([torch.tensor(t) for t in targets],batch_first=True,padding_value=C.PAD).to(device)
    logits = model(src,tgt)
    # Cross entropy promotes logits to FP32 inside autocast and avoids a second full log-softmax tensor.
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        losses = nn.functional.cross_entropy(logits.transpose(1,2),labels,ignore_index=C.PAD,reduction='none')
    sums = -losses.float().sum(1)
    lengths = labels.ne(C.PAD).sum(1)
    return [(float(s),float(s/n),int(n)) for s,n in zip(sums.cpu(),lengths.cpu())]

def flatten(row, tok):
    candidates = [('greedy',0,row['greedy'])]
    assert list(row['samples']) == ['0.7','0.5','1.0']
    assert [len(v) for v in row['samples'].values()] == [32,16,16]
    for temperature, samples in row['samples'].items():
        candidates.extend((temperature,i,c) for i,c in enumerate(samples))
    out=[]
    for index,(temperature,sample_index,candidate) in enumerate(candidates):
        # Rescore the actual saved decoded code, even a cap-stopped candidate, with terminal EOS.
        ids=C.encode(tok,candidate['code'])+[C.EOS]
        out.append({'candidate_index':index,'temperature':temperature,'sample_index':sample_index,
                    'ids':ids,'code_sha256':__import__('hashlib').sha256(candidate['code'].encode()).hexdigest(),
                    'source_eos_emitted':candidate.get('eos_emitted')})
    return out

def run(model_path, spec, candidate_root, sets, tokpath, device, output, deadline, batch=64):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    payload=torch.load(model_path,map_location='cpu',weights_only=True)
    assert payload['format']=='nlp-hw02-model-only-v1'
    model=C.SimpleTransformer(**spec,dropout=0.).to(device).eval()
    model.load_state_dict(payload['model'],strict=True);del payload
    if device.type=='cuda':model.half()
    tok,_=C.init_tokenizer(tokpath)
    summary={'experiment':32,'status':'partial','model_sha256':sha_file(model_path),'model_spec':spec,
             'source':'S2/2 B evaluated t04; S1/1 A all65','score_policy':'untempered logprob of decoded code + EOS; PAD excluded',
             'splits':{},'training_updates':0,'precision':'FP16 forward; FP32 CE accumulation' if device.type=='cuda' else 'FP32'}
    write_json(output/'32-summary.json',summary)
    started=time.monotonic()
    try:
        for split,original in sets.items():
            dest=output/split;dest.mkdir(exist_ok=True)
            count=questions=0;seen=set()
            for path in sorted((Path(candidate_root)/split/'candidates').glob('batch-*.json')):
                if '.sha.' in path.name:continue
                C.check_deadline(deadline)
                assert sha_file(path)==read_json(path.with_suffix('.sha.json'))['sha256']
                rows=read_json(path);pending=[];scored=[]
                for row in rows:
                    i=row['id'];assert i not in seen and 0<=i<len(original) and row['question']==original[i]['question']
                    seen.add(i)
                    src=[C.BOS]+C.encode(tok,row['question'])+[C.EOS]
                    # Exact-code dedup within a question saves compute; every candidate gets its own receipt.
                    unique={}
                    for c in flatten(row,tok):
                        key=tuple(c['ids'])
                        unique.setdefault(key,[]).append({k:v for k,v in c.items() if k!='ids'})
                    for ids,records in unique.items():pending.append((i,src,list(ids),records))
                    questions+=1
                pending.sort(key=lambda item:(len(item[1])+len(item[2]),len(item[2]),item[0]))
                for lo in range(0,len(pending),batch):
                    C.check_deadline(deadline)
                    part=pending[lo:lo+batch]
                    scores=score_batch(model,[x[1] for x in part],[x[2] for x in part],device)
                    for (i,src,ids,records),(total,mean,n) in zip(part,scores):
                        for c in records:
                            scored.append(dict(c,id=i,sum_logprob=total,mean_logprob=mean,length_with_eos=n))
                scored.sort(key=lambda r:(r['id'],r['candidate_index']))
                target=dest/path.name;write_json(target,scored)
                write_json(target.with_suffix('.sha.json'),{'sha256':sha_file(target),'source_sha256':sha_file(path),'scored_candidates':len(scored)})
                count+=len(scored)
                summary['splits'][split]={'questions':questions,'candidates':count,'expected':len(original)*65,
                    'complete':questions==len(original),'seconds_so_far':time.monotonic()-started}
                write_json(output/'32-summary.json',summary)
                print('SCORE B',split,questions,count,flush=True)
            assert seen==set(range(len(original))) and questions==len(original) and count==65*len(original)
        summary['status']='complete';summary['seconds']=time.monotonic()-started
        write_json(output/'32-summary.json',summary)
        return summary
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
