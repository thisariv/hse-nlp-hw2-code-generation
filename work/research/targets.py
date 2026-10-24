"""Stratified 60k CPU normalization, token and numeric-literal audit."""
import io
import time
import tokenize
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from common import *

def components(code, encoded):
    lines=code.splitlines(keepends=True)
    off=np.r_[0,np.cumsum([len(x) for x in lines])]
    spans={'comments':[], 'blank_lines':[], 'header':[], 'tail_fixed':[]}
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type==tokenize.COMMENT:
                spans['comments'].append((int(off[t.start[0]-1])+t.start[1],int(off[t.end[0]-1])+t.end[1]))
    except (tokenize.TokenError,IndentationError):pass
    for i,l in enumerate(lines):
        if not l.strip():spans['blank_lines'].append((int(off[i]),int(off[i+1])))
    if lines and lines[0]=='def solve() -> int:\n':spans['header'].append((0,len(lines[0])))
    nb=no_blank(no_comments(code));_,mode=strip_frame(nb)
    if mode=='header_tail':
        for i in range(len(lines)-1,-1,-1):
            if lines[i].strip()=='return result':
                spans['tail_fixed'].append((int(off[i]),int(off[i+1])))
                break
        for i in range(len(lines)-1,-1,-1):
            if lines[i].startswith('    result = '):
                spans['tail_fixed'].append((int(off[i]),int(off[i])+len('    result = ')))
                break
    # Fractional token attribution for tokens straddling a region boundary.
    return {k:sum(sum(max(0,min(b,y)-max(a,x)) for x,y in ss)/(b-a)
                  for a,b in encoded.offsets if b>a) for k,ss in spans.items()}

def main():
    started=time.monotonic()
    path=LOCAL/'sample-60k.json'
    if path.exists():rows=read(path)
    else:
        protected=set(q for g in read(INPUT/'protected-questions.json')['groups'].values() for q in g)
        rng=np.random.default_rng(20261012);rows=[]
        for p in sorted(RAW.glob('*.parquet')):
            shard=pq.read_table(p).to_pylist();selected=[]
            for i in rng.permutation(len(shard)):
                r=shard[i]
                if classify(r,protected) is None:
                    selected.append(dict(r,shard=p.name,row=int(i)))
                if len(selected)==1000:break
            rows.extend(selected)
        write(path,rows)
    tok=Tokenizer.from_file(str(INPUT/'tokenizer.json'))
    original=[r['code'] for r in rows]
    a=[no_comments(c) for c in original]
    b=[no_blank(c) for c in a]
    cm=[strip_frame(c) for c in b]
    c=[x[0] for x in cm]
    restored=[restore(x,m) for x,m in cm]
    modes=Counter(m for x,m in cm)
    allcodes=list(dict.fromkeys(original+a+b+restored))
    xp=LOCAL/'normalization-execution.json'
    if xp.exists():execution=read(xp)
    else:
        ex=execute(allcodes);execution=dict(zip(allcodes,ex['rows']));write(xp,execution)
    refs=[execution[x] for x in original]
    eligible=[i for i,x in enumerate(refs) if x['status']=='numeric']
    assert len(eligible)>=50000
    comps=Counter();lengths={};totals={}
    for name,cc in [('original',original),('no_comments',a),('no_blank',b),('no_frame',c)]:
        enc=tok.encode_batch(cc,add_special_tokens=False)
        ll=[len(x.ids) for x in enc];lengths[name]=ll
        totals[name]=dict(tokens=sum(ll),mean=float(np.mean(ll)),median=float(np.median(ll)),p90=float(np.quantile(ll,.9)),
                          finite_tokens=sum(ll[i] for i in eligible))
        if name=='original':
            for text,e in zip(original,enc):comps.update(components(text,e))
    variants={}
    for name,cc in [('no_comments',a),('no_blank',b),('no_frame',restored)]:
        compared=[execution[x] for x in cc]
        bad=[i for i in eligible if not same(refs[i],compared[i])]
        statusdiff=sum(not same(x,y) for x,y in zip(refs,compared))
        variants[name]=dict(finite_comparisons=len(eligible),disagreements=len(bad),rate=len(bad)/len(eligible),
                            all_status_or_value_differences=statusdiff,
                            token_saving_fraction=1-totals[name]['tokens']/totals['original']['tokens'],
                            finite_token_saving_fraction=1-totals[name]['finite_tokens']/totals['original']['finite_tokens'],
                            examples=bad[:10])
    nc=Counter();programs=Counter()
    for i in eligible:
        info=number_info(rows[i]['question'],original[i]);nc.update(info['counts'])
        for k in ['exact_only','placeholders_plus_constants','repeated_question_numbers','repeated_used_value']:
            programs[k]+=info[k]
        programs['programs']+=1
    report=dict(sample_rows=len(rows),seed=20261012,sample='1000 randomly chosen protocol26a-eligible rows per shard; equal shard weight',
                finite_reference=len(eligible),reference_statuses=dict(Counter(x['status'] for x in refs)),
                tokenizer_sha256=hashlib.sha256((INPUT/'tokenizer.json').read_bytes()).hexdigest(),
                tokenizer_vocab=tok.get_vocab_size(),tokens=totals,variants=variants,frame_modes=dict(modes),
                token_components=dict(comps),token_component_fractions={k:v/totals['original']['tokens'] for k,v in comps.items()},
                numbers=dict(literal_counts=dict(nc),literal_fractions={k:v/sum(nc.values()) for k,v in nc.items()},
                             programs=dict(programs),program_fractions={k:v/programs['programs'] for k,v in programs.items() if k!='programs'},
                             constant_allowlist=sorted(CONSTANTS),
                             caveat='categories are deterministic heuristics, not semantic proof; exact-only includes signed AST magnitude'),
                seconds=time.monotonic()-started)
    write(LOCAL/'targets.json',report)
    print(report,flush=True)

if __name__=='__main__':main()
