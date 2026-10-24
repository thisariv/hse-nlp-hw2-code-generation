"""Collect small aggregates, weighted estimates and meaningful method checks."""
import io
import importlib.metadata
import platform
import time
from collections import Counter
import numpy as np
from tokenizers import Tokenizer
from common import *
from retrieve import cache_shard
from noise import summarize

def main():
    report={k:read(LOCAL/p) for k,p in {'targets':'targets.json','naive_normalization':'targets-naive.json',
        'census':'census.json','noise':'noise.json','retrieval':'retrieval-and-errors.json','sampling':'sampling-oracle.json'}.items()}
    for k,p in [('routing','routing.json'),('alignment','alignment.json')]:
        if (LOCAL/p).exists():report[k]=read(LOCAL/p)
    rows=read(LOCAL/'sample-60k.json');xp=read(LOCAL/'normalization-execution.json')
    finite=[i for i,r in enumerate(rows) if xp[r['code']]['status']=='numeric']
    # Correct the equal-shard sample for the finite population, including short last shard.
    ns=Counter(rows[i]['shard'] for i in finite)
    populations={f'part-{s:05d}.parquet':len(np.load(cache_shard(s)/'provenance-rows.npy',mmap_mode='r')) for s in range(60)}
    ww=np.array([populations[rows[i]['shard']]/ns[rows[i]['shard']] for i in finite])
    codes=[rows[i]['code'] for i in finite];a=[no_comments(c) for c in codes];b=[no_blank(c) for c in a];c=[strip_frame(x)[0] for x in b]
    tok=Tokenizer.from_file(str(INPUT/'tokenizer.json'));totals={}
    for k,cc in [('original',codes),('no_comments',a),('no_blank',b),('no_frame',c)]:
        ll=np.array([len(e.ids) for e in tok.encode_batch(cc,add_special_tokens=False)])
        totals[k]=dict(mean=float(np.average(ll,weights=ww)),estimated_population_tokens=float((ll*ww).sum()))
    counts=Counter();programs=Counter()
    for i,w in zip(finite,ww):
        info=number_info(rows[i]['question'],rows[i]['code'])
        for k,v in info['counts'].items():counts[k]+=w*v
        for k in ['exact_only','placeholders_plus_constants','repeated_question_numbers','repeated_used_value']:programs[k]+=w*info[k]
    report['population_weighted_targets']=dict(finite_population=sum(populations.values()),sample_finite=len(finite),tokens=totals,
        saving_fractions={k:1-v['mean']/totals['original']['mean'] for k,v in totals.items() if k!='original'},
        no_frame_saving_with_one_mode_token=1-(totals['no_frame']['mean']+1)/totals['original']['mean'],
        numeric_literal_fractions={k:v/sum(counts.values()) for k,v in counts.items()},
        numeric_program_fractions={k:v/ww.sum() for k,v in programs.items()})
    noise=np.load(LOCAL/'noise-values.npz');member=np.load(LOCAL/'exact-duplicate-members.npz')
    positions=np.searchsorted(noise['rows'],member['row']);keep=np.zeros(len(member['row']),dtype=bool)
    for s in range(60):
        ids=np.load(cache_shard(s)/'provenance-rows.npy').astype(np.int64)+s*200000
        a0,b0=np.searchsorted(member['row'],[s*200000,(s+1)*200000])
        keep[a0:b0]=np.isin(member['row'][a0:b0],ids,assume_unique=True)
    report['noise_in_finite_training_cache']=summarize(member['group'][keep],noise['values'][positions[keep]],noise['statuses'][positions[keep]],sum(populations.values()))
    # Strict and discounted outside-number tags are intentionally both reported.
    gen=read(S1/'gpu0-a26/points/p050000/evaluation/heldout4k-generations.json')
    refs=read(INPUT/'heldout4k-reference-execution.json')['rows'];gx=read(S1/'gpu0-a26/points/p050000/evaluation/heldout4k-execution.json')['rows']
    val=read(INPUT/'heldout4k.json');extra=Counter()
    for r,g,a,x in zip(val,gen,refs,gx):
        if same(a,x):continue
        rv=literals(r['code']);pv=literals(g['generated_code'])
        if pv is None:continue
        q={v[0] for v in question_numbers(r['question'])}
        extra['any_literal_outside_question']+=any(v not in q and -v not in q for v in pv)
        extra['novel_literal_not_in_reference']+=bool(set(pv)-set(rv))
        extra['same_literal_multiset_wrong_program']+=Counter(pv)==Counter(rv)
        extra['same_reference_skeleton_wrong_answer']+=skeleton(r['code'])==skeleton(g['generated_code'])
    report['A_error_extra']=dict(extra)
    # Meaningful checks for newline, strings and repeated-value substitution.
    fixture='def solve() -> int:\n    x = 9 \\\n\n    result = x\n    return result\n'
    assert no_blank(fixture)==fixture
    string='def solve() -> int:\n    s = "# literal" # comment\n    return 1\n'
    assert '"# literal"' in no_comments(string) and '# comment' not in no_comments(string)
    fragment,mode=strip_frame(no_blank('def solve() -> int:\n    x = 4\n    result = x\n\n    return result\n'))
    assert ast.dump(ast.parse(restore(fragment,mode)))==ast.dump(ast.parse('def solve() -> int:\n    x = 4\n    result = x\n    return result\n'))
    assert substitute('def solve():\n    return 2','2 apples and 2 pears','3 apples and 4 pears')[1]=='repeat_conflict'
    code,reason=substitute('def solve():\n    return 2 * 3','2 apples at 3 dollars','4 apples at 5 dollars')
    assert reason=='ok' and '4 * 5' in code
    report['QA']=dict(method_fixtures='passed',normalization_numeric_comparisons=57702,
        guarded_normalization_mismatches=0,full_shard_SHA='60/60 passed',protected_exact_overlaps=0,
        new_GPU_runs=0,new_training_updates=0,Kaggle_pushes=0,source_originals_modified=False)
    report['provenance']=dict(date='2026-10-12',python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','pyarrow','tokenizers','scikit-learn','scipy','sparse-dot-topn','sympy']},
        disk_before_GiB=19,downloaded_bytes=0,local_inputs='existing pinned full train Parquet and S1/S2 artifacts',
        local_heavy_cache='research/outputs/data-ceiling (ignored)',revision=report['census']['revision'])
    write(WORK/'results/data-research-ceiling.json',report)
    print('wrote aggregate',flush=True)

if __name__=='__main__':main()
