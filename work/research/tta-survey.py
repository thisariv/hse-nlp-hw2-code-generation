"""Stratified train-only surface-form survey; heldout questions only for coverage."""
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'kaggle/src'))
from final_representation import tokenizer
from final_split import fixed_split
from tta_rules import RULES, apply, numbers

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cache = WORK/'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    manifest = json.loads((cache/'compact-manifest.json').read_text())
    tok, table = tokenizer(inputs/'tokenizer.json', 30)
    patterns = {
        'grouped': r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?',
        'ungrouped-4plus': r'(?<![\w.,])\d{4,}(?![\w.,])',
        'dollar-prefix': r'\$\d', 'dollar-words': r'\b\d+(?:\.\d+)? dollars\b',
        'space-before-punctuation': r' +[?!;:,]',
        'no-space-before-question': r'\S\?',
        'decimal': r'\b\d+\.\d+\b', 'trailing-zero-decimal': r'\b\d+\.\d*0\b',
        'integer-decimal': r'\b\d+\.0+\b', 'percent-symbol': r'\d\s*%',
        'percent-words': r'\d\s+(?:percent|per cent)\b',
    }
    patterns = {k: re.compile(v) for k,v in patterns.items()}
    count = dict.fromkeys(patterns, 0); changed = dict.fromkeys(RULES, 0)
    examples = {k: [] for k in patterns}; n = 0; receipts = []; ids_hash = hashlib.sha256()
    rng = np.random.default_rng(2026101901)
    for rec in manifest['shards']:
        folder = cache/rec['folder']
        # Verify actual source arrays against the sealed training cache manifest.
        for fn in ['source.npy','source-offsets.npy']:
            pin = next(f for f in rec['files'] if f['file']==fn)
            assert sha(folder/fn)==pin['sha256']
        ids = np.load(folder/'source.npy', mmap_mode='r')
        offsets = np.load(folder/'source-offsets.npy', mmap_mode='r')
        selected = np.sort(rng.choice(len(offsets)-1, min(2000,len(offsets)-1), replace=False))
        receipts.append(dict(shard=rec['shard'],rows=len(offsets)-1,surveyed=len(selected),
                             source_sha256=sha(folder/'source.npy')))
        for i in selected:
            q = b''.join(table[int(t)] for t in ids[offsets[i]:offsets[i+1]] if t>=3).decode('utf8')
            ids_hash.update(f'{rec["shard"]}:{i}\n'.encode()); n+=1
            for k,p in patterns.items():
                m=p.search(q)
                if m:
                    count[k]+=1
                    if len(examples[k])<3: examples[k].append(q[max(0,m.start()-30):m.end()+45])
            for k in RULES:
                new=apply(q,k); assert apply(new,k)==new
                changed[k]+=new!=q
        print('survey',rec['shard'],n,flush=True)
    held = json.loads((inputs/'heldout4k.json').read_text()); coverage = {}
    for split, ids in fixed_split().items():
        coverage[split]={}
        for k in RULES:
            changed_rows=0
            for i in ids:
                q=held[i]['question']; new=apply(q,k); assert apply(new,k)==new
                changed_rows+=new!=q
            coverage[split][k]=changed_rows
    result=dict(train_n=n,train_population=manifest['rows'],sampling='2000 uniform without replacement per shard; seed2026101901',
                cache_manifest_sha256=sha(cache/'compact-manifest.json'),sample_ids_sha256=ids_hash.hexdigest(),
                tokenizer_sha256=sha(inputs/'tokenizer.json'),pre_tokenizer=json.loads((inputs/'tokenizer.json').read_text())['pre_tokenizer'],
                forms=count,examples=examples,train_rule_changes=changed,coverage=coverage,shards=receipts,
                invariants='numeric multiset + idempotence checked for every sampled train and covered heldout question')
    (WORK/'results/tta-survey.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['shards','examples','pre_tokenizer']},indent=2))

if __name__=='__main__': main()
