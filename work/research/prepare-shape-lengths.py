"""Read only verified cache offsets and L3 saved generation lengths. No targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

WORK = Path(__file__).resolve().parents[1]
CACHE = WORK/'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = json.loads((CACHE/'compact-manifest.json').read_text())
    assert manifest['sealed'] and manifest['rows'] == 9725737
    lengths = []; receipts = {}
    for shard in manifest['shards']:
        part = []
        for prefix in ('source', 'target'):
            p = CACHE/shard['folder']/(prefix+'-offsets.npy')
            rec = next(r for r in shard['files'] if r['file'] == p.name)
            assert digest(p) == rec['sha256']
            offsets = np.load(p, mmap_mode='r', allow_pickle=False)
            part.append(np.diff(offsets).astype(np.int32)+(prefix == 'target'))
            receipts[str(p.relative_to(WORK))] = rec['sha256']
        lengths.append(np.stack(part, 1))
    lengths = np.concatenate(lengths)
    order_path=CACHE/'order.npy'
    order_record=json.loads((CACHE/'order.json').read_text())
    assert digest(order_path)==order_record['file']['sha256']
    order=np.load(order_path,mmap_mode='r',allow_pickle=False)
    points=json.loads((WORK/'results/l3-final-results.json').read_text())['points']
    curve_tokens={}
    for point in points:
        if point['model'] not in ['C (shared L1)','C-main']: continue
        updates=point['updates']; cycles,tail=divmod(updates*128,len(order))
        totals=cycles*lengths.sum(0,dtype=np.int64)+lengths[order[:tail]].sum(0,dtype=np.int64)
        curve_tokens[str(updates)]={'source':int(totals[0]),'target':int(totals[1]),'pair':int(totals.sum())}
    r1=WORK/'results/kaggle-runs/nlp-hw02-kaggle-r1/1'
    weighted_record=json.loads((r1/'weighted-order/order.json').read_text())
    weighted_path=r1/'weighted-order/order.npy'
    assert digest(weighted_path)==weighted_record['file']['sha256']
    weighted=np.load(weighted_path,mmap_mode='r',allow_pickle=False)
    def prefix(order,cursor):
        cycles,tail=divmod(cursor,len(order))
        return cycles*lengths[order].sum(0,dtype=np.int64)+lengths[order[:tail]].sum(0,dtype=np.int64)
    # L2 long branch resets only its weighted cursor at the shared L1 anchor.
    anchor_updates=209144-20734592//128
    anchor_tokens=prefix(order,anchor_updates*128)
    stress_tokens={}
    for model,point in json.loads((WORK/'results/r1-results.json').read_text())['models'].items():
        if model=='L3-main': continue
        cursor=(point['updates']-anchor_updates)*128
        main_tokens=anchor_tokens+prefix(weighted,cursor)
        extra=prefix(weighted,cursor+point['cooldown_updates']*128)-prefix(weighted,cursor)
        stress_tokens[model]={'main_target':int(main_tokens[1]),'main_pair':int(main_tokens.sum()),
                             'cooldown_target':int(extra[1]),'cooldown_pair':int(extra.sum())}
    # Representative 4096-row bucket, then 8 quantile batches of 128.
    rng = np.random.default_rng(2026101507)
    ids = rng.choice(len(lengths), 4096, replace=False)
    ids = ids[np.lexsort((lengths[ids,1], lengths[ids,0], lengths[ids].sum(1)))]
    batches = [lengths[ids.reshape(32,128)[i]].tolist() for i in [1,5,9,13,17,21,25,29]]
    hard = np.argsort(lengths.sum(1))[-128:]
    hard[0] = lengths[:,0].argmax(); hard[1] = lengths[:,1].argmax()
    final = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
    p = final/'model-C-main__points__l2-62280__evaluation__test-generations.json'
    rows = json.loads(p.read_text()); assert len(rows) == 1919
    assert [r['id'] for r in rows]==list(range(1919))
    assert digest(CACHE/'tokenizer.json')==manifest['tokenizer']['sha256']
    tok = Tokenizer.from_file(str(CACHE/'tokenizer.json'))
    test = [[len(tok.encode(r['question'], add_special_tokens=False).ids)+2,
             len(r['generated_token_ids'])] for r in rows]
    test = sorted(test, key=lambda x:x[0])
    # Resample joint source/generated lengths by source rank, preserving long tails.
    take = np.linspace(0,1918,1024).round().astype(int)
    generation = [test[i] for i in take]
    result = dict(format='nlp-hw02-shape-lengths-v1', seed=2026101507,
        cache_manifest_sha256=digest(CACHE/'compact-manifest.json'), cache_rows=len(lengths),
        source_tokens_per_epoch=int(lengths[:,0].sum()),
        target_tokens_with_EOS_per_epoch=int(lengths[:,1].sum()),
        mean_pair_tokens=float(lengths.sum(1).mean()),
        mean_target_tokens=float(lengths[:,1].mean()),
        C_main_curve_tokens=curve_tokens, ordinary_order_sha256=digest(order_path),
        stress_curve_tokens=stress_tokens,weighted_order_sha256=digest(weighted_path),
        token_accounting='presented tokens, not q10%-retained loss tokens; main excludes disposable cooldowns',
        train_batches=batches, probe_batch=lengths[hard].tolist(),
        test1919_source_lengths=[x[0] for x in test], generation1024=generation,
        provenance=dict(offset_sha256=receipts, test_generation_sha256=digest(p),
                        tokenizer_sha256=digest(CACHE/'tokenizer.json')),
        semantics='Random token IDs with real joint lengths. Target length includes EOS; source includes BOS/EOS. No training text embedded.')
    out = WORK/'kaggle/shape-lengths.json'
    out.write_text(json.dumps(result, ensure_ascii=False, separators=(',',':'))+'\n')
    print(json.dumps({k:result[k] for k in ['cache_rows','source_tokens_per_epoch','target_tokens_with_EOS_per_epoch','mean_pair_tokens']}))


if __name__ == '__main__': main()
