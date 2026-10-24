"""Fill S3's missing B scores on CPU/FP32 for every requested k<=32 + greedy candidate."""
import os
import sys
import time
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'kaggle/src'))


def main():
    import torch
    import exp26_core as C
    import s3_score as B
    from exp26_common import read_json, write_json, sha_file, json_hash
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    out = WORK / 'research/outputs/s3-cpu-b-scores'; out.mkdir(parents=True, exist_ok=True)
    source = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1'
    b = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s2/2/gpu1-exp28'
    receipt = read_json(b / 'points/t04/checkpoint.json'); weights = b / receipt['file']
    digest = sha_file(weights)
    assert digest == receipt['sha256'] == 'cbbe292cf65e37ec4876b8206b19b06b6fbe7541fd7be1115799c3ce085f720a'
    tok, _ = C.init_tokenizer(source / 'inputs/tokenizer.json')
    sources, pending, identities = {}, [], []
    for split in ['heldout4k', 'old1024']:
        original = read_json(source / 'inputs' / (split + '.json')); seen = set()
        for path in sorted((source / 'gpu0-exp27' / split / 'candidates').glob('batch-*.json')):
            if '.sha.' in path.name: continue
            assert sha_file(path) == read_json(path.with_suffix('.sha.json'))['sha256']
            identities.append([split, path.name, sha_file(path)])
            for row in read_json(path):
                i = row['id']; assert i not in seen and row['question'] == original[i]['question']; seen.add(i)
                src = [C.BOS] + C.encode(tok, row['question']) + [C.EOS]; sources[(split, i)] = src
                unique = {}
                for c in B.flatten(row, tok)[:33]:
                    ids = tuple(c.pop('ids')); unique.setdefault(ids, []).append(dict(c, id=i, split=split))
                for ids, records in unique.items(): pending.append((split, i, ids, records))
        assert seen == set(range(len(original)))
    # Global length bucketing removes padding to a cap-stopped outlier in every 16-question source batch.
    pending.sort(key=lambda item: (len(item[2]) + len(sources[(item[0], item[1])]), len(item[2]), item[0], item[1]))
    identity = dict(weights_sha256=digest, candidate_receipts_sha256=json_hash(identities),
        tokenizer_sha256=sha_file(source / 'inputs/tokenizer.json'), candidate_indices=list(range(33)),
        precision='CPU FP32', EOS_policy='retokenize decoded code and append EOS even if cap stopped',
        unique_programs=len(pending), candidates=sum(len(x[3]) for x in pending), chunk_rows=512, batch_rows=64)
    assert identity['candidates'] == 5120 * 33
    if (out / 'identity.json').exists(): assert read_json(out / 'identity.json') == identity
    else: write_json(out / 'identity.json', identity)
    model_payload = torch.load(weights, map_location='cpu', weights_only=True)
    model = C.SimpleTransformer(**model_payload['model_spec'], dropout=0.).eval()
    model.load_state_dict(model_payload['model']); del model_payload
    summary = dict(status='partial', identity=identity, completed_unique=0, scored_candidates=0,
        GPU_runs=0, training_updates=0, full_all65_GPU_scores=0, scope='all9/17/33 candidate pools; unused T0.5/T1.0 not rescored')
    write_json(out / 'summary.json', summary)
    started = time.perf_counter()
    for lo in range(0, len(pending), 512):
        path = out / f'chunk-{lo//512:06d}.json'; part = pending[lo:lo+512]
        if path.exists():
            assert sha_file(path) == read_json(path.with_suffix('.sha.json'))['sha256']
            scored = read_json(path)
            assert len(scored) == sum(len(x[3]) for x in part)
        else:
            scored = []
            for start in range(0, len(part), 64):
                batch = part[start:start+64]
                scores = B.score_batch(model, [sources[(x[0], x[1])] for x in batch],
                    [list(x[2]) for x in batch], torch.device('cpu'))
                for item, (total, mean, n) in zip(batch, scores):
                    for c in item[3]: scored.append(dict(c, sum_logprob=total, mean_logprob=mean, length_with_eos=n))
            write_json(path, scored); write_json(path.with_suffix('.sha.json'), dict(sha256=sha_file(path)))
        summary['completed_unique'] += len(part); summary['scored_candidates'] += len(scored)
        summary['seconds'] = time.perf_counter() - started
        write_json(out / 'summary.json', summary)
        print('B_CPU', summary['completed_unique'], '/', len(pending), 'candidates', summary['scored_candidates'],
            'seconds', round(summary['seconds'], 1), flush=True)
    assert summary['scored_candidates'] == 168960
    summary['status'] = 'complete'; write_json(out / 'summary.json', summary)
    print('B_CPU COMPLETE', round(summary['seconds'], 1), flush=True)


if __name__ == '__main__': main()
