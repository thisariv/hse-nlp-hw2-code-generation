"""One split per visible T4. No test input, training, or accuracy calculation."""
import gc
import gzip
import hashlib
import json
import os
import signal
import sys
import time
import zipfile
from pathlib import Path

if __name__ == '__main__' and sys.platform.startswith('linux'):
    import ctypes
    parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
    if parent == 1 or os.getppid() != parent:
        os.killpg(os.getpgrp(), signal.SIGKILL)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def write_gz(path, rows):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with gzip.open(temp, 'wt', encoding='utf8', compresslevel=5) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
    temp.replace(path)


def prepare(root, target):
    # Deliberately never locate, open, extract or hash test.json.
    from final_session import INPUT_SHA
    root, target = Path(root), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ('tokenizer.json', 'heldout4k.json', 'heldout4k-reference-execution.json'):
        direct = list(root.rglob(name))
        if len(direct) == 1:
            path = direct[0]
        else:
            assert not direct, 'Ambiguous E1 input '+name
            archives = list(root.rglob('26b-training-inputs-d0-initial-and-evaluation.zip'))
            assert len(archives) == 1, 'Missing E1 inputs'
            with zipfile.ZipFile(archives[0]) as z:
                members = [m for m in z.namelist() if Path(m).name == name]
                assert len(members) == 1
                path = target/name
                path.write_bytes(z.read(members[0]))
        assert sha(path) == INPUT_SHA[name], 'Input SHA '+name
        out[name] = str(path)
    return out


def vocabulary(payloads, tok, table):
    """Provenance + actual vocabulary/special-token mapping, before decoding."""
    from final_representation import MODES
    assert len(table) == tok.get_vocab_size()+3 == 8195
    assert [tok.token_to_id(t)+3 for t in MODES.values()] == [8192, 8193, 8194]
    s3 = []
    for p in payloads:
        assert p['format'] == 'nlp-hw02-model-only-v1' and p['optimizer_included'] is False
        assert p['model_spec']['vocab'] == len(table)
        assert p['model']['tokens.weight'].shape[0] == p['model']['head.weight'].shape[0] == len(table)
        identity = p['identity']; recipe = identity['recipe']
        assert identity['cache_manifest_sha256'] == '46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be'
        assert recipe.get('representation', 30) == 30
        s3.append(recipe['implementation_sha256']['s3_representation.py'])
    assert len(set(s3)) == 1
    assert payloads[0]['model_spec'] == payloads[1]['model_spec'], 'Soup architecture differs'
    return dict(vocab=len(table), PAD=0, BOS=1, EOS=2, offset=3, representation=30,
                representation_source_sha256=s3[0], shared_cache_manifest=True,
                token_table_sha256=hashlib.sha256(b'\0'.join(table)).hexdigest())


def main(config, split, phase):
    import torch
    from torch import nn
    import final_core as C
    import final_fast as Fast
    import final_speed as Speed
    from final_inference import DECODE_OPTIONS, pack
    from final_representation import tokenizer
    from final_mounts import find_input, resolve_cache_mount
    from final_session import weight_path
    from final_executor_v6 import Executor
    from final_split import fixed_split
    from e1_decode import decode, teacher_score, soup_state
    start = time.monotonic()
    deadline = float(os.environ.get('E1_DEADLINE', start+4000))-20
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    fixture = config.get('cpu_fixture', False)
    if not fixture:
        assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0)
        assert str(torch.__version__) == '2.11.0+cu128' and torch.version.cuda == '12.8'
    device = torch.device('cpu' if fixture else 'cuda:0')
    out = Path(config['output'])/phase/split; out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', split=split, phase=phase, training=False, test_read=False,
                  timings=[], OOM=[], models=[], seed=config['seed'])
    write(out/'summary.json', report)
    inputs = config['inputs'] if fixture else prepare(find_input('nlp-hw02-26b-inputs'), out/'inputs')
    rows = json.loads(Path(inputs['heldout4k.json']).read_text())
    refs = json.loads(Path(inputs['heldout4k-reference-execution.json']).read_text())['rows']
    assert len(rows) == len(refs) == (config.get('fixture_rows', 4096))
    tok, table = tokenizer(inputs['tokenizer.json'], 30)
    ids = fixed_split(len(rows))[split]
    if phase == 'smoke':
        # Include long sources, not just the first 64 IDs.
        ordered = sorted(ids, key=lambda i: (len(C.encode(tok, rows[i]['question'])), i))
        count = min(config.get('smoke_rows', 64), len(ordered))
        ids = sorted(ordered[j*(len(ordered)-1)//max(count-1, 1)] for j in range(count))
    report.update(rows=len(ids), input_sha256={k: sha(v) for k, v in inputs.items()},
                  gpu_name='CPU fixture' if fixture else torch.cuda.get_device_name(0))
    encoded = {i: [C.BOS]+C.encode(tok, rows[i]['question'])+[C.EOS] for i in ids}
    order = sorted(ids, key=lambda i: (len(encoded[i]), i))
    records = {i: dict(id=i, split=split, question_sha256=hashlib.sha256(rows[i]['question'].encode()).hexdigest(),
                      pools={}, candidates=[], reference=refs[i]) for i in ids}
    unique = {i: {} for i in ids}
    payloads, models = [], []
    for spec in config['models']:
        if fixture:
            path = spec['path']
        else:
            mount = resolve_cache_mount(Path('/kaggle/input'), spec['dataset'], spec['source_kind'])
            path = weight_path(mount, spec)
        assert sha(path) == spec['sha256'], 'Model SHA '+spec['name']
        p = torch.load(path, map_location='cpu', weights_only=True)
        payloads.append(p)
        model = C.SimpleTransformer(**p['model_spec'], dropout=0.).eval()
        model.load_state_dict(p['model'], strict=True)
        if not fixture:
            model.half()
        models.append(model.to(device))
        report['models'].append(dict(name=spec['name'], sha256=spec['sha256'], spec=p['model_spec']))
    if not fixture:
        report['vocabulary'] = vocabulary(payloads, tok, table)
    assert payloads[0]['model_spec'] == payloads[1]['model_spec']
    report['setup_seconds'] = time.monotonic()-start
    write(out/'summary.json', report)
    executor = Executor(config.get('workers', 2))
    cap = config.get('cap', 512)

    def source(chosen):
        return nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in chosen],
                                         batch_first=True, padding_value=C.PAD).to(device)

    def add(i, key, group, sequences):
        occurrences = []
        for candidate, tokens in zip(group, sequences.tolist()):
            active = C.active_tokens(tokens)
            assert active and all(x >= C.TOKEN_OFFSET or x == C.EOS for x in active)
            assert C.EOS not in active[:-1]
            if C.EOS in active:
                at = tokens.index(C.EOS)
                assert all(x == C.PAD for x in tokens[at+1:]), 'Non-PAD after EOS'
            represented = candidate['represented_code']
            if represented not in unique[i]:
                u = len(records[i]['candidates']); unique[i][represented] = u
                candidate.update(ended=bool(active[-1] == C.EOS), ll={})
                records[i]['candidates'].append(candidate)
            u = unique[i][represented]
            occurrences.append(dict(u=u, mean_logprob=candidate['mean_logprob'], ended=active[-1] == C.EOS))
        records[i]['pools'].setdefault(key, []).extend(occurrences)

    def generate(label, members, k, temperature, lanes):
        batch = max(1, min(config.get('max_batch', 512), lanes//k))
        seed = config['seed']+int.from_bytes(hashlib.sha256((label+':K'+str(k)).encode()).digest()[:4], 'little') % 1000000
        torch.manual_seed(seed)
        if not fixture:
            torch.cuda.manual_seed_all(seed)
        lo = 0
        while lo < len(order):
            C.check_deadline(deadline)
            chosen = order[lo:lo+batch]
            rng = torch.get_rng_state(); gpu_rng = None if fixture else torch.cuda.get_rng_state()
            src = contexts = seq = lp = lengths = groups = None
            begin = time.monotonic()
            failed = False
            try:
                src = source(chosen)
                contexts = [Fast.Context(m, src, cap) for m in members]
                stats = {}
                if len(members) == 1:
                    seq, means, _ = Speed.decode(contexts[0], k, temperature, deadline,
                        compute_logprob=True, options=DECODE_OPTIONS, stats=stats)
                else:
                    seq, lp, lengths = decode(contexts, k, temperature, deadline, stats=stats)
                    means = lp/lengths.clamp_min(1)
                assert seq.ndim == 3 and tuple(seq.shape[:2]) == (len(chosen), k)
                assert tuple(means.shape) == (len(chosen), k) and 0 < seq.shape[-1] <= cap
                groups = pack(seq, means, table)
                for i, group, sequences in zip(chosen, groups, seq):
                    assert all(c['mean_logprob'] is not None for c in group), 'Nonfinite generation LP'
                    add(i, label, group, sequences)
                executor.submit([c for g in groups for c in g])
                seconds = time.monotonic()-begin
                report['timings'].append(dict(kind='generation', label=label, rows=len(chosen),
                    sequences=len(chosen)*k, seconds=seconds, samples_per_second=len(chosen)*k/seconds,
                    ms_per_row=1000*seconds/len(chosen), K=k, temperature=temperature,
                    source_tokens=src.shape[1], steps=seq.shape[-1], decoder=stats))
                lo += len(chosen)
                if lo == len(order) or lo % 256 < len(chosen):
                    write(out/'summary.json', report)
                    print('E1',phase,split,label,lo,'seconds',round(time.monotonic()-start,1),flush=True)
            except torch.cuda.OutOfMemoryError:
                failed = True
                torch.set_rng_state(rng); torch.cuda.set_rng_state(gpu_rng)
                report['OOM'].append(dict(label=label, offset=lo, batch=batch))
                batch //= 2
                if batch < 1:
                    raise
            finally:
                del src, contexts, seq, lp, lengths, groups
                if failed:
                    gc.collect(); torch.cuda.empty_cache()

    try:
        for name, model in zip(('C', 'P', 'W'), models):
            generate(name+'/g', [model], 1, None, config.get('single_lanes', 512))
            # First sample separately: measures the actual v10 stage-one shape.
            generate(name+'/0.7', [model], 1, .7, config.get('single_lanes', 512))
            generate(name+'/0.7', [model], config.get('K07', 16)-1, .7, config.get('single_lanes', 512))
            generate(name+'/1.0', [model], config.get('K10', 8), 1., config.get('single_lanes', 512))
        generate('E/g', models, 1, None, config.get('ensemble_lanes', 256))
        generate('E/0.7', models, 1, .7, config.get('ensemble_lanes', 256))
        generate('E/0.7', models, config.get('KE', 8)-1, .7, config.get('ensemble_lanes', 256))
        for alpha in config.get('alphas', [.3, .5, .7]):
            C.check_deadline(deadline)
            begin = time.monotonic()
            state = soup_state(payloads[0]['model'], payloads[1]['model'], alpha)
            soup = C.SimpleTransformer(**payloads[0]['model_spec'], dropout=0.).eval()
            soup.load_state_dict(state, strict=True)
            if not fixture:
                soup.half()
            soup = soup.to(device)
            report.setdefault('soup_setup_seconds', {})[str(alpha)] = time.monotonic()-begin
            generate('S'+str(alpha)+'/g', [soup], 1, None, config.get('single_lanes', 512))
            del soup, state
            gc.collect()
        # Canonical represented-code tokenization, unique within each question.
        jobs = []
        for i in ids:
            for u, c in enumerate(records[i]['candidates']):
                target = C.encode(tok, c['represented_code'])+[C.EOS]
                assert len(target) < models[0].positions.num_embeddings
                c['ll_length'] = len(target)
                jobs.append((i, u, target))
        jobs.sort(key=lambda job: (len(job[2]), len(encoded[job[0]]), job[0], job[1]))
        for name, model in zip(('C', 'P', 'W'), models):
            lo = 0; batch = config.get('ll_batch', 32)
            while lo < len(jobs):
                C.check_deadline(deadline)
                selected = jobs[lo:lo+batch]
                while len(selected)*max(len(x[2]) for x in selected) > config.get('ll_token_budget', 4096) and len(selected) > 1:
                    selected = selected[:len(selected)//2]
                src = sums = means = lengths = None
                begin = time.monotonic(); failed = False
                try:
                    src = source([j[0] for j in selected])
                    sums, means, lengths = teacher_score(model, src, [j[2] for j in selected])
                    for (i, u, target), total, mean, length in zip(selected, sums.tolist(), means.tolist(), lengths.tolist()):
                        assert length == len(target)
                        records[i]['candidates'][u]['ll'][name] = dict(sum=total, mean=mean)
                    report['timings'].append(dict(kind='LL', label=name, rows=len(selected), tokens=int(lengths.sum()),
                                                   seconds=time.monotonic()-begin))
                    lo += len(selected)
                    if lo == len(jobs) or lo % 2048 < len(selected):
                        write(out/'summary.json', report)
                        print('E1',phase,split,'LL',name,lo,'/',len(jobs),flush=True)
                except torch.cuda.OutOfMemoryError:
                    failed = True
                    report['OOM'].append(dict(label='LL/'+name, offset=lo, batch=batch))
                    batch //= 2
                    if batch < 1:
                        raise
                finally:
                    del src, sums, means, lengths
                    if failed:
                        gc.collect(); torch.cuda.empty_cache()
        assert executor.drain(deadline), 'E1 execution incomplete: '+str(executor.summary())
        for i in ids:
            executor.attach(records[i]['candidates'])
            assert all(c['execution']['status'] != 'budget_not_evaluated' and set(c['ll']) == {'C','P','W'}
                       for c in records[i]['candidates']), 'Incomplete E1 candidate'
        manifest = []
        for lo in range(0, len(ids), 64):
            path = out/f'pool-{lo:05d}.jsonl.gz'
            write_gz(path, [records[i] for i in ids[lo:lo+64]])
            # Read back the actual bytes before claiming smoke success.
            with gzip.open(path, 'rt', encoding='utf8') as f:
                restored = [json.loads(line) for line in f]
            assert restored == [records[i] for i in ids[lo:lo+64]], 'E1 gzip roundtrip'
            manifest.append(dict(file=path.name, sha256=sha(path), rows=len(restored)))
        report.update(status='complete', journal_manifest=manifest, unique_candidates=len(jobs),
                      candidates_per_row=sum(len(p) for p in records[ids[0]]['pools'].values()),
                      ll_tokenization='canonical represented_code + EOS; sum and mean; PAD/BOS masked')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['executor'] = executor.summary(); executor.close()
        report['seconds'] = time.monotonic()-start
        if not fixture:
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out/'summary.json', report)
    print('E1 COMPLETE',phase,split,round(report['seconds'],2),flush=True)


if __name__ == '__main__':
    main(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], sys.argv[3])
