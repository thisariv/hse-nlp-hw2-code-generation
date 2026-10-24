"""D1 setup and independent workers under the reviewed L3/P1 watchdog."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from l1_session import Watchdog
from l2_session import disk_guard
from session_runner import stop_workers, tail
from kaggle_io import write, sha


def watchdog_probe(cfg):
    """Torch-free child forces the actual watchdog exit on the same output mount."""
    target = Path(cfg['working_root'])/'watchdog-probe'; target.mkdir(parents=True, exist_ok=True)
    script = Path(cfg['temp_root'])/'d1-watchdog-probe.py'
    script.write_text('''import sys,time,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from d1_session import Watchdog
from kaggle_io import write
root=Path(sys.argv[2]);start=time.monotonic()
write(root/'durable-batch.json',{'rows':17,'artifact_durable_before_forced_exit':True})
cfg=dict(working_root=str(root),session_kind='d1',session_start_monotonic=start,
    deadline_monotonic=start+.2,hard_deadline_monotonic=start+.6,budget_seconds=.2,terminate_grace_seconds=.1)
Watchdog(cfg).start()
while True:time.sleep(.05)
''')
    p = subprocess.run([sys.executable, str(script), str(Path(__file__).parent), str(target)],
                       capture_output=True, timeout=5, start_new_session=True)
    meta = json.loads((target/'session-summary.json').read_text())
    assert p.returncode == 0 and meta['watchdog_forced_exit']
    assert json.loads((target/'durable-batch.json').read_text())['rows'] == 17
    write(target/'probe.json', dict(passed=True, actual_forced_exit=True, child_exit_code=p.returncode,
                                  durable_artifact_preserved=True, same_working_mount=True))


def setup(cfg):
    import numpy as np
    from kaggle_io import materialize_dataset
    from kaggle_runtime import prepare_starter
    from l1_cache import CompactCache
    from l3_resume import resolve_cache_mount
    from d1_data import ROWS, CACHE_SHA, MODELS, WEIGHTS, publish, length_bins
    scratch, inp = Path(cfg['temp_root']), Path(cfg['input_root'])
    work = Path(cfg['working_root']); data = work/'d1'; data.mkdir(exist_ok=True)
    if cfg['gpu_smoke']: watchdog_probe(cfg)
    starter = prepare_starter(resolve_cache_mount(inp, 'thisarivson/nlp-hw02-26b-inputs', 'datasets'),
                              scratch/'starter', 'training-inputs', read_only=True)
    try:
        cache_mount = resolve_cache_mount(inp, cfg['cache_source'], 'notebooks')
        source = cfg['cache_source']
    except ValueError:
        cache_mount = resolve_cache_mount(inp, cfg['cache_fallback_source'], 'datasets')
        source = cfg['cache_fallback_source']
    cache_root = materialize_dataset(cache_mount, scratch/'compact', 'compact-manifest.json')
    assert sha(cache_root/'compact-manifest.json') == CACHE_SHA
    cache = CompactCache(cache_root, verify=True); assert len(cache) == ROWS
    summary = json.loads((cache_root/'cache-summary.json').read_text())
    assert summary['rows'] == ROWS and summary['roundtrip_rows'] >= 50000 and summary['answer_mismatches'] == 0
    assert sha(cache_root/'tokenizer.json') == cache.manifest['tokenizer']['sha256']
    assert sha(starter/'tokenizer.json') == cache.manifest['original_tokenizer_sha256']
    live = resolve_cache_mount(inp, cfg['weights_source'], 'datasets')
    models = {}
    for name in MODELS:
        hits = list(live.rglob(f'model-{name}__weights__l2-62280-fp16.pt'))
        assert len(hits) == 1 and sha(hits[0]) == WEIGHTS[name]
        models[name] = str(hits[0])
    if not cfg['gpu_smoke']:
        seed_root = resolve_cache_mount(inp, cfg['live_dataset'], 'datasets')
        seed = [p for p in seed_root.rglob('*') if p.name in ('seed-evidence.npz', 'd1__seed-evidence.npz')]
        assert len(seed) == 1 and sha(seed[0]) == cfg['seed_sha256']
        provenance = [p for p in seed_root.rglob('*') if p.name in ('seed.receipt.json', 'd1__seed-provenance.json')]
        assert len(provenance) == 1
        provenance_meta = json.loads(provenance[0].read_text())
        assert provenance_meta['cache_manifest_sha256'] == CACHE_SHA
        assert provenance_meta['files'][0]['sha256'] == cfg['seed_sha256']
        shutil.copyfile(provenance[0], data/'seed-provenance.json')
        shutil.copyfile(seed[0], data/'seed-evidence.npz')
    else:
        from d1_data import save_npz
        save_npz(data/'seed-evidence.npz', row_id=np.empty(0, np.uint32), reference=np.empty(0),
                 target_length=cache.target_lengths, source_length=cache.source_lengths,
                 group=np.empty(0, np.uint32), group_size=np.empty(0, np.uint16), conflicting=np.empty(0, bool),
                 minority=np.empty(0, bool), majority=np.empty(0), protected=np.asarray([8545870], np.uint32))
    publish(data/'seed.receipt.json', [data/'seed-evidence.npz'], phase='seed', cache_manifest_sha256=CACHE_SHA)
    ids = np.arange(ROWS, dtype=np.uint32)
    if cfg['gpu_smoke']:
        # Uniform sample estimates full-train CE mix; top4% still ranked within length bins.
        rng = np.random.default_rng(2026101404)
        ids = np.sort(rng.choice(ROWS, cfg['smoke_rows'], replace=False)).astype(np.uint32)
    np.save(data/'scored-row-ids.npy', ids, allow_pickle=False)
    publish(data/'scored-row-ids.receipt.json', [data/'scored-row-ids.npy'], phase='scope',
            rows=len(ids), all_train=not cfg['gpu_smoke'], cache_manifest_sha256=CACHE_SHA)
    # v6 adds the three representation tokens itself; pass the ORIGINAL 8189 BPE.
    cfg.update(compact_cache=str(cache_root), tokenizer=str(starter/'tokenizer.json'),
               starter_dataset=str(starter), models=models, cache_manifest_sha256=CACHE_SHA)
    write(work/'input-preflight.json', dict(passed=True, cache_source=source, rows=ROWS,
          all_shard_SHA_verified=True, tokenizer_SHA_verified=True, models=WEIGHTS,
          seed_sha256=cfg['seed_sha256'], scoring_rows=len(ids), training=False))
    assert time.monotonic()-cfg['session_start_monotonic'] < cfg['setup_cap_seconds']


def launch(config, worker, watchdog, live=None, cpu_test=False, heartbeat=30.):
    working = Path(config['working_root']); path = working/'session-config.json'
    write(path, config); handles = []; processes = []; reason = 'workers completed'; failed = set()
    try:
        for gpu in (0, 1):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu_test else str(gpu), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                       KAGGLE_CONFIG_DIR=str(Path(config['temp_root'])/'worker-empty-config'))
            for secret in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN',
                           'KAGGLE_USER_SECRETS_TOKEN', 'KAGGLE_IAP_TOKEN'): env.pop(secret, None)
            h = (working/f'gpu{gpu}-process.log').open('w'); handles.append(h)
            p = subprocess.Popen([sys.executable, str(worker), '--gpu', str(gpu), '--config', str(path)],
                                 env=env, stdout=h, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(p); watchdog.register(p)
        while any(p.poll() is None for p in processes):
            if time.monotonic() >= config['deadline_monotonic']: reason = 'session deadline'; break
            if live is not None and (live.uninsured or time.monotonic()-live.last_success >= 7200.):
                reason = 'live insurance stale >=2h'
                write(working/'stop-request.json', dict(reason=reason))
            for gpu, p in enumerate(processes):
                if p.poll() not in (None, 0) and gpu not in failed:
                    failed.add(gpu)
                    write(working/f'gpu{gpu}-failure.json', dict(exit_code=p.poll(), healthy_worker_stopped=False))
                print('D1 GPU', gpu, 'status', p.poll(), ' | '.join(tail(working/f'gpu{gpu}-process.log'))[:400], flush=True)
            disk_guard(working, limit=2*1024**3)
            time.sleep(min(heartbeat, max(.01, config['deadline_monotonic']-time.monotonic())))
    except Exception as error:
        reason = 'supervisor error: '+type(error).__name__
    finally:
        survivors = stop_workers(processes, watchdog)
        for h in handles: h.close()
        states = {}
        for name in ('C-main', 'C-long2x'):
            summary = working/'d1'/name/'summary.json'
            try: states[name] = json.loads(summary.read_text()).get('status') if summary.exists() else 'missing'
            except (ValueError, OSError): states[name] = 'unreadable'
        report = dict(status='complete' if len(processes) == 2 and all(p.returncode == 0 for p in processes)
                      and all(s == 'complete' for s in states.values()) else 'partial',
                      worker_status=states, reason=reason, exit_codes=[p.poll() for p in processes],
                      failed_workers=sorted(failed), unreaped_worker_pids=survivors,
                      wall_seconds=time.monotonic()-config['session_start_monotonic'],
                      hard_shutdown_seconds=config['hard_shutdown_seconds'], GPU=not cpu_test, training=False)
        write(working/'session-summary.json', report)
        if live is not None:
            report['final_live_uploaded'] = live.final_sync('normal-exit', timeout=max(.1,
                min(540., watchdog.hard_deadline-time.monotonic()-60.)))
        write(working/'session-summary.json', report)
        if not survivors: watchdog.finish()
    return report
