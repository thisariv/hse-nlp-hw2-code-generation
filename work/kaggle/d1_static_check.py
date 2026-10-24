"""D1 notebook/source/privacy/budget gate. No GPU, uploads or credential reads."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import nbformat
from build_l3 import literals

ROOT = Path(__file__).resolve().parent; WORK = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from kaggle_io import sha, write
from d1_session import Watchdog


def main():
    artifacts = {}
    v6 = json.loads((ROOT/'notebooks/nlp-hw02-adaptive-inference-l3-v6.ipynb').read_text())
    frozen_final = literals(next(c['source'] for c in v6['cells'] if c['cell_type'] == 'code'))['SOURCES']
    p1 = json.loads((ROOT/'notebooks/nlp-hw02-kaggle-p1.ipynb').read_text())
    frozen_p1 = literals(next(c['source'] for c in p1['cells'] if c['cell_type'] == 'code'))['SOURCES']
    for kind, slug in [('d1', 'nlp-hw02-kaggle-d1'), ('d1_gpu_smoke', 'nlp-hw02-d1-gpu-smoke')]:
        path = ROOT/'notebooks'/f'{slug}.ipynb'; nb = json.loads(path.read_text())
        nbformat.validate(nbformat.from_dict(nb))
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        for cell in code:
            assert not cell['outputs'] and cell['execution_count'] is None
            compile(cell['source'], slug, 'exec')
        v = literals(code[0]['source']); cfg, sources = v['CONFIG'], v['SOURCES']
        assert cfg['owner'] == ('thisarivson' if cfg['gpu_smoke'] else 'friend1') and cfg['session_kind'] == 'd1'
        assert cfg['live_dataset'] == 'friend1/nlp-hw02-d1-live'
        assert cfg['port_sha256'] == hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        assert {'final_execute_v6.py', 'final_execute.py', 'final_numeric.py', 'd1_worker.py'} <= sources.keys()
        for name, source in sources.items():
            compile(source, name, 'exec')
            if name.startswith('final_'): assert source == frozen_final[name]
            elif name.startswith('d1_'):
                local = ROOT/name if name == 'd1_worker.py' else ROOT/'src'/name
                assert local.read_text() == source
            else: assert source == frozen_p1[name]
        first = '\n'.join(line for line in code[0]['source'].splitlines() if not line.startswith(('SOURCES=', 'CONFIG=')))
        assert first.index('WATCHDOG=') < first.index('import torch') < first.index('GATE=gate')
        assert not cfg['require_secrets'] and not cfg['enable_live_sync']
        assert "require_secrets=CONFIG['require_secrets']" in first
        assert 'pip' not in first and 'setup(CONFIG)' in code[-1]['source']
        assert 'verify=True' in sources['d1_session.py']
        assert "starter/'tokenizer.json'" in sources['d1_session.py']
        assert 'original_tokenizer_sha256' in sources['d1_session.py']
        assert 'optimizer' not in sources['d1_loss.py']
        assert 'backward' not in sources['d1_loss.py']
        assert 0 < cfg['fraction'] <= .04 and cfg['timing_safety'] >= 1.5
        assert cfg['setup_cap_seconds'] < cfg['loss_stop_seconds'] < cfg['generation_stop_seconds']
        assert cfg['generation_stop_seconds'] < cfg['execution_stop_seconds'] < cfg['worker_deadline_seconds']
        assert cfg['worker_deadline_seconds'] < cfg['hard_shutdown_seconds'] <= (660 if cfg['gpu_smoke'] else 7800)
        assert cfg['session_ceiling_seconds'] <= (720 if cfg['gpu_smoke'] else 8400)
        watch = Watchdog(dict(cfg, session_start_monotonic=0, deadline_monotonic=cfg['worker_deadline_seconds'],
                              hard_deadline_monotonic=cfg['hard_shutdown_seconds']))
        assert watch.hard_deadline == cfg['hard_shutdown_seconds']
        bundle = ROOT/'bundles'/kind
        assert sha(bundle/path.name) == sha(path)
        meta = json.loads((bundle/'kernel-metadata.json').read_text())
        assert meta['id'] == cfg['owner']+'/'+slug and meta['is_private'] and meta['enable_internet']
        assert meta['enable_gpu'] and meta['machine_shape'] == 'NvidiaTeslaT4'
        assert meta['kernel_sources'] == ['thisarivson/nlp-hw02-compact-cache-verified/2']
        assert cfg['weights_source'] in meta['dataset_sources']
        if not cfg['gpu_smoke']: assert cfg['cache_fallback_source'] in meta['dataset_sources']
        artifacts[kind] = dict(notebook_sha256=sha(path), modules=len(sources), hard_wall_seconds=cfg['hard_shutdown_seconds'])
    live = json.loads((WORK/'checks/kaggle-d1-live-created.json').read_text())
    real = json.loads((WORK/'checks/kaggle-d1-real-cpu.json').read_text())
    tests = json.loads((WORK/'checks/kaggle-d1-regressions.json').read_text())
    inherited = json.loads((WORK/'checks/kaggle-d1-inherited-regressions.json').read_text())
    assert live['private'] and live['account'] == 'friend1' and live['passed']
    assert real['passed'] and tests['passed']
    assert inherited['passed'] and inherited['tests'] == 89
    write(WORK/'checks/kaggle-d1-static.json', dict(passed=True, artifacts=artifacts,
          private_live=live['ref'], actual_L3_CPU_parity=True, regressions=tests, inherited_regressions=inherited,
          frozen_v6_sources=True, GPU=False, push=False, ready_for_independent_GPU_review=True))
    print('D1 STATIC PASSED; independent GPU smoke required; GPU/push=0')


if __name__ == '__main__': main()
