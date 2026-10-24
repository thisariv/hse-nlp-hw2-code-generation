"""D1 CPU regressions: CE parity, v6 execution, partial data and order integrity."""
import ast
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent; WORK = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
# Exercise the SAME frozen decoder/service embedded in D1, even while v7 is edited.
FROZEN = tempfile.TemporaryDirectory(prefix='nlp-d1-frozen-v6-')
nb = json.loads((ROOT/'notebooks/nlp-hw02-adaptive-inference-l3-v6.ipynb').read_text())
startup = next(c['source'] for c in nb['cells'] if c['cell_type'] == 'code')
sources = ast.literal_eval(next(n.value for n in ast.parse(startup).body if isinstance(n, ast.Assign)
    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
for name, source in sources.items(): (Path(FROZEN.name)/name).write_text(source)
sys.path.insert(0, FROZEN.name)
import exp26_core as C
from exp26_order import collate
from d1_data import (MODELS, WEIGHTS, CACHE_SHA, save_npz, publish, ranked_candidates, choose_budget,
                     filtered_order, verify_receipt, close)
import d1_loss as L
import d1_generate as G
from d1_live import stage_snapshot, restore_snapshot
from kaggle_io import sha, write
spec = importlib.util.spec_from_file_location('d1_analyze', WORK/'research/analyze-d1-labels.py')
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
CPU = torch.device('cpu')


class Cache:
    def __init__(self, n=29):
        self.rows = [(np.asarray([1, 4+i%4, 6, 2]), np.asarray([7, 8, 9]*(1+i%5))) for i in range(n)]
        self.source_lengths = np.asarray([len(s) for s,t in self.rows], np.uint16)
        self.target_lengths = np.asarray([len(t) for s,t in self.rows], np.uint16)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[int(i)]


def synthetic(root):
    root = Path(root)/'d1'; root.mkdir(parents=True)
    n = 400; lengths = np.repeat([20, 60, 90, 120], 100).astype(np.uint16)
    scores = np.tile(np.arange(100), 4).astype(np.float16)/50
    ranked, _ = ranked_candidates(scores, scores, lengths)
    ids = np.concatenate(ranked)
    save_npz(root/'seed-evidence.npz', row_id=ids[np.argsort(ids)], reference=np.full(len(ids), 10.),
        group=np.arange(len(ids), dtype=np.uint32), group_size=np.full(len(ids), 3, np.uint16),
        conflicting=np.ones(len(ids), bool), minority=np.ones(len(ids), bool), majority=np.full(len(ids), 20.),
        target_length=lengths, source_length=np.full(n, 12, np.uint16), protected=np.asarray([0], np.uint32))
    publish(root/'seed.receipt.json', [root/'seed-evidence.npz'], phase='seed', cache_manifest_sha256=CACHE_SHA)
    np.save(root/'scored-row-ids.npy', np.arange(n, dtype=np.uint32), allow_pickle=False)
    publish(root/'scored-row-ids.receipt.json', [root/'scored-row-ids.npy'], all_train=True)
    save_npz(root/'selection.npz', row_id=ids)
    publish(root/'selection.receipt.json', [root/'selection.npz'], phase='selection', cache_manifest_sha256=CACHE_SHA)
    for m in MODELS:
        loss = root/m/'loss'; loss.mkdir(parents=True)
        write(root/m/'identity.json', dict(model=m, weight_sha256=WEIGHTS[m], cache_manifest_sha256=CACHE_SHA, training=False))
        for name, array in [('ce', scores), ('nll', (scores*(lengths+1)).astype(np.float32)), ('length', lengths+1)]:
            np.save(loss/f'{name}.npy', array, allow_pickle=False)
        publish(loss/'complete.receipt.json', list(sorted(loss.glob('*.npy'))), complete=True,
                rows=n, sum_NLL=float(np.load(loss/'nll.npy').sum(dtype=np.float64)), tokens=int((lengths+1).sum()))
        result = root/m/'generation-00000000.npz'
        arrays = dict(row_id=ids, answer=np.full(len(ids), 20.), status=np.full(len(ids), 'numeric'),
                      generation_status=np.full(len(ids), 'eos'))
        if m == MODELS[0]: arrays.update(reference=np.full(len(ids), 10.), reference_status=np.full(len(ids), 'numeric'))
        save_npz(result, **arrays); publish(result.with_suffix('.receipt.json'), [result], phase='generation', rows=len(ids))
    return root, ids


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nlp-d1-test-'); self.root = Path(self.temp.name)
        torch.manual_seed(17)
        self.model = C.SimpleTransformer(d_model=8, heads=2, encoder_layers=1, decoder_layers=1,
                                        swiglu_dim=16, vocab=24, dropout=0.).eval()
    def tearDown(self): self.temp.cleanup()

    def test_two_long_CE_shards_cover_global_rows_once_and_partial_batches_survive(self):
        cache = Cache(); ids = np.arange(len(cache), dtype=np.uint32)
        for gpu in (0,1):
            L.score_all(self.model, cache, ids[gpu::2], self.root/f'loss-shards/gpu{gpu}', CPU,
                        lambda:None, batch=4, block=8, compact=True)
        merged = L.merge_shards(self.root, ids, cache)
        self.assertTrue(merged['complete']); self.assertEqual(merged['rows'], len(cache))
        nn,ll = L.row_nll(self.model, cache, ids, CPU)
        np.testing.assert_allclose(np.load(self.root/'C-long2x/loss/ce.npy'), nn/ll, rtol=.001)
        calls = []
        def deadline():
            calls.append(1)
            if len(calls)>2: raise TimeoutError('fixture')
        partial = L.score_all(self.model,cache,ids,self.root/'partial',CPU,deadline,batch=4,block=16,compact=True)
        self.assertFalse(partial['complete']); self.assertGreater(partial['rows'],0)
        self.assertFalse((self.root/'partial/complete.receipt.json').exists())
        verify_receipt(self.root/'partial/partial.receipt.json')
        saved = np.load(self.root/'partial/ce.npy')
        self.assertEqual(np.isfinite(saved).sum(),partial['rows'])
        self.assertFalse((self.root/'partial/nll.npy').exists())

    def test_D1_build_budget_optional_secrets_owner_and_input_contracts(self):
        from build_d1 import build
        from build_l3 import literals
        for smoke in (False,True):
            path = build(smoke)
            nb = json.loads(path.read_text()); first = next(c['source'] for c in nb['cells'] if c['cell_type']=='code')
            cfg = literals(first)['CONFIG']
            self.assertFalse(cfg['require_secrets']); self.assertFalse(cfg['enable_live_sync'])
            self.assertEqual(cfg['owner'],'thisarivson' if smoke else 'friend1')
            self.assertLessEqual(cfg['hard_shutdown_seconds'],660 if smoke else 7800)
            self.assertGreaterEqual(cfg['session_ceiling_seconds']-cfg['hard_shutdown_seconds'],60 if smoke else 600)
            self.assertEqual(cfg['fraction'],.04)
            self.assertIn("load_model('C-long2x')",literals(first)['SOURCES']['d1_worker.py'])
        from l2_live import gate
        fake = type('Torch',(),{'cuda':type('Cuda',(),{'device_count':staticmethod(lambda:2),
                         'get_device_name':staticmethod(lambda _: 'T4')})})
        self.assertTrue(gate(fake,secret_getter=lambda _:self.fail('Secrets accessed'),require_secrets=False)['passed'])

    def test_single_CE_analysis_keeps_original_reference_and_rejects_incomplete(self):
        source,ids = synthetic(self.root/'single')
        import shutil
        shutil.rmtree(source/'C-main/loss')
        loss = source/'C-long2x/loss'
        n = len(np.load(loss/'ce.npy'))
        publish(loss/'complete.receipt.json',[loss/'ce.npy'],complete=True,model='C-long2x',
                weight_sha256=WEIGHTS['C-long2x'],rows=n,tokens=int(np.load(loss/'length.npy').sum()))
        result = A.analyze(source,self.root/'single-analysis',allow_partial=True)
        self.assertEqual(result['score_policy'],'C-long2x only')
        self.assertEqual(result['metrics']['filters']['A']['rows'],len(ids))
        (loss/'complete.receipt.json').unlink()
        publish(loss/'partial.receipt.json',[loss/'ce.npy'],complete=False,model='C-long2x',
                weight_sha256=WEIGHTS['C-long2x'],rows=n,tokens=int(np.load(loss/'length.npy').sum()))
        with patch.object(A,'ROWS',n):
            with self.assertRaises(AssertionError): A.analyze(source,self.root/'refused-single',allow_partial=False)
        partial=A.analyze(source,self.root/'allowed-single',allow_partial=True)
        self.assertTrue(partial['partial'])

    def test_generation_timeout_retains_completed_batch_and_reference_pending_marks_partial(self):
        import final_core as FC
        model = FC.SimpleTransformer(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,
                                    swiglu_dim=16,vocab=24,dropout=0.).eval()
        with torch.no_grad(): model.head.weight.zero_();model.head.bias.zero_();model.head.bias[2]=100.
        calls=[]
        def check():
            calls.append(1)
            if len(calls)>1: raise TimeoutError('fixture')
        result=G.generation_wave(model,Cache(4),np.arange(4,dtype=np.uint32),[b'']*24,CPU,
               self.root/'partial-wave.npz',check,time.monotonic()+5,batch=2,cap=4)
        self.assertFalse(result['complete']);self.assertEqual(result['rows'],2)
        verify_receipt(self.root/'partial-wave.receipt.json')
        with np.load(self.root/'partial-wave.npz') as z: np.testing.assert_array_equal(z['row_id'],[0,1])
        with patch.object(G,'reference_for',return_value=(None,{'status':'budget_not_evaluated'},'executed')):
            pending=G.generation_wave(model,Cache(1),np.asarray([0],np.uint32),[b'']*24,CPU,
                self.root/'pending-reference.npz',lambda:None,time.monotonic()+5,
                evidence={'row_id':np.empty(0,np.uint32),'reference':np.empty(0)},cap=4)
        self.assertFalse(pending['complete'])

    def test_actual_watchdog_exits_zero_and_preserves_durable_D1_outputs(self):
        import subprocess
        worker=self.root/'watchdog-fixture.py'
        worker.write_text('''import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from d1_session import Watchdog
from kaggle_io import write
root=Path(sys.argv[2]);root.mkdir(exist_ok=True)
write(root/'durable-batch.json',{'rows':17})
start=time.monotonic()
cfg=dict(working_root=str(root),session_start_monotonic=start,deadline_monotonic=start+.2,
    hard_deadline_monotonic=start+.6,budget_seconds=.2,session_kind='d1',terminate_grace_seconds=.1)
Watchdog(cfg).start()
while True:time.sleep(.05)
''')
        out=self.root/'watchdog-output'
        result=subprocess.run([sys.executable,str(worker),str(ROOT/'src'),str(out)],capture_output=True,timeout=5)
        self.assertEqual(result.returncode,0)
        self.assertEqual(json.loads((out/'durable-batch.json').read_text())['rows'],17)
        self.assertTrue(json.loads((out/'session-summary.json').read_text())['watchdog_forced_exit'])
        from d1_session import watchdog_probe
        watchdog_probe(dict(working_root=str(self.root),temp_root=str(self.root)))
        self.assertTrue(json.loads((self.root/'watchdog-probe/probe.json').read_text())['passed'])

    def test_per_row_loss_matches_ordinary_training_CE_with_PAD_and_EOS(self):
        cache = Cache(); ids = np.arange(len(cache), dtype=np.uint32)
        actual, lengths = L.row_nll(self.model, cache, ids, CPU, ce_chunk=17)
        src, tgt, labels, counts = collate(cache, ids, CPU)
        # Ordinary training objective, without backward/optimizer changes.
        logits = self.model(src, tgt)
        expected = F.cross_entropy(logits.reshape(-1, self.model.vocab), labels.reshape(-1), ignore_index=C.PAD)
        self.assertAlmostEqual(float(actual.sum(dtype=np.float64)/lengths.sum()), float(expected.detach()), places=6)
        np.testing.assert_array_equal(lengths, cache.target_lengths+1)
        self.assertEqual(int(lengths.sum()), counts['target_tokens'])
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        single = np.concatenate([L.row_nll(self.model, cache, [i], CPU)[0] for i in ids])
        np.testing.assert_allclose(actual, single, rtol=2e-6, atol=1e-5)

    def test_score_resume_OOM_and_row_identity(self):
        cache = Cache(); ids = np.arange(len(cache), dtype=np.uint32)
        original = L.row_nll; seen = []
        def injected(model, data, ii, device):
            if not seen:
                seen.append(True); raise torch.cuda.OutOfMemoryError('fixture')
            return original(model, data, ii, device)
        with patch.object(L, 'row_nll', injected):
            result = L.score_all(self.model, cache, ids, self.root/'loss', CPU, lambda:None, batch=16, block=10)
        self.assertEqual(result['OOM_retries'], 1)
        expected, lengths = L.row_nll(self.model, cache, ids, CPU)
        np.testing.assert_allclose(np.load(self.root/'loss/nll.npy'), expected, rtol=2e-6)
        again = L.score_all(self.model, cache, ids, self.root/'loss', CPU, lambda:None, block=10)
        self.assertEqual(result['sum_NLL'], again['sum_NLL'])
        verify_receipt(self.root/'loss/complete.receipt.json')
        with (self.root/'loss/block-00000000.npz').open('ab') as stream: stream.write(b'bad')
        with self.assertRaises(AssertionError):
            L.score_all(self.model, cache, ids, self.root/'loss', CPU, lambda:None, block=10)

    def test_stratified_top8_ties_and_budget(self):
        lengths = np.repeat([20, 60, 90, 120], 100)
        ranked, _ = ranked_candidates(np.ones(400), np.ones(400), lengths)
        for b, ids in enumerate(ranked): np.testing.assert_array_equal(ids, np.arange(100*b, 100*b+8))
        pilots = [dict(bins=[dict(rows=2, seconds=1+b) for b in range(4)])]*2
        selected, budget = choose_budget(ranked, pilots, remaining=40, safety=1, tail=0)
        self.assertEqual(len(selected), 32); self.assertEqual(budget['quotas'], [8]*4)
        with self.assertRaises(TimeoutError): choose_budget(ranked, pilots, remaining=0)

    def test_filter_statuses_cap_isclose_and_duplicate_rules(self):
        n = 7; x = np.full(n, 20.); ref = np.full(n, 10.); s = np.full(n, 'numeric', dtype='U48')
        generation = np.full(n, 'eos', dtype='U16'); generation[1] = 'token_cap'; s[2] = 'timeout'
        other = x.copy(); other[3] = 22.; other[4] = 10.
        high = np.ones(n, bool); high[5] = False; minority = np.ones(n, bool); minority[6] = False
        eligible, masks = A.filter_masks(x, other, ref, s, s, np.full(n, 'numeric'), generation,
            np.full(n, 'eos'), high, np.full(n, 3), minority, np.full(n, 20.))
        self.assertEqual(np.flatnonzero(masks['A']).tolist(), [0, 5, 6])
        self.assertEqual(np.flatnonzero(masks['B']).tolist(), [0, 6])
        self.assertEqual(np.flatnonzero(masks['C']).tolist(), [0])
        self.assertTrue(close(1e9, 1e9+500)); self.assertFalse(close(1., 1.01))

    def test_full_synthetic_analysis_orders_SHA_and_partial_rejection(self):
        source, ids = synthetic(self.root/'run')
        report = A.analyze(source, self.root/'out', allow_partial=True)
        self.assertEqual(report['metrics']['filters']['A']['rows'], 32)
        self.assertEqual(report['metrics']['filters']['B']['rows'], 8)
        self.assertEqual(report['metrics']['filters']['C']['rows'], 8)
        again = A.analyze(source, self.root/'out2', allow_partial=True)
        for rule in ('control', 'A', 'B', 'C'):
            self.assertEqual(report['orders'][rule]['file']['sha256'], again['orders'][rule]['file']['sha256'])
            order = np.load(self.root/'out'/rule/'order.npy'); self.assertNotIn(0, order)
            self.assertEqual(len(order), len(np.unique(order)))
        with self.assertRaises(AssertionError): A.analyze(source, self.root/'strict')
        p = source/MODELS[1]/'generation-00000000.npz'
        with np.load(p) as z: arrays = {k:z[k][:-1] for k in z.files}
        save_npz(p, **arrays); publish(p.with_suffix('.receipt.json'), [p], phase='generation')
        partial = A.analyze(source, self.root/'partial', allow_partial=True)
        self.assertTrue(partial['partial']); self.assertEqual(partial['paired_rows'], 31)

    def test_live_snapshot_recovery_and_tamper(self):
        source, ids = synthetic(self.root/'run')
        stage_snapshot(source.parent, self.root/'stage', 'friend1/nlp-hw02-d1-live', 'fixture')
        restored = restore_snapshot(self.root/'stage', self.root/'restored')
        self.assertEqual(sha(source/'selection.npz'), sha(restored/'d1/selection.npz'))
        verify_receipt(restored/'d1/selection.receipt.json')
        target = self.root/'stage/d1__selection.npz'
        with target.open('ab') as stream: stream.write(b'bad')
        with self.assertRaises(AssertionError): restore_snapshot(self.root/'stage', self.root/'bad')

    def test_worker_isolation_watchdog_and_missing_secrets_gate(self):
        from d1_session import Watchdog, launch
        from l2_live import gate
        fake_torch = type('Torch', (), {'cuda':type('Cuda', (), {'device_count':staticmethod(lambda:2),
            'get_device_name':staticmethod(lambda _: 'T4')})})
        def absent(_): raise KeyError('fixture missing')
        self.assertFalse(gate(fake_torch, secret_getter=absent)['passed'])
        started = time.monotonic()
        cfg = dict(working_root=str(self.root), temp_root=str(self.root), session_kind='d1',
            session_start_monotonic=started, deadline_monotonic=started+5, budget_seconds=5,
            hard_deadline_monotonic=started+10, hard_shutdown_seconds=10, terminate_grace_seconds=.05)
        worker = self.root/'fixture.py'
        worker.write_text('''import sys,os,json,time
from pathlib import Path
gpu=int(sys.argv[sys.argv.index('--gpu')+1])
cfg=json.loads(Path(sys.argv[-1]).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert not any(os.environ.get(k) for k in ('KAGGLE_KEY','KAGGLE_USERNAME','KAGGLE_API_TOKEN','KAGGLE_USER_SECRETS_TOKEN'))
if gpu==0: sys.exit(1)
time.sleep(.2)
p=Path(cfg['working_root'])/'d1/C-long2x';p.mkdir(parents=True)
(p/'summary.json').write_text(json.dumps({'status':'complete'}))
''')
        watchdog = Watchdog(cfg).start()
        with patch.dict('os.environ', {'KAGGLE_KEY':'d1-fixture-key', 'KAGGLE_USERNAME':'d1-fixture-user'}):
            result = launch(cfg, worker, watchdog, cpu_test=True, heartbeat=.03)
        self.assertEqual(result['exit_codes'], [1, 0]); self.assertFalse(result['unreaped_worker_pids'])
        self.assertEqual(result['worker_status']['C-long2x'], 'complete')
        self.assertFalse((self.root/'stop-request.json').exists())
        self.assertEqual(watchdog.hard_deadline, cfg['hard_deadline_monotonic'])

    def test_v6_greedy_and_real_execution_on_tiny_deterministic_program(self):
        # Real incremental decoder (not a fake token stream), trained-to-bias EOS.
        import final_core as FC
        model = FC.SimpleTransformer(d_model=8, heads=2, encoder_layers=1, decoder_layers=1,
                                    swiglu_dim=16, vocab=24, dropout=0.).eval()
        with torch.no_grad(): model.head.weight.zero_(); model.head.bias.zero_(); model.head.bias[2] = 100.
        cache = Cache(3); table = [b'']*24
        # EOS generates an invalid fragment; reference receipts still survive.
        result = G.generation_wave(model, cache, np.arange(3, dtype=np.uint32), table, CPU,
            self.root/'wave.npz', lambda:None, time.monotonic()+30,
            evidence=dict(row_id=np.arange(3), reference=np.full(3, 3.)), batch=2, cap=4)
        self.assertEqual(result['rows'], 3); self.assertEqual(result['reference_reused'], 3)
        with np.load(self.root/'wave.npz') as z:
            self.assertTrue(np.all(z['generation_status'] == 'restore_error'))
        executor = G.Executor(2)
        jobs = [{'code':'def solve():\n    return 1+2\n'}, {'code':'def solve():\n    return 4/2\n'}]
        try:
            executor.submit(jobs); self.assertTrue(executor.drain(time.monotonic()+30)); executor.attach(jobs)
            self.assertEqual([j['execution']['value'] for j in jobs], [3., 2.])
        finally: executor.close()


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = unittest.main(verbosity=2, exit=False).result
    write(WORK/'checks/kaggle-d1-regressions.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), GPU=False, push=False,
        torch=str(torch.__version__), numpy=np.__version__, synthetic_analysis_and_L2_orders=True,
        frozen_v6_decoder_and_executor=True))
    sys.exit(0 if result.wasSuccessful() else 1)
