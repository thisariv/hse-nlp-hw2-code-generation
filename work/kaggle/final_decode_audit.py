"""Reproducible static arithmetic + compact observations from existing T4 receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FINAL = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'


def kv_reads(n, source, steps, layers=6, width=512, itemsize=2):
    # Logical reads: independent Q=1 attention per lane; one K+V read per step.
    # Not measured DRAM traffic: on-chip caching/SDPA implementation may change it.
    unit = 2*layers*n*width*itemsize
    return {'self_KV_logical_read_bytes': unit*steps*(steps+1)//2,
            'cross_KV_logical_read_bytes': unit*source*steps,
            'KV_logical_read_seconds_at_320GBps': (unit*steps*(steps+1)//2+unit*source*steps)/320e9}


def main():
    report = {'kind': 'static estimates and existing observations; not a new GPU profile',
              'KV_preallocated_in_baseline': True, 'cross_projected_once': True,
              'per_step_cat': 0, 'per_step_item': 0, 'per_batch_Python_loop_in_decoder': False,
              'EOS_sync_every_steps': 16, 'baseline_torch_threads': 1,
              'self_KV_capacity_bytes_at_1024_512': 2*6*1024*512*512*2,
              'cross_KV_bytes_at_1024_source203': 2*6*1024*203*512*2,
              'projection_FLOPs_per_step_at_1024': 2*1024*(6*(6*512**2+3*512*1368)+512*8195),
              'KV_model': 'logical K/V reads, no measured DRAM counter; ignores attention buffers/activations/weights',
              'runs': []}
    for path in sorted(FINAL.rglob('timing.json')):
        if not any(x in path.parts for x in ('inference-v4', 'inference-v5', 'inference-v5b')):
            continue
        data = json.loads(path.read_text())
        batches = data['generation_batches']
        row = {'path': str(path.relative_to(WORK)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'batch_count': len(batches), 'cap512_batches': sum(b['decode_steps'] == 512 for b in batches),
               'greedy_phase_seconds': data['phases_seconds']['greedy_generation_restore_submit']}
        if all('wall_seconds' in b for b in batches):
            greedy = [b for b in batches if b['phase'] == 'greedy']
            row['greedy_batches'] = [{k: b[k] for k in ('sequences', 'source_tokens', 'decode_steps', 'wall_seconds')}
                                      | kv_reads(b['sequences'], b['source_tokens'], b['decode_steps']) for b in greedy]
            row['cap512_fraction_greedy_batch_wall'] = sum(b['wall_seconds'] for b in greedy if b['decode_steps']==512)/sum(b['wall_seconds'] for b in greedy)
        report['runs'].append(row)
    target = WORK/'checks/final-decode-audit.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
