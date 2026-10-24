"""Every point: heldout4k and old991 greedy/CE. Optional final-session adds test/K8."""
import gc
import time
from pathlib import Path
import torch
import exp26_exec as N
import l1_eval as E
import l2_eval as Q
import s3_representation as S
from kaggle_io import sha, write


def evaluate(point, out, inputs, sets, refs, device, deadline, final=False, smoke=False):
    if final:
        return Q.evaluate(point, out, inputs, sets, refs, device, deadline, True, smoke,
                          vote_cap_seconds=1200., test_rows=64 if smoke else None)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tok, table = S.tokenizer(Path(inputs)/'tokenizer.json', 30); model = E.load(point, device)
    result = dict(weights_sha256=sha(point), splits={}, fresh_evaluated=0)
    try:
        for split in ('heldout4k', 'old991'):
            rows = sets[split]
            examples, timing = E.generate(model, rows, tok, table, device, deadline, smoke, batch=16 if smoke else 256)
            timing['sequences_per_second'] = len(rows)/timing['seconds']
            write(out/(split+'-generations.json'), examples)
            execution = E.execute(examples, smoke, deadline); write(out/(split+'-execution.json'), execution)
            result['splits'][split] = dict(exec=N.compare_exec(refs[split], execution),
                                         CE=E.ce(model, rows, tok, device, deadline, smoke), generation=timing)
            write(out/'partial-metrics.json', result)
        Q.greedy_journals(out, inputs, sets, refs)
        write(out/'metrics.json', result)
        return result
    finally:
        del model; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
