"""Question-given-compact-code CE; same tokenizer/mode/BOS/EOS as training."""
import time
import numpy as np
import torch
from torch.nn import functional as F
import exp26_core as C
import s3_representation as S
from exp26_order import collate
from p1_math import ReverseCache


class PairCache:
    def __init__(self, rows, tokenizer):
        self.rows = [(np.asarray([C.BOS, *C.encode(tokenizer, r['question']), C.EOS]),
                      np.asarray(C.encode(tokenizer, S.compact(r['code'])))) for r in rows]
        self.source_lengths = np.asarray([len(s) for s,t in self.rows])
        self.target_lengths = np.asarray([len(t) for s,t in self.rows])

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[int(i)]


@torch.no_grad()
def reverse_ce(model, cache, device, deadline, batch=16):
    was_training = model.training; model.eval(); started = time.monotonic()
    order = sorted(range(len(cache)), key=lambda i:int(cache.source_lengths[i]+cache.target_lengths[i]))
    total = 0.; tokens = 0; done = 0; retries = []
    try:
        while done < len(order):
            C.check_deadline(deadline)
            src=tgt=labels=logits=loss=None
            try:
                src,tgt,labels,counts = collate(cache, order[done:done+batch], device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                    logits = model(src,tgt)
                loss = F.cross_entropy(logits.float().reshape(-1,model.vocab), labels.reshape(-1),
                                       ignore_index=C.PAD, reduction='sum')
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite reviewer CE')
                total += float(loss); tokens += counts['target_tokens']; done += len(src)
            except torch.cuda.OutOfMemoryError:
                retries.append(dict(offset=done,batch=batch)); batch //= 2
                if batch < 1: raise
            finally:
                del src,tgt,labels,logits,loss
                if retries and device.type=='cuda':torch.cuda.empty_cache()
        return dict(complete=True, rows=done, tokens=tokens, CE=total/tokens,
                    sum_NLL=total, seconds=time.monotonic()-started, OOM_retries=retries,
                    direction='question|compact-code; question EOS scored, PAD/BOS excluded')
    finally:
        model.train(was_training)
