"""WSD-рецепт L1–L3/P1 без Kaggle orchestration.

Batch 128, AdamW wd=.01, clip=1, AMP FP16/GradScaler1024; constant main,
затем disposable cooldown. Main и cooldown разделены checkpoint-ом.
Скрипт не скачивает данные и не запускает обучение при импорте.
"""

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import CompactCache, collate, make_order, take
from .model import PAD, from_spec

MODEL_C = dict(
    d_model=512,
    heads=8,
    encoder_layers=6,
    decoder_layers=6,
    swiglu_dim=1368,
    vocab=8195,
)
COMPACT_EDGES = (53, 79, 105)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(update, peak, warmup, main_updates, cooldown_updates, kind="sqrt"):
    """WSD на оси updates: warmup → constant → linear или 1-sqrt до нуля.

    Боевые L2/L3 задавали длительность main по wall time; точное число updates
    и старт cooldown берутся из training.json выбранного checkpoint.
    """
    if update <= main_updates:
        return peak * min(1.0, update / max(1, warmup))
    position = min(1.0, (update - main_updates) / max(1, cooldown_updates))
    if kind not in ("sqrt", "linear"):
        raise ValueError("cooldown must be sqrt or linear")
    return peak * (1.0 - (math.sqrt(position) if kind == "sqrt" else position))


def truncation_mask(scores, lengths, q=0.1, edges=COMPACT_EDGES):
    """Глобальная mask effective batch: удалить q самых трудных, с квотами по длине.

    Округление q*N, largest remainder, минимум один retained пример в каждом bin.
    При равной CE побеждает исходная позиция. Это точная политика P1.
    """
    scores = np.asarray(scores, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if (
        scores.ndim != 1
        or scores.shape != lengths.shape
        or not np.isfinite(scores).all()
        or not 0 <= q < 1
    ):
        raise FloatingPointError("Invalid per-example loss")
    bins = np.searchsorted(edges, lengths, side="right")
    counts = np.bincount(bins, minlength=len(edges) + 1)
    capacities = np.maximum(0, counts - 1)
    desired = min(int(math.floor(q * len(scores) + 0.5)), int(capacities.sum()))
    quotas = np.minimum(np.floor(q * counts).astype(int), capacities)
    while int(quotas.sum()) < desired:
        available = [b for b in range(len(counts)) if quotas[b] < capacities[b]]
        b = min(available, key=lambda b: (-(q * counts[b] - quotas[b]), b))
        quotas[b] += 1
    keep = np.ones(len(scores), dtype=bool)
    for b, quota in enumerate(quotas):
        positions = np.flatnonzero(bins == b)
        ranked = sorted(positions, key=lambda j: (-scores[j], int(j)))
        keep[ranked[:quota]] = False
    return keep


def update(model, optimizer, scaler, cache, indices, device, micro, lr, q=0.0):
    """Один effective batch; token-weighted CE, EOS учитывается, PAD игнорируется.

    q=0 повторяет L1.update. q>0 повторяет P1 graph_truncated_update: mask по
    всему effective batch до backward, знаменатель — число retained tokens.
    При OOM/нечисловом состоянии останавливаемся: безопасно продолжать из full
    checkpoint. Автоматические откаты и выбор micro остаются в work/.
    """
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    lengths = np.asarray([len(cache[int(i)][1]) for i in indices])
    tokens = int(sum(lengths + 1))
    graphs, scores = [], []
    for lo in range(0, len(indices), micro):
        src, tgt, labels, _ = collate(cache, indices[lo : lo + micro], device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()
        ):
            logits = model(src, tgt)
            if q:
                losses = (
                    F.cross_entropy(
                        logits.reshape(-1, model.vocab),
                        labels.reshape(-1),
                        ignore_index=PAD,
                        reduction="none",
                    )
                    .reshape_as(labels)
                    .sum(1)
                )
            else:
                loss = (
                    F.cross_entropy(
                        logits.reshape(-1, model.vocab),
                        labels.reshape(-1),
                        ignore_index=PAD,
                        reduction="sum",
                    )
                    / tokens
                )
        if q:
            values = losses.detach() / labels.ne(PAD).sum(1)
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("Nonfinite selection loss")
            scores.extend(values.cpu().tolist())
            graphs.append(losses)
            del losses, values
        else:
            if not bool(torch.isfinite(loss).detach()):
                raise FloatingPointError("Nonfinite loss")
            scaler.scale(loss).backward()
            total += float(loss.detach())
            del loss
        del src, tgt, labels, logits
    keep = np.ones(len(indices), dtype=bool)
    if q:
        keep = truncation_mask(scores, lengths, q)
        tokens = int(sum(lengths[keep] + 1))
        lo = 0
        while graphs:
            losses = graphs.pop(0)
            mask = torch.as_tensor(keep[lo : lo + len(losses)], device=device)
            loss = losses[mask].sum() / tokens
            if not bool(torch.isfinite(loss).detach()):
                raise FloatingPointError("Nonfinite retained loss")
            scaler.scale(loss).backward()
            total += float(loss.detach())
            lo += len(losses)
            del losses, loss, mask
    for group in optimizer.param_groups:
        group["lr"] = lr
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    skipped = (
        bool(
            any(float(t) > 0 for t in scaler._found_inf_per_device(optimizer).values())
        )
        if scaler.is_enabled()
        else False
    )
    if not skipped and not bool(torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite successful gradient")
    scaler.step(optimizer)
    scaler.update()
    tensors = list(model.parameters()) + [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ]
    if not bool(torch.isfinite(torch.stack(torch._foreach_norm(tensors))).all()):
        raise FloatingPointError("Nonfinite model/Adam after update; restore checkpoint")
    return dict(
        CE=total,
        tokens=tokens,
        AMP_skip=int(skipped),
        keep=keep.tolist(),
        gradient_norm=float(norm) if torch.isfinite(norm) else None,
    )


def save_checkpoint(path, model, optimizer, scaler, state, model_spec):
    """Полное состояние перед cooldown; model-only экспортируется отдельно."""
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            state=copy.deepcopy(state),
            model_spec=model_spec,
            torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        ),
        path,
    )


def train(config):
    """Запустить явное число updates по сохранённому order.npy либо L1 permutation."""
    device = torch.device(config.get("device", "cuda"))
    seed_all(config.get("seed", 42))
    cache = CompactCache(config["cache"])
    order = (
        np.load(config["order"], allow_pickle=False)
        if config.get("order")
        else make_order(cache)
    )
    spec = config.get("model_spec", MODEL_C)
    model = from_spec(spec, dropout=config.get("dropout", 0.0)).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["peak_lr"],
        weight_decay=0.01,
        **({"fused": True} if device.type == "cuda" else {}),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=device.type == "cuda", init_scale=1024
    )
    state = dict(updates=0, cursor=0, AMP_skips=0)
    if config.get("resume"):
        # Load only one's own trusted full training checkpoint (contains numpy RNG).
        payload = torch.load(config["resume"], map_location=device, weights_only=False)
        if payload["model_spec"] != spec:
            raise ValueError("Resume model_spec mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        state = payload["state"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    main, cooldown = config["main_updates"], config["cooldown_updates"]
    with (output / "training.jsonl").open("a") as log:
        for step in range(state["updates"] + 1, main + cooldown + 1):
            if step == main + 1:
                save_checkpoint(
                    output / "stable.pt", model, optimizer, scaler, state, spec
                )
            lr = learning_rate(
                step,
                config["peak_lr"],
                config.get("warmup_updates", 1000),
                main,
                cooldown,
                config.get("cooldown", "linear"),
            )
            q = (
                config.get("cooldown_truncation", 0.0)
                if step > main
                else config.get("main_truncation", 0.0)
            )
            indices = take(order, state["cursor"], config.get("batch", 128))
            result = update(
                model,
                optimizer,
                scaler,
                cache,
                indices,
                device,
                config.get("micro_batch", 64),
                lr,
                q,
            )
            state.update(
                updates=step,
                cursor=state["cursor"] + len(indices),
                AMP_skips=state["AMP_skips"] + result["AMP_skip"],
            )
            log.write(json.dumps(dict(step=step, lr=lr, q=q, **result)) + "\n")
            if step % config.get("checkpoint_every", 1000) == 0:
                save_checkpoint(
                    output / "latest.pt", model, optimizer, scaler, state, spec
                )
        save_checkpoint(output / "latest.pt", model, optimizer, scaler, state, spec)
    torch.save(
        dict(
            format="nlp-hw02-model-only-v1",
            optimizer_included=False,
            model_spec=spec,
            model={k: v.detach().cpu().half() for k, v in model.state_dict().items()},
        ),
        output / "weights-fp16.pt",
    )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()))
