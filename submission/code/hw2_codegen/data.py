"""Чтение sealed compact-кеша и точный порядок исходного обучения L1/L2.

Источник: только исходные пары tinysms_qa_12m. Целевые строки не синтезируются.
Полная подготовка корпуса сохранена в work (26a/26c/L1 cache).
"""

import json
from pathlib import Path

import numpy as np

ORDER_SEED = 2026101201


class CompactCache:
    """Int16 content + int64 offsets; mmap не загружает весь корпус в RAM.

    Совместим с compact-manifest.json боевого L1; source уже содержит BOS/EOS.
    """

    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root / "compact-manifest.json").read_text())
        if not manifest["sealed"] or manifest["representation"] != 30:
            raise ValueError("Expected sealed compact representation 30")
        self.parts, starts, sl, tl = [], [0], [], []
        for shard in manifest["shards"]:
            folder = self.root / shard["folder"]
            arrays = {
                name: np.load(
                    folder / (name + ".npy"), mmap_mode="r", allow_pickle=False
                )
                for name in ("source", "target", "source-offsets", "target-offsets")
            }
            for prefix in ("source", "target"):
                tokens, offsets = arrays[prefix], arrays[prefix + "-offsets"]
                if tokens.dtype != np.int16 or offsets.dtype != np.int64:
                    raise ValueError("Expected int16 tokens and int64 offsets")
                lengths = np.diff(offsets)
                if (
                    offsets[0] != 0
                    or offsets[-1] != len(tokens)
                    or np.any(lengths <= 0)
                ):
                    raise ValueError("Invalid cache offsets")
                if lengths.max() + (prefix == "target") > 2048:
                    raise ValueError("Position overflow")
            n = len(arrays["source-offsets"]) - 1
            if n != shard["rows"] or n != len(arrays["target-offsets"]) - 1:
                raise ValueError("Source/target row mismatch")
            self.parts.append(arrays)
            starts.append(starts[-1] + n)
            sl.append(np.diff(arrays["source-offsets"]).astype(np.uint16))
            tl.append(np.diff(arrays["target-offsets"]).astype(np.uint16))
        self.starts = np.asarray(starts)
        self.source_lengths, self.target_lengths = (
            np.concatenate(sl),
            np.concatenate(tl),
        )
        if len(self) != manifest["rows"]:
            raise ValueError("Manifest row mismatch")

    def __len__(self):
        return int(self.starts[-1])

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self.starts, index, side="right") - 1)
        local = index - int(self.starts[shard])
        arrays = self.parts[shard]
        return tuple(
            arrays[p][arrays[p + "-offsets"][local] : arrays[p + "-offsets"][local + 1]]
            for p in ("source", "target")
        )


def collate(cache, indices, device):
    """Right padding: src=BOS+Q+EOS, tgt=BOS+compact, labels=compact+EOS."""
    import torch

    pairs = [cache[int(i)] for i in indices]
    ns = [len(s) for s, t in pairs]
    nt = [len(t) for s, t in pairs]
    src = np.zeros((len(pairs), max(ns)), dtype=np.int64)
    tgt = np.zeros((len(pairs), max(nt) + 1), dtype=np.int64)
    labels = np.zeros_like(tgt)
    for j, (s, t) in enumerate(pairs):
        src[j, : len(s)] = s
        tgt[j, 0] = 1
        tgt[j, 1 : len(t) + 1] = t
        labels[j, : len(t)] = t
        labels[j, len(t)] = 2
    counts = {
        "pairs": len(pairs),
        "source_tokens": sum(ns),
        "target_tokens": sum(n + 1 for n in nt),
        "source_slots": src.size,
        "target_slots": tgt.size,
    }
    return (
        torch.from_numpy(src).to(device),
        torch.from_numpy(tgt).to(device),
        torch.from_numpy(labels).to(device),
        counts,
    )


def make_order(cache, seed=ORDER_SEED, batch=128, chunk=4096):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cache)).astype(np.uint32)
    for lo in range(0, len(order), chunk):
        part = order[lo : lo + chunk]
        ns = cache.source_lengths[part].astype(np.int32)
        nt = cache.target_lengths[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns + nt + 1))]
        full = len(part) // batch
        if full:
            part[: full * batch] = (
                part[: full * batch]
                .reshape(full, batch)[rng.permutation(full)]
                .reshape(-1)
            )
        order[lo : lo + len(part)] = part
    assert np.array_equal(np.sort(order), np.arange(len(cache), dtype=np.uint32))
    return order


def threshold_for(lengths, fraction):
    """Nearest attainable >=threshold fraction; ties choose the higher threshold."""
    counts = np.bincount(np.asarray(lengths, dtype=np.int64))
    tails = np.cumsum(counts[::-1])[::-1]
    options = [
        (abs(int(tails[t]) / len(lengths) - fraction), -t, t)
        for t in np.flatnonzero(counts)
    ]
    options.append((fraction, -len(counts), len(counts)))
    return min(options)[2]


def long2x_order(
    cache,
    heldout_lengths,
    original_lengths=None,
    seed=2026101302,
    batch=128,
    chunk=4096,
):
    """Каждая длинная строка встречается дважды; порог по доле heldout >=90.

    original_lengths — исходные raw-BPE длины, выровненные с compact-кешем.
    Без них используется compact analogue порога, как в L2 fallback.
    """
    lengths = (
        cache.target_lengths
        if original_lengths is None
        else np.asarray(original_lengths)
    )
    fraction = float(np.mean(np.asarray(heldout_lengths) >= 90))
    threshold = threshold_for(lengths, fraction)
    slots = np.concatenate(
        (
            np.arange(len(cache), dtype=np.uint32),
            np.flatnonzero(lengths >= threshold).astype(np.uint32),
        )
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(slots)
    for lo in range(0, len(slots), chunk):
        part = slots[lo : lo + chunk]
        ns = cache.source_lengths[part].astype(np.int32)
        nt = cache.target_lengths[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns + nt + 1))]
        full = len(part) // batch
        if full:
            part[: full * batch] = (
                part[: full * batch]
                .reshape(full, batch)[rng.permutation(full)]
                .reshape(-1)
            )
        slots[lo : lo + len(part)] = part
    return slots


def take(order, cursor, batch):
    """Продолжить ту же перестановку циклически; cursor считает предъявления."""
    return np.asarray(order[(np.arange(batch, dtype=np.int64) + cursor) % len(order)])
