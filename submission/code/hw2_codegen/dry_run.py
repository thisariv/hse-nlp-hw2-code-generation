"""Восемь синтетических вопросов и случайные tiny-модели для CPU smoke.

Не оценивает качество и не заменяет cold T4 замер на test1919.
"""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from .model import from_spec
from .tokenizer_repr import MODES


def fixtures():
    vocabulary = [
        "[UNK]",
        *MODES.values(),
        "How",
        "many",
        "is",
        "plus",
        "?",
        *map(str, range(12)),
        "def solve() -> int:\n    return 1\n",
    ]
    tok = Tokenizer(
        models.WordLevel({v: i for i, v in enumerate(vocabulary)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    table = [b"", b"", b""] + [word.encode() for word in vocabulary]
    spec = dict(
        d_model=16,
        heads=2,
        encoder_layers=1,
        decoder_layers=1,
        swiglu_dim=24,
        vocab=len(table),
        max_positions=64,
    )
    torch.manual_seed(42)
    models_by_name = {name: from_spec(spec).eval() for name in ("tiny-a", "tiny-b")}
    rows = [dict(question=f"How many is {i} plus 1 ?") for i in range(8)]
    specs = [dict(name=name, K1=1, K2=2) for name in models_by_name]
    return rows, models_by_name, tok, table, specs
