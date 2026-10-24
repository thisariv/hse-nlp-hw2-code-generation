"""Byte-level BPE: PAD/BOS/EOS=0/1/2, BPE id+3, три compact mode tokens."""

import ast
import io
import tokenize

from tokenizers import AddedToken, Tokenizer

from .model import BOS, EOS, PAD, TOKEN_OFFSET

MODES = {"header_tail": "<|header_tail|>", "header": "<|header|>", "raw": "<|raw|>"}


def no_comments(code):
    """Удалить только комментарии, сохранив строки."""
    lines = code.splitlines(keepends=True)
    try:
        spans = [
            t
            for t in tokenize.generate_tokens(io.StringIO(code).readline)
            if t.type == tokenize.COMMENT
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    for t in reversed(spans):
        i = t.start[0] - 1
        lines[i] = lines[i][: t.start[1]].rstrip(" \t") + lines[i][t.end[1] :]
    return "".join(lines)


def no_blank(code):
    """Убрать пустые строки только при неизменном AST."""
    protected = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.STRING:
                protected.update(range(t.start[0], t.end[0] + 1))
        candidate = "".join(
            (
                line
                for i, line in enumerate(code.splitlines(keepends=True), 1)
                if line.strip() or i in protected
            )
        )
        if ast.dump(ast.parse(code)) == ast.dump(ast.parse(candidate)):
            return candidate
    except (tokenize.TokenError, SyntaxError, ValueError, RecursionError):
        pass
    return code


def restore_compact(text):
    """Восстановить полный solve() по явному токену режима."""
    mode = next((k for k, token in MODES.items() if text.startswith(token)), None)
    if mode is None:
        raise ValueError("Missing representation mode")
    fragment = text[len(MODES[mode]) :]
    if mode == "raw":
        return fragment
    if mode == "header":
        return "def solve() -> int:\n" + fragment
    lines = fragment.splitlines(keepends=True)
    if not lines or not lines[-1].startswith("    "):
        raise ValueError("Missing final result expression")
    lines[-1] = "    result = " + lines[-1][4:]
    body = "".join(lines)
    return (
        "def solve() -> int:\n"
        + body
        + ("" if body.endswith("\n") else "\n")
        + "    return result\n"
    )


def compact(code):
    """Удалить шаблон solve/result/return, проверив равенство AST."""
    normalized = no_blank(no_comments(code))
    raw = MODES["raw"] + normalized
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0] != "def solve() -> int:\n":
        return raw
    try:
        tree = ast.parse(normalized)
        f = tree.body[0]
        if len(tree.body) != 1 or not isinstance(f, ast.FunctionDef):
            return raw
        tail = (
            len(f.body) >= 2
            and isinstance(f.body[-1], ast.Return)
            and isinstance(f.body[-1].value, ast.Name)
            and (f.body[-1].value.id == "result")
            and isinstance(f.body[-2], ast.Assign)
            and (len(f.body[-2].targets) == 1)
            and isinstance(f.body[-2].targets[0], ast.Name)
            and (f.body[-2].targets[0].id == "result")
            and (lines[-1].strip() == "return result")
            and lines[-2].startswith("    result = ")
        )
        if tail:
            lines[-2] = "    " + lines[-2][len("    result = ") :]
            text = MODES["header_tail"] + "".join(lines[1:-1])
        else:
            text = MODES["header"] + "".join(lines[1:])
        return (
            text
            if ast.dump(ast.parse(restore_compact(text))) == ast.dump(tree)
            else raw
        )
    except (SyntaxError, ValueError, IndexError, RecursionError):
        return raw


def encode(tokenizer, text):
    return [
        i + TOKEN_OFFSET for i in tokenizer.encode(text, add_special_tokens=False).ids
    ]


def decode_bytes(tokenbytes, ids):
    chunks = []
    for i in ids:
        if i == EOS:
            break
        if i >= TOKEN_OFFSET:
            chunks.append(tokenbytes[i])
    return b"".join(chunks)


def active_tokens(ids):
    ids = [int(i) for i in ids if i != PAD]
    return ids[: ids.index(EOS) + 1] if EOS in ids else ids


def tokenizer(path, variant=30):
    if variant != 30:
        raise ValueError("Submission uses compact representation 30")
    tok = Tokenizer.from_file(str(path))
    assert tok.get_vocab_size() == 8189
    added = list(MODES.values()) if variant == 30 else []
    assert tok.add_special_tokens(
        [AddedToken(t, normalized=False, special=True) for t in added]
    ) == len(added)
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    bchars = {b: chr(b) for b in visible}
    for i, b in enumerate(b for b in range(256) if b not in bchars):
        bchars[b] = chr(256 + i)
    charbytes = {c: b for b, c in bchars.items()}
    tokenbytes = [b"", b"", b""] + [
        bytes(charbytes[c] for c in tok.id_to_token(i)) for i in range(8189)
    ]
    tokenbytes += [t.encode() for t in added]
    assert all(
        (
            tok.encode(t, add_special_tokens=False).ids == [8189 + i]
            for i, t in enumerate(added)
        )
    )
    return (tok, tokenbytes)


def encode_question(tok, question):
    """Вопрос без переписывания чисел; BOS+content+EOS."""
    return [BOS] + encode(tok, question) + [EOS]
