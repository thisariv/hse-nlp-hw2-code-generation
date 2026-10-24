"""Reversible representations; no new semantic targets or changed row selection."""
import ast, io, math, re, tokenize
from fractions import Fraction
from tokenizers import Tokenizer, AddedToken

MODES = {'header_tail': '<|header_tail|>', 'header': '<|header|>', 'raw': '<|raw|>'}
NUMBERS = [f'<|N{i}|>' for i in range(1, 21)]
NUMBER = re.compile(r'(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?:/\d+)?')

def no_comments(code):
    lines = code.splitlines(keepends=True)
    try:
        spans = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    for t in reversed(spans):
        i = t.start[0] - 1
        lines[i] = lines[i][:t.start[1]].rstrip(' \t') + lines[i][t.end[1]:]
    return ''.join(lines)

def no_blank(code):
    protected = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.STRING:
                protected.update(range(t.start[0], t.end[0] + 1))
        candidate = ''.join(line for i, line in enumerate(code.splitlines(keepends=True), 1)
                            if line.strip() or i in protected)
        if ast.dump(ast.parse(code)) == ast.dump(ast.parse(candidate)):
            return candidate
    except (tokenize.TokenError, SyntaxError, ValueError, RecursionError):
        pass
    return code

def restore_compact(text):
    mode = next((k for k, token in MODES.items() if text.startswith(token)), None)
    if mode is None:
        raise ValueError('Missing representation mode')
    fragment = text[len(MODES[mode]):]
    if mode == 'raw':
        return fragment
    if mode == 'header':
        return 'def solve() -> int:\n' + fragment
    lines = fragment.splitlines(keepends=True)
    if not lines or not lines[-1].startswith('    '):
        raise ValueError('Missing final result expression')
    lines[-1] = '    result = ' + lines[-1][4:]
    body = ''.join(lines)
    return 'def solve() -> int:\n' + body + ('' if body.endswith('\n') else '\n') + '    return result\n'

def compact(code):
    normalized = no_blank(no_comments(code))
    raw = MODES['raw'] + normalized
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0] != 'def solve() -> int:\n':
        return raw
    try:
        tree = ast.parse(normalized)
        f = tree.body[0]
        if len(tree.body) != 1 or not isinstance(f, ast.FunctionDef):
            return raw
        tail = (len(f.body) >= 2 and isinstance(f.body[-1], ast.Return)
                and isinstance(f.body[-1].value, ast.Name) and f.body[-1].value.id == 'result'
                and isinstance(f.body[-2], ast.Assign) and len(f.body[-2].targets) == 1
                and isinstance(f.body[-2].targets[0], ast.Name) and f.body[-2].targets[0].id == 'result'
                and lines[-1].strip() == 'return result' and lines[-2].startswith('    result = '))
        if tail:
            lines[-2] = '    ' + lines[-2][len('    result = '):]
            text = MODES['header_tail'] + ''.join(lines[1:-1])
        else:
            text = MODES['header'] + ''.join(lines[1:])
        # Protect multiline assignments/tails and any unusual source layout.
        return text if ast.dump(ast.parse(restore_compact(text))) == ast.dump(tree) else raw
    except (SyntaxError, ValueError, IndexError, RecursionError):
        return raw

def question_mapping(question):
    mapping, occurrences = {}, []
    for match in NUMBER.finditer(question):
        try:
            text = match.group().replace(',', '')
            value = Fraction(text) if '/' not in text else Fraction(text.split('/')[0]) / Fraction(text.split('/')[1])
            if not math.isfinite(float(value)):
                continue
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if value not in mapping and len(mapping) < 20:
            # Integers stay integers (range/index/slices); decimal/fraction values stay floats.
            number = int(value) if value.denominator == 1 else float(value)
            mapping[value] = (NUMBERS[len(mapping)], number)
        if value in mapping:
            occurrences.append((match.start(), match.end(), mapping[value][0]))
    marked = question
    for start, end, token in reversed(occurrences):
        marked = marked[:start] + token + ' ' + marked[start:]
    return marked, mapping

def placeholder_code(code, mapping):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return code
    lines = code.encode('utf-8').splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    def literal(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = literal(node.operand)
            if v is not None:
                return -v if isinstance(node.op, ast.USub) else v
        return None
    def visit(node):
        v = literal(node)
        if v is not None and (type(v) is int or math.isfinite(v)):
            key = Fraction(str(v))
            if key not in mapping and type(v) is float:
                # Fraction question values such as2/3 map to the same Python float literal.
                equivalent = [q for q,(token,number) in mapping.items() if float(number) == v]
                if equivalent:key = equivalent[0]
            if key in mapping:
                token, number = mapping[key]
                # Preserve literal type; merely replacing 5.0 by integer 5 can change execution.
                replacement = token if type(v) is type(number) else ('float(' + token + ')' if type(v) is float else 'int(' + token + ')')
                changes.append((offsets[node.lineno - 1] + node.col_offset,
                                offsets[node.end_lineno - 1] + node.end_col_offset, replacement.encode()))
                return
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    raw = code.encode('utf-8')
    for start, end, replacement in sorted(changes, reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    return raw.decode('utf-8')

def restore_numbers(text, question):
    _, mapping = question_mapping(question)
    reverse = {token: '(' + repr(number) + ')' for token, number in mapping.values()}
    def replace(match):
        if match.group() not in reverse:
            raise ValueError('Unknown generated number placeholder')
        return reverse[match.group()]
    return re.sub(r'<\|N\d+\|>', replace, text)

def transform(question, code, variant):
    if variant == 30:
        return question, compact(code)
    marked, mapping = question_mapping(question)
    return marked, placeholder_code(code, mapping)

def restore(text, question, variant):
    return restore_compact(text) if variant == 30 else restore_numbers(text, question)

def tokenizer(path, variant):
    tok = Tokenizer.from_file(str(path))
    assert tok.get_vocab_size() == 8189
    added = list(MODES.values()) if variant == 30 else NUMBERS
    assert tok.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in added]) == len(added)
    # Original byte alphabet is unchanged; added specials decode literally, never get skipped.
    visible = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    bchars = {b:chr(b) for b in visible}
    for i,b in enumerate(b for b in range(256) if b not in bchars):
        bchars[b] = chr(256+i)
    charbytes = {c:b for b,c in bchars.items()}
    tokenbytes = [b'',b'',b''] + [bytes(charbytes[c] for c in tok.id_to_token(i)) for i in range(8189)]
    tokenbytes += [t.encode() for t in added]
    assert all(tok.encode(t, add_special_tokens=False).ids == [8189+i] for i,t in enumerate(added))
    return tok, tokenbytes

def init_encoder(path, variant):
    global _TOKENIZER, _VARIANT
    _TOKENIZER, _ = tokenizer(path, variant)
    _VARIANT = variant

def encode_block(rows):
    import numpy as np
    from array import array
    texts = [transform(q,c,_VARIANT) for q,c in rows]
    sources = _TOKENIZER.encode_batch([q for q,c in texts],add_special_tokens=False)
    targets = _TOKENIZER.encode_batch([c for q,c in texts],add_special_tokens=False)
    s,t,so,to = array('h'),array('h'),array('q',[0]),array('q',[0])
    for source,target in zip(sources,targets):
        s.extend([1]+[i+3 for i in source.ids]+[2]);so.append(len(s))
        t.extend(i+3 for i in target.ids);to.append(len(t))
    return [np.frombuffer(a,dtype=d) for a,d in zip((s,t,so,to),('int16','int16','int64','int64'))]
