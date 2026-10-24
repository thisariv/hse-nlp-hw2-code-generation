"""CPU research utilities; never train on diagnostics or substituted programs."""
import ast
import copy
import difflib
import hashlib
import io
import json
import math
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
LOCAL = WORK / 'research/outputs/data-ceiling'
LOCAL.mkdir(parents=True, exist_ok=True)
RAW = WORK / 'kaggle/staging/drive-merged/26-token-cache/raw'
S1 = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1'
INPUT = S1 / 'inputs'
sys.path.insert(0, str(WORK / 'kaggle/src'))
from exp26_cache import classify
from exp26_fast_exec import run_chunks

NUMBER = re.compile(r'(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?')
WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), range(21)))
WORDS.update(twice=2, double=2, doubled=2, triple=3, tripled=3, half=.5, dozen=12, quarter=.25, hundred=100, thousand=1000)
CONSTANTS = {0, 1, 2, 3, 4, 7, 12, 24, 60, 100, 365, 360, 1000}

def read(p):
    return json.loads(Path(p).read_text())

def write(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def question_numbers(q):
    out = []
    for m in NUMBER.finditer(q):
        s = m.group().replace(',', '')
        try:
            v = float(s) if '/' not in s else float(s.split('/')[0]) / float(s.split('/')[1])
            out.append((v, m.start(), m.end(), m.group()))
        except (ValueError, ZeroDivisionError):
            pass
    return out

def masked(q):
    return ' '.join(NUMBER.sub(' NUM ', q).lower().split())

def literals(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    return [float(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and type(n.value) in (int, float)]

def number_info(q, code):
    vals = literals(code)
    if vals is None:
        return {'parse_error': True}
    qn = [x[0] for x in question_numbers(q)]
    lexical = {WORDS[w] for w in re.findall(r'[a-z]+', q.lower()) if w in WORDS}
    derived = set(lexical)
    # Conservative algebraic percentage forms only where percent is explicit.
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', q.lower()):
        x = float(m.group(1)) / 100
        derived.update((x, 1+x, 1-x))
    counts = Counter()
    for v in vals:
        if v in qn or -v in qn:
            counts['question'] += 1
        elif any(math.isclose(v, x, abs_tol=1e-9, rel_tol=1e-9) for x in derived):
            counts['derived_or_word'] += 1
        elif v in CONSTANTS:
            counts['constant_heuristic'] += 1
        else:
            counts['other'] += 1
    return dict(counts=counts, literals=len(vals), exact_only=all(v in qn or -v in qn for v in vals),
                placeholders_plus_constants=counts['other'] == 0,
                repeated_question_numbers=len(qn) != len(set(qn)),
                repeated_used_value=any(qn.count(v)>1 and v in vals for v in set(qn)),
                question_numeric_count=len(qn))

def skeleton(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None
    names = {}
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in {'int', 'float', 'round', 'sum', 'range', 'len', 'max', 'min', 'abs', 'math'}:
                node.id = names.setdefault(node.id, f'v{len(names)}')
            return node
        def visit_Constant(self, node):
            if type(node.value) in (int, float):
                node.value = 0
            return node
    return ast.dump(Normalize().visit(tree), include_attributes=False)

def no_comments(code):
    lines = code.splitlines(keepends=True)
    try:
        spans = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    for t in reversed(spans):
        i = t.start[0]-1
        lines[i] = lines[i][:t.start[1]].rstrip(' \t') + lines[i][t.end[1]:]
    return ''.join(lines)

def no_blank(code):
    # Don't remove physical lines inside multiline strings.
    protected = set()
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.STRING:
                protected.update(range(t.start[0], t.end[0]+1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    candidate = ''.join(l for i, l in enumerate(code.splitlines(keepends=True), 1) if l.strip() or i in protected)
    # A blank line after explicit backslash continuation can be syntactic.
    # AST equality also protects other unusual newline-sensitive constructs.
    try:
        if ast.dump(ast.parse(code), include_attributes=False) != ast.dump(ast.parse(candidate), include_attributes=False):
            return code
    except (SyntaxError, ValueError, RecursionError):
        return code
    return candidate

def strip_frame(code):
    """Return compact fragment and reversible mode; unusual modules fall back."""
    lines = code.splitlines(keepends=True)
    if not lines or lines[0] != 'def solve() -> int:\n':
        return code, 'raw'
    tail = False
    try:
        tree = ast.parse(code)
        f = tree.body[0]
        tail = (len(tree.body)==1 and isinstance(f, ast.FunctionDef) and len(f.body)>=2
                and isinstance(f.body[-1], ast.Return) and isinstance(f.body[-1].value, ast.Name)
                and f.body[-1].value.id=='result' and isinstance(f.body[-2], ast.Assign)
                and len(f.body[-2].targets)==1 and isinstance(f.body[-2].targets[0], ast.Name)
                and f.body[-2].targets[0].id=='result'
                and lines[-1].strip()=='return result' and lines[-2].startswith('    result = '))
    except (SyntaxError, ValueError):
        return code, 'raw'
    if tail:
        lines[-2] = '    ' + lines[-2][len('    result = '):]
        return ''.join(lines[1:-1]), 'header_tail'
    return ''.join(lines[1:]), 'header'

def restore(fragment, mode):
    if mode == 'raw':
        return fragment
    if mode == 'header':
        return 'def solve() -> int:\n' + fragment
    lines = fragment.splitlines(keepends=True)
    lines[-1] = '    result = ' + lines[-1][4:]
    return 'def solve() -> int:\n' + ''.join(lines) + '    return result\n'

def substitute(code, oldq, newq, rule='order'):
    """AST numeric substitution, no generated training targets; abstain on ambiguity."""
    old, new = question_numbers(oldq), question_numbers(newq)
    if not old or (rule!='aligned_role' and len(old)!=len(new)):
        return None, 'number_count'
    pairs = list(zip(old, new))
    if rule=='aligned_role':
        def tokens(q):
            return re.findall(r'_number_|[a-z]+',NUMBER.sub(' _number_ ',q).lower())
        aa,bb=tokens(oldq),tokens(newq)
        if aa.count('_number_')!=len(old) or bb.count('_number_')!=len(new):
            return None,'unparsed_number'
        ai={p:i for i,p in enumerate(j for j,t in enumerate(aa) if t=='_number_')}
        bi={p:i for i,p in enumerate(j for j,t in enumerate(bb) if t=='_number_')}
        pairs=[]
        for block in difflib.SequenceMatcher(None,aa,bb,autojunk=False).get_matching_blocks():
            for k in range(block.size):
                if block.a+k in ai and block.b+k in bi:
                    pairs.append((old[ai[block.a+k]],new[bi[block.b+k]]))
        used=set(literals(code) or []) & {x[0] for x in old}
        if not used.issubset({a[0] for a,b in pairs}):
            return None,'unmatched_used_role'
    if rule == 'role':
        def role(q, x):
            a = re.findall(r'[a-z]+', q[max(0,x[1]-28):x[1]].lower())[-2:]
            b = re.findall(r'[a-z]+', q[x[2]:x[2]+28].lower())[:2]
            return tuple(a+b)
        oldroles = [role(oldq, x) for x in old]
        newroles = [role(newq, x) for x in new]
        if len(set(oldroles))==len(oldroles) and set(oldroles)==set(newroles):
            pairs = [(old[i],new[newroles.index(r)]) for i,r in enumerate(oldroles)]
    mapping = {}
    for a,b in pairs:
        if a[0] in mapping and mapping[a[0]] != b[0]:
            return None, 'repeat_conflict'
        mapping[a[0]] = b[0]
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, 'syntax'
    class Change(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) in (int,float) and float(node.value) in mapping:
                x = mapping[float(node.value)]
                node.value = int(x) if x.is_integer() else x
            return node
    tree = Change().visit(tree)
    return ast.unparse(tree), 'ok'

def execute(codes, workers=4):
    return run_chunks(codes, workers=workers, chunk_rows=1000, wall_limit=1., cpu_limit=2.)

def same(a,b):
    return a['status']==b['status'] and (a['status']!='numeric' or math.isclose(a['value'],b['value'], rel_tol=1e-6,abs_tol=1e-6))

def bucket(n):
    return '<60' if n<60 else '60-90' if n<90 else '90-120' if n<120 else '>=120'
