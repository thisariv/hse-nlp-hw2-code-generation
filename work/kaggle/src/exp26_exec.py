"""Bounded numeric-program execution for this public dataset; no arbitrary imports/I/O."""
import ast
import builtins
import math
import multiprocessing as mp
import os
import resource
import signal
import sys
import time
from collections import Counter, deque
from types import SimpleNamespace

SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    'abs','all','any','bool','dict','divmod','enumerate','float','int','len','list',
    'max','min','pow','range','reversed','round','set','sorted','str','sum','tuple','zip',
    'Exception','ValueError','TypeError','ZeroDivisionError','OverflowError','ArithmeticError')}
FORBIDDEN_NAMES = {'open','input','eval','exec','compile','globals','locals','vars','getattr',
    'setattr','delattr','dir','help','breakpoint','exit','quit','os','sys','subprocess',
    'socket','ctypes','importlib','builtins','resource','signal','pickle','marshal',
    'memoryview','type','object','super'}
FORBIDDEN_ATTRS = {'mro','subclasses','format_map','gi_frame','f_globals','f_locals',
    'tb_frame','func_globals','code','modules','system','popen','load','loads','dump','dumps'}
SAFE_BUILTINS['print']=lambda *args,**kwargs:None  # no program output or file handle
ALLOWED_IMPORTS = {'math','sympy'}
SYM_NAMES = {'Symbol','symbols','Rational','Integer','Float','Eq','solve','sqrt','simplify'}
MAX_CODE_BYTES=100000
MAX_AST_NODES=5000

class ProgramTimeout(Exception): pass

def filter_program(code):
    if not isinstance(code,str) or len(code.encode('utf-8'))>MAX_CODE_BYTES:
        return None,'blocked_size'
    try: tree=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError): return None,'syntax_error'
    nodes=list(ast.walk(tree))
    if len(nodes)>MAX_AST_NODES:return None,'blocked_size'
    for n in nodes:
        if isinstance(n,ast.Name) and (n.id.startswith('__') or n.id in FORBIDDEN_NAMES):
            return None,'blocked_name'
        if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or n.attr in FORBIDDEN_ATTRS):
            return None,'blocked_attribute'
        if isinstance(n,(ast.ClassDef,ast.AsyncFunctionDef,ast.Await,ast.Yield,ast.YieldFrom)):
            return None,'blocked_construct'
        if isinstance(n,(ast.FunctionDef,ast.arg)):
            name=n.name if isinstance(n,ast.FunctionDef) else n.arg
            if name.startswith('__') or name in FORBIDDEN_NAMES:return None,'blocked_name'
        if isinstance(n,ast.Import):
            if any(a.name not in ALLOWED_IMPORTS or (a.asname and a.asname.startswith('_'))for a in n.names):
                return None,'blocked_import'
        if isinstance(n,ast.ImportFrom):
            if n.level or n.module not in ALLOWED_IMPORTS:return None,'blocked_import'
            if any(a.name.startswith('_') or a.name=='*'or (a.asname and a.asname.startswith('_'))for a in n.names):
                return None,'blocked_import'
            if n.module=='sympy' and any(a.name not in SYM_NAMES for a in n.names):return None,'blocked_import'
    return tree,None

def safe_sympy():
    import sympy
    def no_strings(x):
        if isinstance(x,str):raise ValueError('String expressions forbidden')
        if isinstance(x,(tuple,list)):
            for y in x:no_strings(y)
        if isinstance(x,dict):
            for k,v in x.items():no_strings(k);no_strings(v)
    def wrapped(function):
        def call(*args,**kwargs):
            no_strings(args);no_strings(list(kwargs.values()))
            return function(*args,**kwargs)
        return call
    def symbol(name,*args,**kwargs):
        if not isinstance(name,str) or not name or not name.replace('_','').isalnum():raise ValueError('Bad symbol')
        return sympy.Symbol(name,*args,**kwargs)
    def symbols(names,*args,**kwargs):
        if not isinstance(names,str)or any(not(c.isalnum()or c in ' _,:')for c in names):raise ValueError('Bad symbols')
        return sympy.symbols(names,*args,**kwargs)
    return SimpleNamespace(Symbol=symbol,symbols=symbols,
        **{n:wrapped(getattr(sympy,n))for n in SYM_NAMES-{'Symbol','symbols'}})

_MODULES={}
def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
    if level or name not in ALLOWED_IMPORTS:raise ImportError('Blocked import')
    if name not in _MODULES:
        _MODULES[name]=SimpleNamespace(**{n:getattr(math,n)for n in dir(math)if not n.startswith('_')})if name=='math'else safe_sympy()
    return _MODULES[name]

def worker_init(memory_bytes):
    # Spawned workers import no Torch. No GPU state or model is inherited.
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if sys.platform.startswith('linux') and hasattr(resource,'RLIMIT_AS'):
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        value=memory_bytes if hard==resource.RLIM_INFINITY else min(memory_bytes,hard)
        resource.setrlimit(resource.RLIMIT_AS,(value,value))
    signal.signal(signal.SIGALRM,lambda signum,frame:(_ for _ in ()).throw(ProgramTimeout()))

def execute_task(payload):
    code,wall_limit,cpu_limit=payload
    started=time.monotonic()
    tree,status=filter_program(code)
    if status:return {'status':status}
    # RLIMIT_CPU is cumulative per process. Set a new SOFT deadline for EACH task;
    # retain the hard limit so the soft deadline can increase for reused workers.
    used=resource.getrusage(resource.RUSAGE_SELF)
    soft=max(1,math.ceil(used.ru_utime+used.ru_stime)+int(math.ceil(cpu_limit)))
    _,hard=resource.getrlimit(resource.RLIMIT_CPU)
    if hard!=resource.RLIM_INFINITY:soft=min(soft,hard)
    resource.setrlimit(resource.RLIMIT_CPU,(soft,hard))
    signal.setitimer(signal.ITIMER_REAL,wall_limit)
    try:
        env={'__builtins__':dict(SAFE_BUILTINS,__import__=safe_import)}
        exec(compile(tree,'<bounded-dataset-program>','exec'),env,env)
        solve=env.get('solve')
        if not callable(solve):return {'status':'missing_solve'}
        value=solve()
        if isinstance(value,bool):value=int(value)
        if not isinstance(value,(int,float)):
            # SymPy real scalars may be used; containers/strings/complex refused.
            if type(value).__module__.startswith('sympy') and value.is_real is True:value=float(value)
            else:return {'status':'non_number'}
        value=float(value)
        if not math.isfinite(value):return {'status':'non_finite'}
        return {'status':'numeric','value':value}
    except ProgramTimeout:return {'status':'timeout'}
    except BaseException as e:return {'status':'execution_error','error_type':type(e).__name__}
    finally:signal.setitimer(signal.ITIMER_REAL,0)

def create_pool(workers,memory_bytes):
    return mp.get_context('spawn').Pool(workers,initializer=worker_init,
        initargs=(memory_bytes,),maxtasksperchild=256)

def evaluate_programs(codes,workers=2,wall_limit=1.0,cpu_limit=2.0,
                      parent_timeout=3.0,memory_bytes=768*1024**2,deadline=None,progress=None):
    """Bounded inflight tasks; every async result has a parent deadline.
    A killed worker loses its task even if Pool replaces it: terminate/recreate
    the WHOLE pool, retain completed results and retry collateral pending tasks.
    Pool.map is deliberately never used. Output order and duplicates preserved.
    """
    if parent_timeout<=wall_limit:raise ValueError('Parent deadline must exceed program deadline')
    unique=list(dict.fromkeys(codes));todo=deque(range(len(unique)));answers={};pending={}
    pool=None;restarts=0;start=time.monotonic();cancelled=False;last_progress=-1
    try:
        while todo or pending:
            if deadline is not None and time.monotonic()>=deadline:
                cancelled=True;break
            if pool is None:pool=create_pool(workers,memory_bytes)
            while todo and len(pending)<workers:
                i=todo.popleft();tree,status=filter_program(unique[i])
                if status:answers[i]={'status':status};continue
                pending[i]=(pool.apply_async(execute_task,((unique[i],wall_limit,cpu_limit),)),time.monotonic())
            for i,(future,submitted)in list(pending.items()):
                if future.ready():
                    try:answers[i]=future.get(timeout=0)
                    except Exception as e:answers[i]={'status':'worker_error','error_type':type(e).__name__}
                    del pending[i]
            timed=[i for i,(_,submitted)in pending.items()if time.monotonic()-submitted>=parent_timeout]
            if timed:
                for i in timed:answers[i]={'status':'timeout_or_worker_exit'}
                retry=[i for i in pending if i not in timed]
                pool.terminate();pool.join();pool=None;restarts+=1;pending.clear()
                todo.extendleft(reversed(retry))
            if progress and len(answers)!=last_progress and (len(answers)%1000==0 or len(answers)==len(unique)):
                progress(len(answers),len(unique));last_progress=len(answers)
            if pending:time.sleep(.001)
        if cancelled:
            for i in todo:answers[i]={'status':'budget_not_evaluated'}
            for i in pending:answers[i]={'status':'budget_not_evaluated'}
    finally:
        if pool is not None:pool.terminate();pool.join()
    by_code={c:answers.get(i,{'status':'budget_not_evaluated'})for i,c in enumerate(unique)}
    rows=[dict(by_code[c])for c in codes]
    return {'rows':rows,'summary':{'programs':len(codes),'unique_programs':len(unique),
        'statuses':dict(Counter(r['status']for r in rows)),'pool_restarts':restarts,
        'seconds':time.monotonic()-start,'complete':not cancelled}}

def compare_exec(reference_results,generation_results,rel_tol=1e-6,abs_tol=1e-6):
    refs=reference_results['rows']if isinstance(reference_results,dict)else reference_results
    gens=generation_results['rows']if isinstance(generation_results,dict)else generation_results
    if len(refs)!=len(gens):raise ValueError('Order/length mismatch')
    eligible=[i for i,r in enumerate(refs)if r['status']=='numeric']
    matches=[i for i in eligible if gens[i]['status']=='numeric'and math.isclose(
        gens[i]['value'],refs[i]['value'],rel_tol=rel_tol,abs_tol=abs_tol)]
    return {'denominator':len(eligible),'matches':len(matches),
        'accuracy':len(matches)/len(eligible)if eligible else None,
        'eligible_indices':eligible,'match_indices':matches,
        'reference_statuses':dict(Counter(r['status']for r in refs)),
        'generation_statuses':dict(Counter(r['status']for r in gens)),
        'rel_tol':rel_tol,'abs_tol':abs_tol}
