"""Torch-free execution initializer: exact per-program wall3s/CPU2s timers."""
import signal
import exp26_exec as N

ORIGINAL_INIT=N.worker_init
ORIGINAL_TASK=N.execute_task

def worker_init(memory_bytes):
    ORIGINAL_INIT(memory_bytes)
    N.execute_task=execute_task

def execute_task(payload):
    code,wall_limit,cpu_limit=payload;reason=[None]
    def timeout(kind):
        def handler(signum,frame):
            reason[0]=kind;raise N.ProgramTimeout()
        return handler
    signal.signal(signal.SIGALRM,timeout('wall'))
    # ITIMER_PROF measures user+system CPU per program; RLIMIT_CPU remains a backup.
    signal.signal(signal.SIGPROF,timeout('cpu'));signal.setitimer(signal.ITIMER_PROF,cpu_limit)
    try:
        result=ORIGINAL_TASK((code,wall_limit,cpu_limit))
        if reason[0]:result['timeout_limit']=reason[0]
        return result
    except N.ProgramTimeout:return {'status':'timeout','timeout_limit':reason[0] or 'cpu'}
    finally:signal.setitimer(signal.ITIMER_PROF,0.)

def install():N.worker_init=worker_init
