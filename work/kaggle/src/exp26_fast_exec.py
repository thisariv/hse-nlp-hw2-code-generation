"""Reference filter: ~2000 programs per worker task; per-program SIGALRM.
No Torch/NumPy import in spawned workers. A dead chunk is retried per-program.
"""
import multiprocessing as mp,time
from collections import Counter,deque
import exp26_exec as numeric

MEMORY=768*1024**2

def execute_chunk(codes,wall_limit,cpu_limit):
 answers={}
 for c in dict.fromkeys(codes):answers[c]=numeric.execute_task((c,wall_limit,cpu_limit))
 return [answers[c]for c in codes]

def run_chunks(codes,workers=2,chunk_rows=2000,wall_limit=1.,cpu_limit=2.,chunk_timeout=None,
               kill_first_worker_fixture=False,saved_chunks=None,on_chunk=None,progress=None):
 if not codes:return {'rows':[],'summary':{'programs':0,'seconds':0.,'chunk_retries':0,'pool_restarts':0}}
 chunks=[codes[i:i+chunk_rows]for i in range(0,len(codes),chunk_rows)];done=dict(saved_chunks or {});todo=deque(i for i in range(len(chunks))if i not in done);pending={}
 pool=None;old_pids=set();start=time.monotonic();retries=restarts=0;killed=False
 try:
  while todo or pending:
   if pool is None:
    pool=mp.get_context('spawn').Pool(workers,initializer=numeric.worker_init,initargs=(MEMORY,))
    old_pids={p.pid for p in pool._pool}
   while todo and len(pending)<workers:
    i=todo.popleft();pending[i]=(pool.apply_async(execute_chunk,(chunks[i],wall_limit,cpu_limit)),time.monotonic())
   if kill_first_worker_fixture and not killed and pending:
    pool._pool[0].terminate();killed=True
   for i,(future,_)in list(pending.items()):
    if future.ready():
     try:
      row=future.get(timeout=0)
     except Exception:continue  # A failed result follows the per-program fallback below.
     if len(row)!=len(chunks[i]):continue
     if on_chunk:on_chunk(i,row)  # Drive/journal errors must propagate.
     done[i]=row;del pending[i]
     if progress:progress(sum(len(v)for v in done.values()),len(codes),time.monotonic()-start)
   pids={p.pid for p in pool._pool}
   death=pids!=old_pids or any(p.exitcode is not None for p in pool._pool)
   overdue=[i for i,(_,t)in pending.items()if time.monotonic()-t>(chunk_timeout or max(30.,len(chunks[i])*.02+10.))]
   if pending and (death or overdue or any(f.ready()for f,_ in pending.values())):
    retry=list(pending);pool.terminate();pool.join();pool=None;pending.clear();restarts+=1
    # Pool.map never waits for a task lost with its worker. Collateral chunks also
    # get a fresh, bounded per-task pool and preserve their input order.
    for i in retry:
     evaluated=numeric.evaluate_programs(chunks[i],workers=workers,wall_limit=wall_limit,
       cpu_limit=cpu_limit,parent_timeout=max(3.,wall_limit+2.),memory_bytes=MEMORY)
     assert evaluated['summary']['complete'];done[i]=evaluated['rows'];retries+=1
     if on_chunk:on_chunk(i,done[i])
     if progress:progress(sum(len(v)for v in done.values()),len(codes),time.monotonic()-start)
   elif death and not pending:
    pool.terminate();pool.join();pool=None;restarts+=1
   if pending:time.sleep(.005)
 finally:
  if pool is not None:pool.terminate();pool.join()
 rows=[r for i in range(len(chunks))for r in done[i]]
 return {'rows':rows,'summary':{'programs':len(codes),'statuses':dict(Counter(r['status']for r in rows)),
   'seconds':time.monotonic()-start,'chunk_retries':retries,'pool_restarts':restarts,'chunk_rows':chunk_rows,
   'workers':workers,'wall_limit':wall_limit,'cpu_limit':cpu_limit}}
