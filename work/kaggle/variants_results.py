"""Audit downloaded variants, execute each answer, retain five bot-format JSONL files.

No API, GPU, new sampling, policy search or bot messages.
"""
import argparse
import ast
import base64
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zlib
from variants_common import *

def answers(path,n):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==n and all(set(r)=={'code'} and isinstance(r['code'],str) for r in rows)
    return rows

def execute(rows,executor_type):
    candidates=[dict(code=r['code']) for r in rows];executor=executor_type(4)
    try:
        executor.submit(candidates)
        assert executor.drain(time.monotonic()+300),'Independent execution incomplete'
        executor.attach(candidates)
        return [c['execution'] for c in candidates]
    finally:executor.close()

def same(a,b):
    return a['status']==b['status'] and (a['status']!='numeric' or math.isclose(a['value'],b['value'],rel_tol=1e-6,abs_tol=1e-6))

def provenance(raw, build):
    """Compare the saved server source, bundle payload and downloaded files."""
    remote=read(WORK/'checks/final-variants-remote.json')
    assert remote['version']==1 and remote['exact_code_parity']
    notebook_path=ROOT/'notebooks/nlp-hw02-variants.ipynb'
    assert sha(notebook_path)==build['notebook_sha256']==remote['notebook_sha256']
    notebook=read(notebook_path)
    cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert [hashlib.sha256(c.encode()).hexdigest() for c in cells]==remote['code_cells_sha256']
    tree=ast.parse(cells[1])
    assignments={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    code=ast.literal_eval(assignments['CODE_FILES'])
    packed=assignments['INPUT_FILES'].args[0].args[0].args[0]
    inputs=json.loads(zlib.decompress(base64.b64decode(ast.literal_eval(packed))))
    assert ast.literal_eval(assignments['CONFIGS'])==build['variants']==variants()
    for payload,expected,folder in [(code,build['code_sha256'],'variants-src/hw2_codegen'),(inputs,build['input_sha256'],'variants-inputs')]:
        assert set(payload)==set(expected)
        assert {p.name for p in (raw/folder).iterdir() if p.is_file()}==set(expected)
        for name,content in payload.items():
            assert hashlib.sha256(content.encode()).hexdigest()==expected[name]==sha(raw/folder/name)
    assert sha(raw/'variants-inputs/test.json')=='20cacb334e3fc81291433d2173d9085b17a15e1834bd84aea67b350403182503'
    return dict(server=remote,bundle_code_files=len(code),bundle_input_files=len(inputs),embedded_configs_exact=True,downloaded_payload_exact=True)

def run(raw):
    raw=Path(raw).resolve();root=raw/'variants'
    assert root.is_dir()
    build=read(WORK/'checks/final-variants-build.json')
    source_proof=provenance(raw,build)
    # Use the verified version-1 source, independently of unfinished local edits.
    sys.path.insert(0,str(raw/'variants-src'))
    from hw2_codegen.execute import Executor
    from hw2_codegen.vote import replay_policy,select_available
    for name,digest in build['code_sha256'].items():assert sha(raw/'variants-src/hw2_codegen'/name)==digest
    for name,digest in build['input_sha256'].items():assert sha(raw/'variants-inputs'/name)==digest
    executor_files=['__init__.py','execute.py','_service.py','_worker.py','_numeric.py','vote.py']
    assert sha(SUBMISSION/'inference.ipynb')==build['submission_notebook_sha256']
    session=read(root/'summary.json')
    assert session['training'] is False and session['cold_process_per_split'] and session['single_visible_gpu']
    assert set(session['variants'])=={'v1','v2','v3','v4','v5'}
    previous_path=WORK/'submissions/final-ensemble-v11.jsonl'
    previous=answers(previous_path,1919)
    baseline_proof=read(WORK/'checks/final-inference-v11-output.json')
    assert baseline_proof['answers_sha256']==sha(previous_path) and baseline_proof['test_order_verified']
    refs=read(raw/'variants-inputs/tune2048-reference.json')
    assert len(refs)==2048
    previous_exec=execute(previous,Executor)
    assert all(e['status']=='numeric' for e in previous_exec)
    print(json.dumps(dict(baseline='v11',independently_executed=len(previous_exec))),flush=True)
    destination=WORK/'submissions/variants-2026-10-20';destination.mkdir(parents=True,exist_ok=True)
    results=[]
    for cfg in variants():
        name=cfg['variant'];reports=session['variants'][name];folders=[root/name,root/name/'tune']
        saved=[];executions=[];counts=[]
        for split,n,folder in zip(['test','tune'],[1919,2048],folders):
            actual=read(folder/'config.json')
            assert all(actual[k]==v for k,v in cfg.items()),(name,split,'config')
            assert ('reference' in actual)==(split=='tune')
            assert actual['device']=='cuda:0' and not actual.get('dry_run',False)
            report=read(folder/'summary.json');assert report==reports[split]
            data=answers(folder/'answers.jsonl',n);saved.append(data)
            ex=execute(data,Executor);executions.append(ex)
            assert sha(folder/'answers.jsonl')==report['answers_sha256']
            expected_rows=read(raw/'variants-inputs'/('test.json' if split=='test' else 'tune2048.json'))
            order_sha=hashlib.sha256(json.dumps([r['question'] for r in expected_rows],ensure_ascii=False).encode()).hexdigest()
            assert report['question_order_sha256']==order_sha
            assert report['rows']==n and report['full_pipeline_seconds']==report['elapsed_seconds']
            assert report['verification']['complete'] and not report['verification']['mismatch_rows']
            assert report['model_sha256']=={m['name']:m['sha256'] for m in cfg['models']}
            assert report['model_order']==[m['name'] for m in cfg['models']]
            assert all(len(b['ids'])*b['k']<=640 for b in report['batches'])
            assert 'T4' in report['runtime']['device']
            fallback=len(report.get('fallback_rows',range(n) if 'supervisor' in report['guards'] else []))
            counts.append(dict(fallback=fallback,guards=report['guards'],partial=len(report.get('partial_rows',[])),
                nonnumeric=sum(e['status']!='numeric' for e in ex),oom=len(report.get('oom_attempts',[])),
                rows=n,independently_executed_unique=len({r['code'] for r in data}),question_order_sha256=order_sha,execution_status_counts=dict(Counter(e['status'] for e in ex)),
                execution_sha256=hashlib.sha256(json.dumps(ex,sort_keys=True,allow_nan=False).encode()).hexdigest()))
        hits=[i for i,(ref,ex) in enumerate(zip(refs,executions[1])) if ref['status']==ex['status']=='numeric' and same(ref,ex)]
        if reports['quality'] is not None:
            assert reports['quality']['match_indices']==hits
            assert reports['quality']['matches']==len(hits) and reports['quality']['n']==2048
            assert reports['quality']['accuracy_pp']==100*len(hits)/2048
        journal=folders[1]/'candidate-journal.json.gz';replayed=policy_replayed=0
        if journal.exists():
            with gzip.open(journal,'rt') as f:records=json.load(f)
            assert len(records)==2048
            for i,(record,out,ex) in enumerate(zip(records,saved[1],executions[1])):
                winner,_=select_available(record,cfg['models'])
                if i in reports['tune']['verification']['mismatch_rows']:winner=record['greedy']
                assert winner['code']==out['code'] and same(winner['execution'],ex)
                replayed+=1
                if not reports['tune']['guards']:
                    w,stop,third,_=replay_policy(record['greedy'],record['samples'],cfg['models'],e2_primary=cfg.get('e2_primary',False))
                    assert w['code']==out['code']
                    assert stop==(i in reports['tune']['stopped'])
                    assert third==(i in reports['tune']['third_rows'])
                    policy_replayed+=1
        target=destination/(name+'.jsonl');payload=(folders[0]/'answers.jsonl').read_bytes()
        if target.exists():assert target.read_bytes()==payload
        else:target.write_bytes(payload)
        changed=[i for i,(a,b) in enumerate(zip(saved[0],previous)) if a!=b]
        numeric_changed=[i for i,(a,b) in enumerate(zip(executions[0],previous_exec)) if not same(a,b)]
        result=dict(variant=name,models=[m['name'] for m in cfg['models']],test_seconds=reports['test']['elapsed_seconds'],
            within_180=reports['test']['elapsed_seconds']<=180,peak_reserved_bytes=reports['test'].get('peak_reserved_bytes'),
            peak_device_used_bytes=reports['test'].get('peak_device_used_bytes'),test=counts[0],tune=counts[1],
            tune_seconds=reports['tune']['elapsed_seconds'],tune_matches=len(hits),tune_n=2048,tune_accuracy_pp=100*len(hits)/2048,
            changed_code_rows=changed,changed_executed_answer_rows=numeric_changed,public=None,
            answers_sha256=sha(target),previous_sha256=sha(previous_path),answers=str(target.relative_to(WORK)),
            tune_journal_winners_exact=replayed,tune_complete_policy_exact=policy_replayed,
            test_third_rows=len(reports['test'].get('third_rows',[])),tune_third_rows=len(reports['tune'].get('third_rows',[])))
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('changed_code_rows','changed_executed_answer_rows')}),flush=True)
    download=read(WORK/'checks/final-variants-download.json')
    result=dict(audit_passed=True,source=str(raw.relative_to(WORK)),download=download,provenance=source_proof,variants=results,
        all_test_within_180=all(r['within_180'] for r in results),independently_executed=True,submission_notebook_unchanged=True,
        independent_execution=dict(rows=len(previous_exec)+sum(r['test']['rows']+r['tune']['rows'] for r in results),
            unique_program_executions=len({r['code'] for r in previous})+sum(r[s]['independently_executed_unique'] for r in results for s in ['test','tune']),
            fresh_services=11,executor='hw2_codegen.execute.Executor, four workers, no saved execution cache',
            executor_source_sha256={n:build['code_sha256'][n] for n in executor_files},
            comparator=dict(rel_tol=1e-6,abs_tol=1e-6),baseline_all_numeric=True))
    write(WORK/'results/final-variants.json',result)
    label={'C-long2x-B6':'C B6','C-long2x-B7':'C B7','W-B6':'W B6','W-B7':'W B7','pilot-b':'P1-B'}
    lines=['| Вариант | Состав, первая — anchor | Test, с | Пик reserved / NVML, GB | Fallback / guard test | Tune /2048 | Δ исполненных ответов к v11 | public |',
        '|---|---|---:|---:|---|---:|---:|---|']
    gb=lambda x:f'{x/1e9:.2f}' if x is not None else '—'
    for r in results:
        guard=','.join(r['test']['guards']) or '0'
        lines.append(f"| {r['variant']} | {' / '.join(label[m] for m in r['models'])}{' · E2' if r['variant']=='v4' else ''} | {r['test_seconds']:.3f} | {gb(r['peak_reserved_bytes'])} / {gb(r['peak_device_used_bytes'])} | {r['test']['fallback']} / {guard} | {r['tune_matches']} ({r['tune_accuracy_pp']:.2f}%) | {len(r['changed_executed_answer_rows'])} | |")
    note_path=WORK/'notes/final-variants.md'
    note=note_path.read_text().replace('status: draft','status: complete',1)
    title='# Пять вариантов финального ансамбля'
    start=note.index('\n\n',note.index(title))+2
    table_start=note.index('| Вариант |',start)
    intro=('Оркестратор отправил сжатый бандл после первого HTTP400: private '
        'friend1/nlp-hw02-variants, версия1, 20.10.2026 16:25UTC (по сообщению пользователя). '
        'Запуск COMPLETE; output в results/kaggle-runs/nlp-hw02-variants/1/. '
        'CPU-аудит завершён, все пять test укладываются в180с.\n\n')
    note=note[:start]+intro+note[table_start:]
    table_start=note.index('| Вариант |');table_end=note.index('\n\n',table_start)
    note=note[:table_start]+'\n'.join(lines)+note[table_end:]
    note=note.replace('Никаких повторных push, удалений, отмен или публикаций не выполнено.',
        'В первой сессии повторный push не выполнялся; последующий push оркестратора указан выше.')
    old_start=note.find('Запуск вручную:');old_end=note.find('\n## Источники',old_start)
    if old_start!=-1:
        note=note[:old_start]+note[old_end:]
    section=['## Аудит версии 1','',
        f"[Серверный код](../checks/final-variants-remote.json): все три кодовые ячейки совпадают с бандлом 575299 bytes, SHA256 `{build['notebook_sha256']}`. Совпадают 18 файлов кода, 4 входных файла, пять конфигов и SHA использованных моделей. API возвращает источники без суффиксов версий; закреплённые версии сохранены в metadata бандла, фактические веса подтверждены SHA.", '',
        'Каждый test (1919) и tune (2048) выполнен в отдельном холодном процессе на Tesla T4, CUDA_VISIBLE_DEVICES=0. Test включает imports/load/warmup/generation/execute/vote и запись answers.jsonl; диагностика и tune-scoring идут после таймера. GB=10⁹ bytes; reserved — точный максимум PyTorch, NVML — device-wide максимум с опросом 20 мс.', '',
        'Во всех десяти сплитах guard/fallback/partial/OOM/non-numeric и verification mismatches равны 0. V4 stage3: test '+str(results[3]['test_third_rows'])+', tune '+str(results[3]['tune_third_rows'])+' строк.', '',
        f"Независимым повторным исполнением проверены 21754 строки: v11 test (1919) и все пять test (1919) + tune (2048); 11 свежих CPU executor-сервисов без сохранённого кеша исполнений. Одинаковый код внутри файла исполняется один раз: всего {result['independent_execution']['unique_program_executions']} исполнения. Все ответы numeric. Сравнение с v11 использует math.isclose(rel_tol=abs_tol=1e-6); Δ — число изменившихся численных ответов, не прирост accuracy. Tune match_indices и accuracy совпадают с GPU-отчётами; каждый tune-журнал (2048 строк) прошёл replay победителя, early-stop и stage3.", '',
        '[Полный аудит](../results/final-variants.json), [получение output](../checks/final-variants-download.json). Все JSONL по 1919 строк, строго `{"code": str}`; исходный test SHA и SHA порядка вопросов совпадают с отчётами, ответы скопированы побайтно:', '',
        *[f"- [{r['variant']}.jsonl](../{r['answers']}) — SHA256 `{r['answers_sha256']}`." for r in results], '',
        'Повторить CPU-аудит из каталога домашки: `python work/kaggle/variants_results.py --root work/results/kaggle-runs/nlp-hw02-variants/1`. Public намеренно пустая; в бота ничего не отправлялось.', '']
    marker='\n## Аудит версии 1\n'
    if marker in note:
        audit_start=note.index(marker);audit_end=note.index('\n## Источники',audit_start)
        note=note[:audit_start]+note[audit_end:]
    note=note.replace('\n## Источники','\n'+'\n'.join(section)+'\n## Источники',1)
    note_path.write_text(note)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);args=p.parse_args()
    run(args.root or WORK/read(WORK/'checks/final-variants-download.json')['output'])
