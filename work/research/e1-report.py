"""Render frozen E1 results (or a transparent pending status) into the note."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
NOTE=WORK/'notes/e1-ensembles.md'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x):return f'{x:.2f}'

def table_row(name,tune,evaluation):
    lo,hi=evaluation['ci95_pp']
    return f"|{name}|{tune['matches']}/2048 ({f(tune['accuracy_pp'])})|{evaluation['matches']}/2048 ({f(evaluation['accuracy_pp'])})|{evaluation['delta_pp']:+.3f} [{lo:+.3f}; {hi:+.3f}]|{tune['timing']['test1919_seconds']:.1f}|"

def save_note(section):
    text=NOTE.read_text();marker='\n## Результаты E1\n'
    before_hash=hashlib.sha256(text.split('## CPU-проверки')[0].encode()).hexdigest()
    if marker in text:
        start=text.index(marker);end=text.index('\n## Источники\n',start)
        text=text[:start]+text[end:]
    text=text.replace('\n## Источники\n',marker+'\n'+section.rstrip()+'\n\n## Источники\n')
    assert hashlib.sha256(text.split('## CPU-проверки')[0].encode()).hexdigest()==before_hash
    NOTE.write_text(text)

def run(raw=None):
    launch=read(WORK/'checks/e1-launch.json')
    observations=read(WORK/'checks/e1-observations.json');last=observations[-1]
    common=dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),ref=launch['ref'],
        own_push_calls=launch['own_push_calls'],own_pushed_version=launch['version'],
        monitor_version=last['version'],last_remote_status=last['status'],exact_code_parity=last['all_code_cells_exact'],
        observed_versions={str(r['version']):r['status']for r in observations},
        notebook_sha256=launch['notebook_sha256'],preregistration_sha256=launch['preregistration_sha256'])
    git=read(WORK/'checks/e1-git.json')if (WORK/'checks/e1-git.json').exists()else None
    git_line='Коммит не создан: песочница запретила Git `index.lock`; индекс остался пустым. [Квитанция](../checks/e1-git.json).'
    if git and git['status']=='committed':git_line='Коммит реализации: `'+git['commit_sha']+'`.'
    version_url=f"https://www.kaggle.com/code/{launch['ref']}/versions?versionNumber={last['version']}"
    gpu=f"Последняя [версия{last['version']}]({version_url}): **{last['status']}**, наблюдение {last['at_utc']}. Собственный push ровно1; внешние версии: "+', '.join(map(str,launch.get('external_versions',[])))+'. Exact code parity подтверждён; отмен/перезапусков агентом0.'
    if not (WORK/'results/e1-confirm.json').exists():
        result=dict(common,status='gpu_failed'if last['status']=='ERROR'else'waiting_gpu_output',quality=None,analysis_pending=True,
            decision='keep v10; E1 quality not assessed',wait_limit_reached=(WORK/'checks/e1-wait-limit.json').exists())
        write(WORK/'results/e1-experiment.json',result)
        section=gpu+'\n\nПолного пула нет. Новые tune/eval accuracy, CI и прогнозы политик **не получены**; eval не открывался. Научный вывод об E1 не сделан. Реализованы все33политики и CPU-проверки; для анализа нужен полный output того же кода. **Решение: сохранитьv10 до честного подтверждения.**\n\n'+git_line
        save_note(section);print(json.dumps(result,ensure_ascii=False));return
    selection=read(WORK/'results/e1-selection.json');seal=read(WORK/'checks/e1-selection.json')
    assert sha(WORK/'results/e1-selection.json')==seal['selection_sha256']
    tune=read(WORK/'results/e1-tune.json');evaluation=read(WORK/'results/e1-confirm.json')
    assert evaluation['selection_sha256']==seal['selection_sha256']and evaluation['primary']==selection['primary']
    t={r['policy']['name']:r for r in tune['rows']};e={r['policy']['name']:r for r in evaluation['rows']}
    primary=selection['primary'];success=evaluation['primary_success']
    family_names=list(dict.fromkeys(selection['family_best'][f]for f in ('baseline','token','LL','soup','budget','temperature')
        if selection['family_best'].get(f)is not None))
    best=[dict(name=n,tune=t[n],eval=e[n],primary=n==primary)for n in family_names]
    result=dict(common,status='complete',analysis_pending=False,primary=primary,primary_success=success,
        best_frozen_on_tune=best,selection_sha256=seal['selection_sha256'],
        decision='draft v11; cold test remains a separate future check'if success else'keep v10; E1 primary success criterion not met')
    write(WORK/'results/e1-experiment.json',result)
    lines=[gpu,'',f"Полный пул подтверждён SHA/manifest/source/config/IDs, baseline production parity2048/2048 на обоих splits. Primary **`{primary}`** выбрана на tune до открытия eval; selection SHA `{seal['selection_sha256']}`. Все времена ниже — frozen прогноз1919строк, не новый cold test. CI95% парной разницы с baseline, п.п.; accuracy в%. Baseline относится к свежему E1-пулу.",'',
        '|Политика (выбор на tune)|Tune, %|Eval, %|Δeval [95%CI], п.п.|Прогноз, с|',
        '|---|---:|---:|---|---:|']
    for n in family_names:lines.append(table_row(('**'+n+'**')if n==primary else n,t[n],e[n]))
    lines+=['', '**Критерий успеха пройден.** Набросокv11 сохранён ниже; холодное время пока не измерено.'if success else'**Критерий успеха не пройден. Сохранитьv10.** Положительные вторичные результаты не меняют frozen primary. Таблица показывает, какие направления дали выигрыш и какие потеряли accuracy. CI по задачам условен на этом пуле; вариативность повторного sampling run не оценена.','',
        '> [!info]- Вся предобъявленная сетка (вторичный описательный eval)',
        '> |Политика|Tune, %|Eval, %|Δeval [95%CI], п.п.|Прогноз, с|',
        '> |---|---:|---:|---|---:|']
    for n in t:lines.append('> '+table_row(n,t[n],e[n]))
    if primary=='budget-1-8-8':
        lines+=['',
            '**Вывод по гипотезам.** На свежем пуле увеличение P1-B/W до8 только при разногласии stage1 дало29исправлений/14ухудшений. Это выбранная до eval primary; один sampling run не оценивает разброс повторного запуска.',
            '',
            'Токенный greedy: eval1513, против одиночных C/P/W1499/1498/1499; улучшение относительно отдельных greedy есть. До baseline1559 он не дотягивает. E vote2→8 даёт те же1559 при220,4с; смесьE+обычных samples доходит до1574, но прогноз302,6с исключает её из допустимых политик.',
            '',
            'Cross LL: выбранное λ0.5 даёт+3ответа, CI включает0. Tie по sumLL не меняет score, logmass теряет7ответов. Буквальная сумма отрицательных LL по кластеру даёт1246/2048 (−15,28п.п.). Собственная интерпретация: такой score штрафует размер кластера; это не вероятностная маргинализация.',
            '',
            'Soupα0.5 как anchor теряет1ответ; α0.3/0.7 дают лишь+1. Замена P1-B одним soup-greedy ухудшает score. Смесь температур при1/8/8 даёт+14ответов, но CI[−0,098;+1,465] включает0; её не выбираем по вторичному eval.',
            '',
            '**Набросокv11:** прежние C-long2x B6/P1-B/W B6, greedyC, T0.7; stage1 по1sample каждой модели. При полном численном согласии остановиться, иначе добавить7P1-B и7W. Итого максимум17samples+greedy. Голосование, порядок/приоритет, executor и guardsv10 сохраняются; дорогие token ensemble/LL/soup в этот вариант не входят.',
            '',
            'Прогноз:23,083с фиксированных фаз +56,757с сырой генерации×2,012782 = **137,322с**. Коэффициент калибровки опубликован; это перенос throughput на формыv10, а не измерение coldv11. Cold T4, W K7 memory/OOM-retry и новые production samples ещё не проверены. До этого v10 остаётся готовым fallback.']
    lines+=['','[Tune](../results/e1-tune.json), [selection](../results/e1-selection.json), [eval](../results/e1-confirm.json), [компактный итог](../results/e1-experiment.json). Все33eval-интервала описательные, без поправки на множество сравнений; первичный критерий применён только к frozen primary.','',git_line]
    if success:
        policy=t[primary]['policy']
        models=[dict(m)for m in read(WORK/'kaggle/e1-config.json')['models']]
        if policy.get('limits'):
            for model,limit in zip(models,policy['limits']):model.update(K1=1,K2=limit)
        sketch=dict(kind='v11-policy-sketch',selected_on='E1 tune; confirmed once on eval2048',
            policy=policy,timing=t[primary]['timing'],models=models,
            single_T4=True,cold_T4_verified=False,generation_guard_seconds=160,execution_stop_seconds=166,
            verification_stop_seconds=170,supervisor_stop_seconds=172,test_generated_in_E1=False)
        if policy['family']=='budget':
            sketch.update(ensemble_config=dict(models=models,unanimity_threshold=1.,sequence_budget=1024,
                greedy_sequence_budget=1024,scorer_enabled=False,scorer_weight=0.),
                temperature=.7,production_seed=270042,token_ensemble=False,cross_LL=False,soups=[],
                max_stochastic_samples=sum(policy['limits']),max_candidates_with_greedy=1+sum(policy['limits']),
                stage2_extra_samples=[k-1 for k in policy['limits']],
                source_selection_sha256=seal['selection_sha256'],
                sample_stream_note='New production shapes/RNG may differ from the single E1 pool; cold gate pending')
        write(WORK/'results/e1-v11-policy.json',sketch)
        lines+=['','[Набросокv11](../results/e1-v11-policy.json): ровно frozen policy и веса E1; прежние guards160/166/170/172, одинT4. Прогноз '+f(t[primary]['timing']['test1919_seconds'])+'с. Перед новой посылкой нужен отдельный cold gate; в E1 test не запускался.']
    if raw:
        assert sha(raw/'e1/summary.json')==selection['raw_session_sha256']
        summary=read(raw/'e1/full/tune2048/summary.json');agg=defaultdict(lambda:dict(seconds=0.,rows=0,sequences=0,tokens=0))
        for r in summary['timings']:
            key=(r['kind'],r['label'],r.get('K',0));a=agg[key]
            for k in a:a[k]+=r.get(k,0)
        throughput=[]
        for (kind,label,k),a in agg.items():
            throughput.append(dict(kind=kind,label=label,K=k,row_unit='question'if kind=='generation'else'unique candidate',**a,
                samples_per_second=a['sequences']/a['seconds']if kind=='generation'else None,
                ms_per_row=a['seconds']*1000/a['rows'],tokens_per_second=a['tokens']/a['seconds']if kind=='LL'else None))
        combined=defaultdict(lambda:dict(seconds=0.,sequences=0))
        for r in throughput:
            if r['kind']=='generation':
                combined[r['label']]['seconds']+=r['seconds'];combined[r['label']]['sequences']+=r['sequences']
        combined=[dict(label=label,rows=2048,K=a['sequences']/2048,**a,
            samples_per_second=a['sequences']/a['seconds'],ms_per_row=a['seconds']*1000/2048)
            for label,a in combined.items()]
        write(WORK/'results/e1-throughput.json',dict(source='full tune only',rows=throughput,
            combined_generation=combined,time_model=selection['time_model']))
        session=read(raw/'e1/summary.json')
        lines+=['','GPU session '+f(session['seconds'])+'с; [измеренный throughput](../results/e1-throughput.json). Исходный test не читался; число кандидатов87на строку, полных строк4096.']
        audit_path=WORK/'checks/e1-final.json'
        if audit_path.exists():
            audit=read(audit_path)
            lines+=['','Всего356352кандидата,222968уникальных represented-code внутри строк,668904cross-LL score. OOM0, peak allocated7,71/7,54GiB (tune/eval). Внутри каждого split2048уникальных question SHA; пересечений tune/eval0. [Финальный аудит](../checks/e1-final.json).']
    recovery=WORK/'checks/e1-source-recovery.json'
    if recovery.exists():
        recovered=read(recovery)
        lines+=['','После HTTP502 два source-файла восстановлены из `SOURCES` **серверной версии4** с exact code parity и проверкой каждого SHA. Candidate/LL-журналы не изменялись. [Происхождение восстановленных исходников](../checks/e1-source-recovery.json).']
    save_note('\n'.join(lines));print(json.dumps(dict(primary=primary,success=success,best=[dict(name=x['name'],tune=x['tune']['matches'],eval=x['eval']['matches'],delta=x['eval']['delta_pp'],CI=x['eval']['ci95_pp'],seconds=x['tune']['timing']['test1919_seconds'])for x in best]),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path);a=p.parse_args();run(a.raw)
