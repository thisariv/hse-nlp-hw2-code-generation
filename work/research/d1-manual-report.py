"""Render the fixed 30-row CPU replay with explicit, source-level judgements."""
import json
import sys
from collections import Counter
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'kaggle/src'))
from kaggle_io import sha,write

# Judgements by the assistant from the question and programs, not duplicate labels.
# The sample is fixed before inspecting questions: A, seed 2026101501, SRS n=30.
JUDGEMENTS=[
 (56340,'malformed','Условия противоречат друг другу: из возраста Danny15 получается Ellie14, сумма через4 года37, а не43. По условию суммы получилось бы Ellie20. Единственного ответа нет; модель19 использует лишь второе условие, reference15 не обоснован.'),
 (539334,'model_correct','С17:12 до23:00 проходит5ч48мин =5,8ч. Reference игнорирует12 минут.'),
 (833875,'model_correct','При обычном прочтении exchanges как «отдала» остаётся (50−20+6)·3/4=27. Reference вместо оставшихся30 берёт переданные20. Формулировка обмена не задаёт встречную передачу; это условный вердикт, отдельно учтён в консервативной оценке.'),
 (1382003,'malformed','Начало условия потеряно: нет расстояния в понедельник и полного недельного расписания. Reference подставляет14, модель10; оба числа взяты вне условия.'),
 (1624384,'both_wrong','Буквально5 комплектов за выходные:5·40/60=10/3ч. Если имелось в виду5 в каждый день, получается20/3ч модели. Reference8ч не соответствует ни одному прочтению: добавляет ещё80мин. Модель не подтверждена буквальным условием.'),
 (1780108,'both_wrong','Денег20·5+40=140; можно купить floor(140/15)=9 целых игр. Reference9⅓ не учитывает целочисленность; модель вычитает стоимость купонов из наличных и даёт−4.'),
 (1919857,'model_correct','Остаются порции для14−8=6 человек. Reference (14/8)−14 не имеет нужного смысла.'),
 (1929301,'rounding','Без возвращения вероятность6/10·5/9=1/3. Reference вычисляет ту же формулу и намеренно округляет до4 знаков. Это численное расхождение допуска1e−6, а не содержательная ошибка решения; не считаю реальной ошибкой reference.'),
 (2038200,'reference_correct','В каждую сторону5·7 передач: всего70 конфет. Модели складывают число участников и дают12.'),
 (2119680,'ambiguous','Four times more допускает обычное «в4 раза» (4 бутылки) и буквальное «на400% больше» (3,2). Reference использует второе, модели первое; без соглашения о языке нельзя объявлять reference ошибкой.'),
 (2219509,'reference_correct','Для одной дистанции время Jack вдвое больше Sarah: t+t/2=45, t=30мин. Модели возвращают время Sarah15мин.'),
 (2356451,'both_wrong','По Герону полупериметр15, площадь sqrt(15·7·5·3)=15sqrt(7). Объём225sqrt(7)≈595,294. Reference25312,5 и модели225 используют неверную площадь.'),
 (2419026,'model_correct','Работа10 человеко-часов; троим нужно10/3ч. Reference умножает время на число работников вместо обратной пропорции.'),
 (2591265,'model_correct','6x+5=56, x=8,5. Аннотация ->int не требует обрезать математический возраст; int в reference теряет полгода.'),
 (3809793,'model_correct','Цветущих роз30·6/2=90. Reference смешивает количество кустов и цветков и получает45.'),
 (3866446,'reference_correct','Если маффинов x, круассанов2x, то3x=45 и x=15. Модели переставляют категории и отвечают30. Необычные tuple-переменные reference не мешают фактическому numeric15.'),
 (4020725,'reference_correct','У четырёхугольника сумма360°, максимальный угол360·7/21=120°. Модели используют180° и дают60°.'),
 (4265800,'model_correct','За неделю150мин; при7мин/км расстояние150/7≈21,4286км. Reference смешивает часы и минуты и дополнительно умножает на5.'),
 (4540250,'ambiguous','На250 долларов можно купить13 тюков, до желаемых15 не хватает2. Reference2 отвечает на разумное исправленное «сколько не хватает»; буквальное «сколько ещё может купить» не задаёт исходной покупки. Модель−2 неверна; reference удалять по согласию моделей нельзя.'),
 (5576677,'malformed','Стоимость камеры и гарантия не определяют ёмкость карты памяти. Reference выдумывает35/50 и деление на10, модели подставляют800.'),
 (5991325,'malformed','Сэндвичи Riley/Max не содержат сведений о книгах Ada. Ни7,875, ни4,5 не следуют из условия.'),
 (6124558,'malformed','Размер обуви Emma не дан и не выводится из фокуса и аплодисментов. Обе стороны выдумывают размер.'),
 (6563351,'model_correct','При отсутствии дополнительного времени на дорогу начать нужно в14:00, то есть2pm:18−3−1=14. Модели2 используют12-часовую запись. Reference смешивает60мин и18часов и даёт−60.'),
 (7239421,'model_correct','Для двух половин расстояния по75км:150/(75/60+75/40)=48км/ч. Reference неверно переводит расстояния во время.'),
 (7327167,'both_wrong','Количество эквивалентных по площади пицц (14/10)^2=1,96, либо2 целых при заказе не меньше исходной площади. Reference1 и модели1,4 неверны.'),
 (7617448,'both_wrong','Взнос отца удваивает месячные276:2·276·12·4=26496 при нулевом неуказанном начальном балансе. Reference выдумывает ещё3000+7000; модели забывают взнос отца.'),
 (7854986,'malformed','Одинаковая дистанция каждый день противоречит прибавке40. Даже если оставить арифметическую прогрессию, первый член350/7−3·40=−70 миль: физически невозможен. Reference−30 и модели−190 не исправляют условие.'),
 (8441972,'malformed','Первое условие даёт большой стеллаж60 книг; после удаления10 из малого разность40, а не50. Reference70 использует только разность, модели110 добавляют50 к исходным60. Все условия несовместимы.'),
 (9213450,'malformed','Не дана модель расхода энергии или соответствующий коэффициент. Вес и время не определяют единственное число калорий. Reference вводит эмпирическую формулу, модели100ккал/ч; оба добавляют внешние предположения.'),
 (9578661,'malformed','В условии Lily и фото; спрашивается расстояние Natalie без чисел и связи с рассказом. Reference6 и модели15 выдуманы.'),
]


def main():
    raw=WORK/'results/d1-labels/manual-review-data.json'
    replay=json.loads(raw.read_text());rows=replay['rows']
    assert [r['row_id'] for r in rows]==[j[0] for j in JUDGEMENTS]
    assert all(v['matches_D1_answer'] and v['EOS'] for r in rows for v in r['cpu_replay'].values())
    assert all(r['reference_reexecution']['status']=='numeric' for r in rows)
    counts=dict(Counter(j[1] for j in JUDGEMENTS))
    summary=dict(seed=2026101501,sampling='SRS without replacement from all 38644 A candidates',n=30,
                 counts=counts,reference_errors=14,conservative_reference_errors=13,
                 models_correct=9,conservative_models_correct=8,CPU_replay_D1_answer_matches=60,
                 reference_reexecutions=30,original_T4_code_available=False,
                 replay_sha256=sha(raw),
                 caveat='Assistant semantic proxy; one conditional exchange reading; no gold adjudication; no full-train prevalence estimate',
                 judgements=[dict(row_id=i,verdict=k,reason=t) for i,k,t in JUDGEMENTS])
    write(WORK/'results/d1-manual-review.json',summary)
    text='''---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-15
updated: 2026-10-15
sources:
  - nlp-hw02-d1-results
aliases: []
---

# D1: ручная проверка 30 кандидатов

Собственный разбор условия и кода; **прокси**, не независимая эталонная разметка.
Выборка:30 случайных без возвращения из всех38 644 кандидатов A,
NumPy default_rng, seed2026101501; row ids отсортированы только для показа.
Условия не подбирались по содержанию или доступности дублей. В выборке нет строк C.

**Reference ошибочен в14/30 (46,7%)**, включая одно обычное прочтение обмена
в №3; консервативно без него13/30 (43,3%). Модели правы в9/30 (30,0%),
консервативно8/30. Ещё5 случаев имеют ошибочный reference и неверный/неподтверждённый
модельный ответ. Reference содержательно верен в4/30; в1/30 только округление.
В9/30 условие неполно/противоречиво; ещё2/30 допускают разные трактовки.
Эти11 случаев не записаны в подтверждённые ошибки reference.

Исходный D1 сохранил **ответы и статусы, но не программы или токены моделей**.
Ниже исходный target восстановлен из pinned compact-cache, а модельный код —
**новый CPU FP32 greedy replay** обеих pinned L3 FP16 weights на frozen D1 decoder.
Все60 ответов replay совпали с сохранёнными D1; все30 reference переисполнены
и совпали. Это делает код пригодным для содержательной проверки ответа,
но не доказывает совпадение исходных T4 токенов. Модельный код не становится train-target.

## Примеры

'''
    labels=dict(model_correct='Reference неверен; модели правы',both_wrong='Reference неверен; модели также не дают буквального ответа',
                reference_correct='Reference прав; модели ошиблись',rounding='Округление reference',
                ambiguous='Неоднозначное условие',malformed='Неполное или противоречивое условие')
    for num,(r,j) in enumerate(zip(rows,JUDGEMENTS),1):
        lines=[f'[!example]- {num}. row {r["row_id"]}: {labels[j[1]]}',
               '',r['question'],'',f'**Вердикт:** {j[2]}','',
               f'**Reference → {r["reference_answer"]:g}**','```python',r['reference_code'].rstrip(),'```']
        previous=None
        for name,v in r['cpu_replay'].items():
            answer=r['D1_answers'][name]
            lines+=['',f'**{name} → {answer:.12g}** (CPU replay; D1 ответ совпал)']
            if previous==v['code']:lines+=['Тот же код, что у C-main.']
            else:lines+=['```python',v['code'].rstrip(),'```']
            previous=v['code']
        text+='\n'.join('> '+line for block in lines for line in block.split('\n'))+'\n\n'
    text+='''## Источники

- [[d1-results]] и [[d1-plan]] — собственный протокол, SHA моделей и границы вывода.
- [Вердикты и sampling receipt](../results/d1-manual-review.json), [рендер отчёта](../research/d1-manual-report.py), [CPU replay](../research/d1-results.py).
- Локальный ignored `results/d1-labels/manual-review-data.json`: восстановленные программы, ответы и execution receipts; SHA в sampling receipt.
- Библиография: [[knowledge/bibliography#nlp-hw02-d1-results|nlp-hw02-d1-results]].
'''
    (WORK/'notes/d1-manual-review.md').write_text(text)
    print(counts)


if __name__=='__main__':main()
