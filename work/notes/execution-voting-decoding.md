---
type: experiment
course:
  - nlp
status: learning
understanding: 3
created: 2026-10-11
updated: 2026-10-11
sources:
  - https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html
  - https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
aliases: []
---

# ДЗ2, опыт27: декодирование и голосование по исполнению

[Ноутбук S1](../kaggle/notebooks/s1-t4x2-26b-27-28.ipynb) · [Kaggle/команды](../kaggle/README.md) · [26](wsd-data-scaling-and-sharded-cache.md) · Декодирование. GPU0 после полного финала26b/50000. **S1 v1 завершил №27; нового обучения нет.**

[Результат S1 v1](27-execution-voting-s1-results.md) · [Сводка сессии](../results/kaggle-s1-v1-summary.json). Пороги ниже сохранены без изменений.

## Гипотеза и данные

**Наша гипотеза:** разные программы могут давать один правильный числовой ответ; голосование по исполнению повысит exec-match при допустимом времени. Предшествующий контроль — greedy FP16 sortedb512/cap512 на тех же финальных весах A. Улучшение одной исполняемости недостаточно: на3,84M numeric4028, matches2446/4096.

Наборы/tokenizer/reference/policy прежние. Primary4096; secondary991 изold1024. Fresh только исключается по вопросу, не оценивается. Решения и пороги заданы пользователем заранее, не заимствованы из внешней работы.

## Протокол

1. Сохранить контрольный greedy из финальной оценки26b. Отдельный encoder проход контрольной оценки допустим; внутри candidate-grid encoder и cross K/V считаются один раз на вопрос для всех температур.
2. $T=0.7$:32 samples на4096 иold1024, $k=4,8,16,32$ — первыеk кандидатов одной генерации. $T=0.5,1.0$:16 samples, также первые4/8/16. $T$ делит logits перед softmax, $k$ — количество численных голосов до отбрасывания нечисленных.
3. Greedy+exec-fallback: если greedy не вернул число, взять первый numeric sample из первыхk приT0.7; если такого нет, оставить greedy. Это не выбирает ответ по gold.
4. Голосование: finite solve/bool→int, кандидатные результаты близки по `math.isclose(rel_tol=abs_tol=1e-6)`. Близость не транзитивна: кластер — связная компонента графа попарной близости. Максимум голосов; ничья → кластер, близкий greedy-ответу, затем максимум средней logprob кандидатов. Представитель — программа с максимальной mean logprob в победившем кластере. При отсутствии numeric оставить greedy.
5. Каждая voting-конфигурация имеет вариант, где greedy добавлен одним голосом. Logprob — средняя по сгенерированным токенам, включаяEOS, под **исходным** распределением модели; PAD/BOS запрещены. Это позволяет сравнивать ветки с разнымиT.

Cross K/V имеют форму `[question,1,heads,source,head_dim]`, broadcast по sample lane без `repeat/contiguous`; собственный self KV отдельный. По512 candidate rows по умолчанию,1024/2048 конфигурируемы. Backend5D может быть медленнее: скорость не установлена. Кандидаты сохраняются по question-batches, RNG детерминирован на batch/T; исполнение чанками2000, по2 worker, результаты/числа/SHA resumable. Граница каждогоk восстанавливается из одной сохранённой генерации.

## Оценка и решение

Для каждой конфигурации: exec4096/991, exact paired McNemar против контрольногоgreedy, accuracy по длине referenceBPE безEOS `<60`,60–90,90–120,≥120; время, число фактически исполненных и использованных кандидатов, длины выбранных программ. CE уже получена на тех же весах в26b; декодирование CE не меняет.

По heldout4k выбрать три лучшие альтернативы, дополнительно greedy; для каждой выполнить **свежий** test1919. Полный таймер: чтениеtest, загрузка checkpoint/tokenizer, encoder/decode, исполнение, голосование, закрытие answersfile. ПроверкиSHA/summary идут после таймера; расходы промежуточной записи внутри него учитываются. Ответ — код представителя победившего кластера; test gold не используется.

Из трёх измеренных альтернатив принять максимум heldout exec при test≤120с и улучшении≥1.5п.п. (минимум62 новых matches/4096). Иначе greedy. При равенстве primary — меньший full test time. Это ограниченный поиск, а не доказательство оптимальногоT/k. Если бюджет прервал сетку/тайминг, итог partial, решение не завершено.

## Проверки и ограничения до S1

CPU: shared5D KV logits совпали с исходным CachedDecoder на tiny-модели; greedy teacher replay, реальное sampling/exec, вложенныеk, tie/fallback/greedy-vote и1.5п.п./120с, повторное использование batches/SHA, full timer проверены. Полный A inference0, GPU0. CUDA half/SDPA/backend, peak memory, полный4096/991/test1919 и3ч бюджет не проверены. Один seed и повторный выбор поheldout; результат не является official score.

## Источники

- Пользовательский протокол S1 от2026-10-11; [полученный26b](wsd-data-scaling-and-sharded-cache.md) и его сводка — исходные веса/данные/контроль, не новые измерения27.
- PyTorch2.11, [scaled_dot_product_attention](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html): API внимания; фактическая CUDA скорость требует замера.
- Kaggle, [kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md): `machine_shape`/private/data sources; проверка установленного2.2.4 — в [CPU metadata report](../checks/kaggle-cli-224-metadata-checks.json).
