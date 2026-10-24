---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-20
updated: 2026-10-20
sources:
  - kaggle-b7-results
  - nlp-hw02-e1-ensembles
aliases: []
---

# Пять вариантов финального ансамбля

Оркестратор отправил сжатый бандл после первого HTTP400: private friend1/nlp-hw02-variants, версия1, 20.10.2026 16:25UTC (по сообщению пользователя). Запуск COMPLETE; output в results/kaggle-runs/nlp-hw02-variants/1/. CPU-аудит завершён, все пять test укладываются в180с.

| Вариант | Состав, первая — anchor | Test, с | Пик reserved / NVML, GB | Fallback / guard test | Tune /2048 | Δ исполненных ответов к v11 | public |
|---|---|---:|---:|---|---:|---:|---|
| v1 | W B7 / C B7 / W B6 | 132.373 | 11.24 / 11.88 | 0 / 0 | 1552 (75.78%) | 211 | 80,00% |
| v2 | W B7 / C B7 / P1-B | 111.421 | 11.07 / 11.71 | 0 / 0 | 1557 (76.03%) | 212 | 80,10% |
| v3 | C B6 / P1-B / W B7 | 116.451 | 11.07 / 11.71 | 0 / 0 | 1552 (75.78%) | 181 | 79,90% |
| v4 | C B6 / P1-B / W B6 · E2 | 150.813 | 10.99 / 11.62 | 0 / 0 | 1548 (75.59%) | 187 | 79,90% |
| v5 | C B6 / W B6 / C B7 | 110.644 | 10.99 / 11.62 | 0 / 0 | 1548 (75.59%) | 205 | 80,10% |


**Лидерборд (сообщение пользователя, 21.10):** v1 80,00%, v2 80,10%, v3 79,90%, v4 79,90%, v5 80,10%; для сравнения v11 79,37% и budget640-пакет v11 79,17%. Все пять вариантов выше v11 на 0,5–0,9 п.п.; между собой они в пределах шума (±2 задачи из 960). По совокупности свежего tune (1557, лучший) и public (80,10%, лучший) финальным кандидатом выбран **v2: W B7 (якорь) + C B7 + P1-B, 1/1/1→1/8/8, budget640, test 111 с**.

Обычные варианты: greedy anchor +1/1/1 →1/8/8 при отсутствии численного единогласия. V4: C2/P8/W8, каждый4-й slot P/W при T1.0; stage3 ещё8P при доле победившей numeric-компоненты <0.5 от всех19голосов, с неуспехами в знаменателе. Budget640, guard160с, seed270042/T0.7, прежние исполнитель/голосование. E2 реализована опционально в пакете; default config и финальный inference.ipynb не изменены.

CPU: все пять SHA и strict-load FP16; C6+6/d512, W6+6/d768/QK-norm. E2 exact на4096строках журнала,2545early stops/826stage3;21тест и10cold dry-run процессов. [CPU](../checks/final-variants-cpu.json), [веса](../checks/final-variants-source-weights.json).

Первая попытка:20.10.2026 16:19:24UTC, notebook1,79МБ; HTTP400. Owned-инвентарь не содержит nlp-hw02-variants, чтение metadata —403. [Launch ledger](../checks/final-variants-launch.json), [проверка отказа](../checks/final-variants-push-failure.json). Вероятная причина — превышение1МБ исходника, но тело ошибки не сохранено. Сжаты только входные данные; новый notebook575299bytes, распакованные inputs exact поSHA. В первой сессии повторный push не выполнялся; последующий push оркестратора указан выше.


## Аудит версии 1

[Серверный код](../checks/final-variants-remote.json): все три кодовые ячейки совпадают с бандлом 575299 bytes, SHA256 `8a50a626ee602575ccc348871c7b1344905f7a897b0eebaa1240d54c8dd6bca7`. Совпадают 18 файлов кода, 4 входных файла, пять конфигов и SHA использованных моделей. API возвращает источники без суффиксов версий; закреплённые версии сохранены в metadata бандла, фактические веса подтверждены SHA.

Каждый test (1919) и tune (2048) выполнен в отдельном холодном процессе на Tesla T4, CUDA_VISIBLE_DEVICES=0. Test включает imports/load/warmup/generation/execute/vote и запись answers.jsonl; диагностика и tune-scoring идут после таймера. GB=10⁹ bytes; reserved — точный максимум PyTorch, NVML — device-wide максимум с опросом 20 мс.

Во всех десяти сплитах guard/fallback/partial/OOM/non-numeric и verification mismatches равны 0. V4 stage3: test 381, tune 416 строк.

Независимым повторным исполнением проверены 21754 строки: v11 test (1919) и все пять test (1919) + tune (2048); 11 свежих CPU executor-сервисов без сохранённого кеша исполнений. Одинаковый код внутри файла исполняется один раз: всего 21653 исполнения. Все ответы numeric. Сравнение с v11 использует math.isclose(rel_tol=abs_tol=1e-6); Δ — число изменившихся численных ответов, не прирост accuracy. Tune match_indices и accuracy совпадают с GPU-отчётами; каждый tune-журнал (2048 строк) прошёл replay победителя, early-stop и stage3.

[Полный аудит](../results/final-variants.json), [получение output](../checks/final-variants-download.json). Все JSONL по 1919 строк, строго `{"code": str}`; исходный test SHA и SHA порядка вопросов совпадают с отчётами, ответы скопированы побайтно:

- [v1.jsonl](../submissions/variants-2026-10-20/v1.jsonl) — SHA256 `87cb2ab48be79f2b584f8b06261ccb20c6cf6b5121150ddc89efd76e88456a1c`.
- [v2.jsonl](../submissions/variants-2026-10-20/v2.jsonl) — SHA256 `6a9685aad377db9826087baebce75c4719c967f06922efd8ed16b6db85ebadf5`.
- [v3.jsonl](../submissions/variants-2026-10-20/v3.jsonl) — SHA256 `37d7a0470b05eff4118c7c48cd1bd629e1aa57ce780038d7e62b02daaecfccae`.
- [v4.jsonl](../submissions/variants-2026-10-20/v4.jsonl) — SHA256 `c60213904897137068b4a51818e7fcfee0b6e0f803662b1838e14f8898e9fce0`.
- [v5.jsonl](../submissions/variants-2026-10-20/v5.jsonl) — SHA256 `65cc4c4393e760b3719d19b97b27588816056687bf08276daefad51f17eb3c4b`.

Повторить CPU-аудит из каталога домашки: `python work/kaggle/variants_results.py --root work/results/kaggle-runs/nlp-hw02-variants/1`. Public намеренно пустая; в бота ничего не отправлялось.

## Источники

- [final-package-v11](final-package-v11.md), [b7-results](b7-results.md), [e2-policy](e2-policy.md), [p1-results](p1-results.md); существующие записи опытов в knowledge/bibliography.md.
- [Первичное описание такого же HTTP400 из-за лимита исходника1МБ](https://www.kaggle.com/product-feedback/236663): возможное объяснение отказа, не подтверждённое телом текущей ошибки.
