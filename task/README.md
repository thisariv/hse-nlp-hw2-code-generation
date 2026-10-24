# Источники ДЗ 2

Проверено 2026-10-07. Оригиналы в этом каталоге сохранены без изменения содержимого, включая отсутствие Properties. План и пояснения находятся в [обзоре ДЗ](../hw02-code-generation.md).

- Репозиторий: [ashaba1in/hse-nlp](https://github.com/ashaba1in/hse-nlp), ветка `2026`.
- HEAD ветки проверен через `git ls-remote`: `7f837940c4af9a686fca7bff5f02345218b6fe80`.
- [Оригинальное условие](https://github.com/ashaba1in/hse-nlp/blob/7f837940c4af9a686fca7bff5f02345218b6fe80/homeworks/hw2_code_generation.md) → `hw2-code-generation.md`, 7882 байта.
- SHA-256 условия: `9381b7ab7b5d2cab501095c68a59240a3d2893506958f0e2159683c5d905ae75`.
- Датасет: [SlavaYus/tinysms_qa_12m](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m), revision `b2724689ff079e9c22efd548fe134bfb29036dbb`.
- [Карточка датасета](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/blob/b2724689ff079e9c22efd548fe134bfb29036dbb/README.md) → `dataset-card.md`, 1804 байта.
- SHA-256 карточки: `a2fbc05746d9849e36cdfb124ae5c900008939886f74ce3449f03542af29fea7`.

Прочитаны условие, README курса, карточка и `meta.json` датасета; метаданные splits сверены с публичным Dataset Viewer. Просмотрены три строки train через API, без исполнения кода. Датасет целиком и checkpoints не загружались; обучение, инференс, локальная оценка и измерения на T4 не выполнялись. Лидерборд и бот не использовались.

Для будущего отчёта: на этом этапе агент использован для поиска актуального условия, проверки источников и подготовки плана; это определило порядок работ и список неуточнённых деталей. Гипотезы об архитектуре и ускорениях пока не проверены экспериментально.
