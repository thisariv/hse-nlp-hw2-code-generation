---
language:
- en
license: other
pretty_name: tinysms qa 12m
tags:
- code
- mathematics
- tinysms
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train/part-*.parquet
  - split: test
    path: data/test/test.parquet
---

# tinysms qa 12m

Every training row of [`TinyGSM/TinyGSM`](https://huggingface.co/datasets/TinyGSM/TinyGSM) as a math question
, preprocessed exactly as the model is trained on them:
the docstring is stripped and the entry point is renamed to `solve`.
Nothing else is stored -- no answers, no token counts, no rendered prompt, no ids.

* `train`: **11,838,003** rows in 60 parquet file(s)
* `test`: the 1,919 questions of the project's test split -- **questions only**, no reference
  programs and no gold values, so the split is safe to publish
* the last 4,000 rows of shard 0 -- the validation and test rows of `src/data.py` -- are **not** in `train`
* 4,106 further rows whose **question text** is one of the val/test questions were dropped as well (`DROP_EVAL_COPIES = True`): TinyGSM ships the same question with several reference programs, so those rows are copies of the evaluation questions
* the reference programs are **not** executed, unlike the smaller `tinysms_train_qa` and
  `tinysms_train_qa_1m` sets: no answers are shipped, so a reference that fails to return a number is
  not filtered out
* built shard by shard: every parquet file is uploaded to the Hub and deleted before the next one is
  written, so the whole set never exists on disk at once

## Loading

```python
from datasets import load_dataset
ds = load_dataset("SlavaYus/tinysms_qa_12m")           # DatasetDict(train=..., test=...)
print(ds["train"][0], len(ds["test"]))
```

## Schema

`train`:
```
question: string
code: string
```

`test`:
```
question: string
```
