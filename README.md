---
dataset_info:
- config_name: main_ru
  features:
  - name: subject
    dtype: string
  - name: property
    dtype: string
  - name: object
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 1554325
    num_examples: 14209
  - name: validation
    num_bytes: 223596
    num_examples: 2057
  - name: test
    num_bytes: 460280
    num_examples: 4211
  download_size: 0
  dataset_size: 2238201
- config_name: main_en
  features:
  - name: subject
    dtype: string
  - name: property
    dtype: string
  - name: object
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 2728724
    num_examples: 30903
  - name: validation
    num_bytes: 395767
    num_examples: 4479
  - name: test
    num_bytes: 812209
    num_examples: 9202
  download_size: 0
  dataset_size: 3936700
- config_name: answerable_ru
  features:
  - name: subject
    dtype: string
  - name: property
    dtype: string
  - name: object
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 1179440
    num_examples: 8483
  - name: validation
    num_bytes: 174117
    num_examples: 1263
  - name: test
    num_bytes: 348433
    num_examples: 2509
  download_size: 0
  dataset_size: 1701990
- config_name: answerable_en
  features:
  - name: subject
    dtype: string
  - name: property
    dtype: string
  - name: object
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: train
    num_bytes: 2024362
    num_examples: 17331
  - name: validation
    num_bytes: 299179
    num_examples: 2561
  - name: test
    num_bytes: 599222
    num_examples: 5136
  download_size: 0
  dataset_size: 2922763
---
# Wikidata Simplequestions

Huggingface Dataset wrapper for Wikidata-simplequestion dataset

### Usage

```bash
git clone git@github.com:skoltech-nlp/wikidata-simplequestions-hf.git wikidata_simplequestions
```

```python3
from datasets import load_dataset;
load_dataset('../wikidata_simplequestions', 'answerable_en', cache_dir='/YOUR_PATH_TO_CACHE/', ignore_verifications=True)
```