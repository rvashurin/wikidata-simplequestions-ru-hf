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
