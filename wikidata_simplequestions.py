import csv
import json
import os
from typing import Any

import datasets
from datasets.utils import logging


_DESCRIPTION = """\
HuggingFace wrapper for https://github.com/askplatypus/wikidata-simplequestions dataset
Simplequestions dataset based on Wikidata.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_LANGS = [
    "ru",
    "en",
]

_URL = "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/"
_DATA_DIRECTORY = "./simplequestion"
VERSION = datasets.Version("0.0.1")


class WikidataSimpleQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for WikidataSimpleQuestions."""

    def __init__(self, **kwargs):
        """BuilderConfig for WikidataSimpleQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikidataSimpleQuestionsConfig, self).__init__(**kwargs)


class WikidataSimpleQuestions(datasets.GeneratorBasedBuilder):
    """HuggingFace wrapper for https://github.com/askplatypus/wikidata-simplequestions dataset"""

    BUILDER_CONFIG_CLASS = WikidataSimpleQuestionsConfig
    BUILDER_CONFIGS = []
    BUILDER_CONFIGS += [
        WikidataSimpleQuestionsConfig(
            name=f"main_{ln}",
            version=VERSION,
            description="main version of wikidata simplequestions",
        )
        for ln in _LANGS
    ]
    BUILDER_CONFIGS += [
        WikidataSimpleQuestionsConfig(
            name=f"answerable_{ln}",
            version=VERSION,
            description="answerable version of wikidata simplequestions",
        )
        for ln in _LANGS
    ]

    DEFAULT_CONFIG_NAME = "answerable_en"

    def _info(self):
        features = datasets.Features(
            {
                "subject": datasets.Value("string"),
                "property": datasets.Value("string"),
                "object": datasets.Value("string"),
                "question": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        if self.config.name == "default":
            version, lang = "main", "en"
        else:
            version, lang = self.config.name.split("_")

        if version == "main":
            version = ""
        else:
            version = "_" + version

        data_dir = os.path.join(self.base_path, _DATA_DIRECTORY)
        if lang == "en":
            vocab_path = os.path.join(data_dir, "reverse_vocab_wikidata_en.json")
        elif lang == "ru":
            vocab_path = os.path.join(data_dir, "reverse_vocab_wikidata_ru.json")
        else:
            raise ValueError(f"Language {lang} not supported")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"annotated_wd_data_train{version}.txt"),
                    "vocab_path": vocab_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"annotated_wd_data_valid{version}.txt"),
                    "vocab_path": vocab_path,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, f"annotated_wd_data_test{version}.txt"),
                    "vocab_path": vocab_path,
                },
            ),
        ]

    def _generate_examples(self, filepath, vocab_path):

        with open(vocab_path, "r") as file_handler:
            wikidata_vocab = json.load(file_handler)
        wikidata_vocab = {v:k for k,v in wikidata_vocab.items()}
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = row.split("\t")
                if (
                    wikidata_vocab.get(data[0]) is not None
                    and wikidata_vocab.get(data[2]) is not None
                ):
                    yield (
                        key,
                        {
                            "subject": wikidata_vocab[data[0]],
                            "property": data[1],
                            "object": wikidata_vocab[data[2]],
                            "question": data[3],
                        },
                    )
