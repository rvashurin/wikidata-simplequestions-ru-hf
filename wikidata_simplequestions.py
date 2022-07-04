# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import csv
import json
import os
from typing import Any

import datasets
from datasets.utils import logging


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
HuggingFace wrapper for https://github.com/askplatypus/wikidata-simplequestions dataset
Simplequestions dataset based on Wikidata.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = {
    "https://www.wikidata.org/wiki/"
}

_LANGS = [
    'ru',
    'en',
]


_DATA_DIRECTORY = './simplequestion'


class WikidataSimpleQuestionsConfig(datasets.BuilderConfig):
    """BuilderConfig for WikidataSimpleQuestions."""

    def __init__(self, **kwargs):
        """BuilderConfig for WikidataSimpleQuestions.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikidataSimpleQuestionsConfig, self).__init__(**kwargs)


def custom_download(src_url: str, dst_path: str) -> Any:
    import pywikibot
    from pywikibot.exceptions import NoPageError
    from joblib import Parallel, delayed
    import itertools

    with open(src_url, 'r') as f:
        lines = f.readlines()
        identifiers = []
        for line in lines:
            subj, _, obj, _ = line.split('\t')
            identifiers.append(subj)
            identifiers.append(obj)

    def _load_from_wikidata(identifier):
        site = pywikibot.Site("wikidata", "wikidata")
        repo = site.data_repository()

        try:
            item = pywikibot.ItemPage(repo, identifier)
            data = item.get(get_redirect=True)['labels']
            return (identifier, {ln: data.get(ln) for ln in _LANGS})
        except NoPageError:
            return (identifier, None)

    wikidata = Parallel(n_jobs=10)(
        delayed(_load_from_wikidata)(identifier)
        for identifier in logging.tqdm(identifiers, desc=f'Download {src_url}')
    )
    wikidata = dict(wikidata)

    with open(dst_path, 'w', encoding="utf-8") as f:
        json.dump(wikidata, f)



# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class WikidataSimpleQuestions(datasets.GeneratorBasedBuilder):
    """HuggingFace wrapper for https://github.com/askplatypus/wikidata-simplequestions dataset"""

    VERSION = datasets.Version("0.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = WikidataSimpleQuestionsConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    BUILDER_CONFIGS = []
    BUILDER_CONFIGS += [
        WikidataSimpleQuestionsConfig(
            name=f"main_{ln}",
            # version=VERSION,
            description="main version of wikidata simplequestions",
        )
        for ln in _LANGS
    ]
    BUILDER_CONFIGS += [
        WikidataSimpleQuestionsConfig(
            name=f"answerable_{ln}",
            # version=VERSION,
            description="answerable version of wikidata simplequestions",
        )
        for ln in _LANGS
    ]

    DEFAULT_CONFIG_NAME = "answerable_en"  # It's not mandatory to have a default configuration. Just use one if it make sense.

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
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)


        version, lang = self.config.name.split('_')
        if version == 'main':
            version = ''
        else:
            version = '_'+version

        results = []

        simplequestion_file_path = os.path.join(_DATA_DIRECTORY, f"annotated_wd_data_train{version}.txt")
        downloaded_wikidata_path = dl_manager.download_custom(
            simplequestion_file_path,
            custom_download
        )
        results.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": simplequestion_file_path,
                    "downloaded_wikidata_path": downloaded_wikidata_path,
                    "split": "train",
                    "lang": lang,
                },
            )
        )

        simplequestion_file_path = os.path.join(_DATA_DIRECTORY, f"annotated_wd_data_valid{version}.txt")
        downloaded_wikidata_path = dl_manager.download_custom(
            simplequestion_file_path,
            custom_download
        )
        results.append(
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": simplequestion_file_path,
                    "downloaded_wikidata_path": downloaded_wikidata_path,
                    "split": "validataion",
                    "lang": lang,
                },
            )
        )

        simplequestion_file_path = os.path.join(_DATA_DIRECTORY, f"annotated_wd_data_test{version}.txt")
        downloaded_wikidata_path = dl_manager.download_custom(
            simplequestion_file_path,
            custom_download
        )
        results.append(
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": simplequestion_file_path,
                    "downloaded_wikidata_path": downloaded_wikidata_path,
                    "split": "test",
                    "lang": lang,
                },
            )
        )


        return results

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, downloaded_wikidata_path, split, lang):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(downloaded_wikidata_path, 'r', encoding="utf-8") as f:
            wikidata = json.load(f)

        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = row.split('\t')
                if wikidata[data[0]] is not None and wikidata[data[2]] is not None:
                    yield (
                        key,
                        {
                            "subject": wikidata[data[0]].get(lang),
                            "property": data[1],
                            "object": wikidata[data[2]].get(lang),
                            "question": data[3],
                        },
                    )
