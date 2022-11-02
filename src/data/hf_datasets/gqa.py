# coding=utf-8
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
"""GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering"""


import json

import datasets

import logging

logger = logging.getLogger(__name__)


_CITATION = """\
@article{Hudson2019GQAAN,
title={GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering},
author={D. A. Hudson and Christopher D. Manning},
journal={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2019},
pages={6693-6702}
}
"""

_DESCRIPTION = """\
We introduce GQA, a new dataset for real-world visual reasoning and compositional question answering, seeking to address key shortcomings of previous VQA datasets.
"""


class GqaConfig(datasets.BuilderConfig):
    """BuilderConfig for GQA."""

    def __init__(self, features_granularity="minimal", **kwargs):
        super().__init__(**kwargs)
        self.features_granularity = features_granularity

    @property
    def features(self):
        common_features = {
            "id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "image_file_name": datasets.Value("string"),
        }
        if self.features_granularity == "minimal":
            features_dict = {
                "answer": datasets.Value("string"),
            }
        elif self.features_granularity == "test":
            features_dict = {}
        elif self.features_granularity == "full":
            # Add all fields, including semantic and question types
            features_dict = {}

        features_dict.update(common_features)
        return datasets.Features(features_dict)


class Gqa(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = GqaConfig
    BUILDER_CONFIGS = [GqaConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage="https://cs.stanford.edu/people/dorarad/gqa/index.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "file_paths": paths,
                    "split": split,
                },
            )
            for split, paths in self.config.data_files.items()
        ]

    def _generate_examples(self, file_paths, split):
        """Yields examples as (key, example) tuples."""

        for file_path in file_paths:
            logger.info(f"Loading {file_path} for {split} split")
            with open(file_path, encoding="utf-8") as f:
                dataset = json.load(f)

            for id_, question_obj in dataset.items():
                example = {
                    "id": id_,
                    "question": question_obj["question"],
                    "image_file_name": question_obj["imageId"] + ".jpg",
                }

                if self.config.features_granularity == "full":
                    pass
                elif self.config.features_granularity == "minimal":
                    example["answer"] = question_obj["answer"]

                yield id_, example
