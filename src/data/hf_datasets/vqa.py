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
"""Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering"""


import os
import json
from collections import defaultdict

import datasets

import logging

logger = logging.getLogger(__name__)


_CITATION = """\
@article{balanced_vqa_v2,
title={Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering},
author={Yash Goyal and Tejas Khot and Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2017},
}
"""

_DESCRIPTION = """\
We propose to counter these language priors for the task of Visual Question Answering (VQA) and make vision (the V in VQA) matter! Specifically, we balance the popular VQA dataset by collecting complementary images such that every question in our balanced dataset is associated with not just a single image, but rather a pair of similar images that result in two different answers to the question. Our dataset is by construction more balanced than the original VQA dataset and has approximately twice the number of image-question pairs.
"""


class VqaConfig(datasets.BuilderConfig):
    """BuilderConfig for VQA."""

    def __init__(
        self, features_granularity="minimal", original_format=False, group_non_train_by_image=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.features_granularity = features_granularity
        self.original_format = original_format
        self.group_non_train_by_image = group_non_train_by_image

    @property
    def features(self):
        common_features = {
            "id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "image_file_name": datasets.Value("string"),
            "official_split": datasets.Value("string"),
        }
        if self.features_granularity == "minimal":
            features_dict = {
                "multiple_choice_answer": datasets.Value("string"),
                "answers": datasets.features.Sequence(datasets.Value("string")),
            }
        elif self.features_granularity == "test":
            features_dict = {}
        elif self.features_granularity == "full":
            # Add all annotation fields
            features_dict = {}

        features_dict.update(common_features)
        return datasets.Features(features_dict)


IMAGE_FILE_PREFIXES = {
    "train": "train2014/COCO_train2014_",
    "validation": "val2014/COCO_val2014_",
    "test": "test2015/COCO_test2015_",
}


def get_image_file_prefix(split):
    return IMAGE_FILE_PREFIXES[split]


class Vqa(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = VqaConfig
    BUILDER_CONFIGS = [VqaConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage="https://visualqa.org/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_files = dl_manager.download_and_extract(self.config.data_files)
        for paths in data_files.values():
            for i, path in enumerate(paths):
                if os.path.isdir(path) and len(os.listdir(path)) == 1:
                    paths[i] = os.path.join(path, os.listdir(path)[0])

        if not self.config.original_format:
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "file_paths": paths,
                        "split": split,
                    },
                )
                for split, paths in data_files.items()
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "file_paths": paths,
                        "split": split,
                        "annotations_paths": data_files[f"{split}_annotations"]
                        if f"{split}_annotations" in data_files
                        else None,
                    },
                )
                for split, paths in data_files.items()
                if not split.endswith("annotations")
            ]

    def _generate_examples(self, file_paths, split, annotations_paths=None):
        tuples = self._generate_examples_helper(file_paths, split, annotations_paths)
        if split != "train" and self.config.group_non_train_by_image:
            tuples_per_image = defaultdict(list)
            for id_, example in tuples:
                tuples_per_image[example["image_file_name"]].append((id_, example))

            for tuples in tuples_per_image.values():
                for id_, example in tuples:
                    yield id_, example
        else:
            for id_, example in tuples:
                yield id_, example

    def _generate_examples_helper(self, file_paths, split, annotations_paths=None):
        """Yields examples as (key, example) tuples."""
        if annotations_paths is None:
            annotations_paths = [None] * len(file_paths)
        for file_path, annotations_path in zip(file_paths, annotations_paths):
            if not self.config.original_format:
                logger.info(f"Loading {file_path} for {split} split")
                with open(file_path, "r") as f:
                    for line in f:
                        example = json.loads(line)
                        yield example["id"], example
            else:
                logger.info(f"Loading {file_path} for {split} split")
                with open(file_path, encoding="utf-8") as f:
                    dataset = json.load(f)["questions"]

                if annotations_path is not None:
                    logger.info(f"Loading {annotations_path} for {split} split")
                    with open(annotations_path, encoding="utf-8") as f:
                        annotations = json.load(f)["annotations"]
                else:
                    annotations = [None for _ in dataset]

                for question_obj, annotation_obj in zip(dataset, annotations):
                    id_ = str(question_obj["question_id"])
                    image_file_name = question_obj.get(
                        "image_file_name",
                        get_image_file_prefix(split) + str(question_obj["image_id"]).zfill(12) + ".jpg",
                    )
                    example = {
                        "id": id_,
                        "question": question_obj["question"],
                        "image_file_name": image_file_name,
                        "official_split": split,
                    }

                    if self.config.features_granularity == "full":
                        pass
                    elif self.config.features_granularity == "minimal":
                        example["multiple_choice_answer"] = annotation_obj["multiple_choice_answer"]
                        example["answers"] = [answer_obj["answer"] for answer_obj in annotation_obj["answers"]]

                    yield id_, example
