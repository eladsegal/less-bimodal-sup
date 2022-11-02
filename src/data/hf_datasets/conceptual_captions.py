import os
import csv
import json

import datasets

from src.data.hf_datasets.helpers import BaseConfig, base_split_generators, base_generate_examples
from utils.general import get_files_with_extensions

import logging

logger = logging.getLogger(__name__)


class ConceptualCaptionsConfig(BaseConfig):
    def __init__(self, features_granularity="full", **kwargs):
        super().__init__(**kwargs)
        self.features_granularity = features_granularity

    @property
    def features(self):
        common_features = {
            "id": datasets.Value("string"),
            "image_file_name": datasets.Value("string"),
            "index": datasets.Value("int32"),
        }
        if self.features_granularity == "full":
            features_dict = {
                "caption": datasets.Value("string"),
            }
        elif self.features_granularity == "test":
            features_dict = {}

        features_dict.update(common_features)
        return datasets.Features(features_dict)


class ConceptualCaptions(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = ConceptualCaptionsConfig
    BUILDER_CONFIGS = [ConceptualCaptionsConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=self.config.features,
        )

    @base_split_generators
    def _split_generators(self, dl_manager):
        split_generators = []
        for split in os.listdir(self.config.data_dir):
            if os.path.isdir(os.path.join(self.config.data_dir, split)) and split in self.config.data_files:
                split_generators.append(
                    datasets.SplitGenerator(
                        name=split,
                        gen_kwargs={
                            "tsv_path": self.config.data_files[split][0],
                            "split": split,
                        },
                    )
                )
        return split_generators

    @base_generate_examples
    def _generate_examples(self, split, tsv_path=None, file_paths=None):
        data = []
        with open(tsv_path) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for row in read_tsv:
                data.append(row)

        split_data_dir = os.path.join(self.config.data_dir, split)
        image_paths = set(get_files_with_extensions(split_data_dir, [".jpg"]))

        subdir_ranges = [get_subdir_range(subdir) for subdir in os.listdir(split_data_dir)]

        for i, row in enumerate(data):
            id_ = f"{i:08d}"

            subdir = None
            for subdir_range in subdir_ranges:
                start, end = subdir_range
                if start <= i < end:
                    subdir = f"{start:08d}_{end:08d}"
                    break
            if subdir is None:
                raise ValueError(f"Could not find subdir for row {i} in {split_data_dir}")

            image_path = os.path.join(split_data_dir, subdir, f"{id_}.jpg")
            if image_path not in image_paths:
                continue

            example = {
                "id": id_,
                "image_file_name": os.path.relpath(image_path, self.config.data_dir),
                "index": i,
            }

            if self.config.features_granularity == "full":
                example["caption"] = row[0]

            yield id_, example


def get_subdir_range(subdir):
    return tuple(map(int, subdir.split("_")))
