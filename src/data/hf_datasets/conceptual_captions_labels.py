import os
import csv

import datasets

from utils.general import get_files_with_extensions

import logging

logger = logging.getLogger(__name__)


class ConceptualCaptionsLabelsConfig(datasets.BuilderConfig):
    def __init__(self, index_to_url_tsv_path=None, features_granularity="full", **kwargs):
        super().__init__(**kwargs)
        self.index_to_url_tsv_path = index_to_url_tsv_path
        self.features_granularity = features_granularity

    @property
    def features(self):
        common_features = {
            "id": datasets.Value("string"),
            "image_file_name": datasets.Value("string"),
        }
        if self.features_granularity == "full":
            features_dict = {
                "labels": datasets.Sequence(datasets.Value("string")),
                "confidence_scores": datasets.Sequence(datasets.Value("float32")),
            }
        elif self.features_granularity == "test":
            features_dict = {}

        features_dict.update(common_features)
        return datasets.Features(features_dict)


class ConceptualCaptionsLabels(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = ConceptualCaptionsLabelsConfig
    BUILDER_CONFIGS = [ConceptualCaptionsLabelsConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=self.config.features,
        )

    def _split_generators(self, dl_manager):
        split_generators = []
        split_generators.append(
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "tsv_path": self.config.data_files["train"][0],
                    "split": "train",
                },
            )
        )
        return split_generators

    def _generate_examples(self, tsv_path, split):
        split_data_dir = os.path.join(self.config.data_dir, split)

        data = []
        with open(tsv_path) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for row in read_tsv:
                data.append(row)

        url_to_index = {}
        with open(self.config.index_to_url_tsv_path) as f:
            read_tsv = csv.reader(f, delimiter="\t")
            for i, row in enumerate(read_tsv):
                url_to_index[row[1]] = i

        subdir_ranges = [get_subdir_range(subdir) for subdir in os.listdir(split_data_dir)]
        image_paths = set(get_files_with_extensions(split_data_dir, [".jpg"]))

        for row in data:
            i = url_to_index[row[1]]
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
            }

            if self.config.features_granularity == "full":
                example["labels"] = [label.strip() for label in row[2].split(",")]
                if len(example["labels"]) == 0 or example["labels"][0] == "":
                    continue
                example["confidence_scores"] = [float(confidence_score) for confidence_score in row[4].split(",")]

            yield id_, example


def get_subdir_range(subdir):
    return tuple(map(int, subdir.split("_")))
