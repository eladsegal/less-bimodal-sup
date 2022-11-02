import os
import json
import re

import datasets

import logging

logger = logging.getLogger(__name__)


_URL = "https://github.com/lil-lab/nlvr/raw/master/nlvr2/data"
_URLS = {
    "default": {
        "train": os.path.join(_URL, "train.json"),
        "dev": os.path.join(_URL, "dev.json"),
        "test1": os.path.join(_URL, "test1.json"),
    },
    "balanced": {
        "balanced_dev": os.path.join(_URL, "balanced", "balanced_dev.json"),
        "balanced_test1": os.path.join(_URL, "balanced", "balanced_test1.json"),
    },
    "unbalanced": {
        "unbalanced_dev": os.path.join(_URL, "unbalanced", "unbalanced_dev.json"),
        "unbalanced_test1": os.path.join(_URL, "unbalanced", "unbalanced_test1.json"),
    },
}
_URLS["all"] = {u: w for k, v in _URLS.items() for u, w in v.items()}


class Nlvr2Config(datasets.BuilderConfig):
    """BuilderConfig for NLVR2."""

    def __init__(self, features_granularity="minimal", **kwargs):
        super().__init__(**kwargs)
        self.features_granularity = features_granularity

    @property
    def features(self):
        common_features = {
            "id": datasets.Value("string"),
            "sentence": datasets.Value("string"),
            "image_file_name_0": datasets.Value("string"),
            "image_file_name_1": datasets.Value("string"),
        }
        if self.features_granularity == "minimal":
            features_dict = {
                "label": datasets.Value("string"),
            }
        elif self.features_granularity == "test":
            features_dict = {}
        elif self.features_granularity == "full":
            features_dict = {
                "set_id": datasets.Value("string"),
                "pair_id": datasets.Value("string"),
                "sentence_id": datasets.Value("string"),
                "label": datasets.Value("string"),
            }

        features_dict.update(common_features)
        return datasets.Features(features_dict)


class Nlvr2(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = Nlvr2Config
    BUILDER_CONFIGS = [
        Nlvr2Config(name="default"),
        Nlvr2Config(name="balanced"),
        Nlvr2Config(name="unbalanced"),
        Nlvr2Config(name="all"),
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=self.config.features,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS[self.config.name])

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "file_path": path,
                    "split": split,
                },
            )
            for split, path in downloaded_files.items()
        ]

    def _generate_examples(self, file_path, split):
        """Yields examples as (key, example) tuples."""

        logger.info(f"Loading {file_path} for {split} split")
        with open(file_path, encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line]

        id_regex = re.compile("(.+)-(.+)-(.+)-(.+)")
        for instance in dataset:
            id_ = instance["identifier"]
            split, set_id, pair_id, sentence_id = id_regex.search(id_).groups()

            if split == "train":
                image_file_name_0 = os.path.join(
                    split, str(instance["directory"]), f"{split}-{set_id}-{pair_id}-img0.png"
                )
                image_file_name_1 = os.path.join(
                    split, str(instance["directory"]), f"{split}-{set_id}-{pair_id}-img1.png"
                )
            else:
                image_file_name_0 = os.path.join(split, f"{split}-{set_id}-{pair_id}-img0.png")
                image_file_name_1 = os.path.join(split, f"{split}-{set_id}-{pair_id}-img1.png")

            example = {
                "id": id_,
                "image_file_name_0": image_file_name_0,
                "image_file_name_1": image_file_name_1,
            }

            if self.config.features_granularity == "full":
                example.update(
                    {
                        "set_id": set_id,
                        "pair_id": pair_id,
                        "sentence_id": sentence_id,
                        "sentence": instance["sentence"],
                        "label": instance["label"],
                    }
                )
            elif self.config.features_granularity == "minimal":
                example.update(
                    {
                        "sentence": instance["sentence"],
                        "label": instance["label"],
                    }
                )
            elif self.config.features_granularity == "test":
                example.update(
                    {
                        "sentence": instance["sentence"],
                    }
                )

            yield id_, example
