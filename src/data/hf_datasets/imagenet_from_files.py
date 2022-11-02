import json

import datasets

import logging


logger = logging.getLogger(__name__)


class ImageNetConfig(datasets.BuilderConfig):
    """BuilderConfig for ImageNet."""

    def __init__(self, features_granularity="minimal", manual_class_collapse=False, **kwargs):
        super().__init__(**kwargs)
        self.features_granularity = features_granularity
        self.manual_class_collapse = manual_class_collapse

    @property
    def features(self):
        common_features = {
            "id": datasets.Value("string"),
            "image_file_name": datasets.Value("string"),
        }
        if self.features_granularity == "minimal":
            features_dict = {
                "class_id": datasets.Value("string"),
                "class_text": datasets.Value("string"),
            }
            if self.manual_class_collapse:
                features_dict["original_class_id"] = datasets.Value("string")
        elif self.features_granularity == "test":
            features_dict = {}

        features_dict.update(common_features)
        return datasets.Features(features_dict)


class ImageNetFromFiles(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = ImageNetConfig
    BUILDER_CONFIGS = [ImageNetConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=self.config.features,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "file_path": paths[0],
                },
            )
            for split, paths in self.config.data_files.items()
        ]

    def _generate_examples(self, file_path):
        with open(file_path, "r") as f:
            for line in f:
                example_from_file = json.loads(line)

                example = {
                    "id": example_from_file["id"],
                    "image_file_name": example_from_file["image_file_name"],
                }

                if self.config.features_granularity == "minimal":
                    example.update(
                        {
                            "class_id": example_from_file["class_id"],
                            "class_text": example_from_file["class_text"],
                        }
                    )
                    if self.config.manual_class_collapse:
                        example["original_class_id"] = example_from_file["original_class_id"]

                yield example["id"], example
