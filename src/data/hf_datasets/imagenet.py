from collections import defaultdict
import os
import csv
import glob
from pathlib import Path
from copy import deepcopy

import datasets

import logging

from src.data.hf_datasets.utils.imagenet import HIERARCHY, MANUAL_COLLAPSE_CLASS_IDS, DO_NOT_COLLAPSE
from utils.general import get_files_with_extensions

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


DIR_NAME_TO_SPLIT = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


class ImageNet(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = ImageNetConfig
    BUILDER_CONFIGS = [ImageNetConfig(name="default")]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            features=self.config.features,
        )

    def _split_generators(self, dl_manager):
        split_generators = []
        for split_dir_name in os.listdir(os.path.join(self.config.data_dir, "ILSVRC/Data/CLS-LOC")):
            class_from_folder = split_dir_name == "train"

            image_to_class_csv_path = None
            if not class_from_folder:
                image_to_class_csv_path = os.path.join(self.config.data_dir, f"LOC_{split_dir_name}_solution.csv")
                if not os.path.isfile(image_to_class_csv_path):
                    image_to_class_csv_path = None

            no_classes = not class_from_folder and image_to_class_csv_path is None
            if self.config.features_granularity != "test" and no_classes:
                continue
            split_generators.append(
                datasets.SplitGenerator(
                    name=DIR_NAME_TO_SPLIT[split_dir_name],
                    gen_kwargs={
                        "dir_path": os.path.join(self.config.data_dir, "ILSVRC/Data/CLS-LOC", split_dir_name),
                        "image_to_class_csv_path": image_to_class_csv_path,
                        "class_from_folder": class_from_folder,
                        "synset_mapping_path": os.path.join(self.config.data_dir, "LOC_synset_mapping.txt"),
                    },
                )
            )
        return split_generators

    def _generate_examples(self, dir_path, image_to_class_csv_path, class_from_folder, synset_mapping_path):
        images = get_files_with_extensions(dir_path, [".JPEG"])

        if image_to_class_csv_path is not None:
            image_to_class = {}
            with open(image_to_class_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_to_class[row["ImageId"]] = row["PredictionString"].split()[0]
        else:
            image_to_class = None

        class_id_to_text = {}
        with open(synset_mapping_path, "r") as f:
            for line in f:
                line = line.strip()
                separator_index = line.index(" ")
                class_id, class_text = line[:separator_index], line[separator_index + 1 :]
                class_id_to_text[class_id] = class_text

        if self.config.manual_class_collapse:
            add_hierarchy_texts(class_id_to_text)

        class_id_to_anecstry_lines = defaultdict(list)
        flatten_hierarchy(HIERARCHY, [], class_id_to_anecstry_lines)

        for image_path in images:
            id_ = Path(image_path).stem
            example = {
                "id": id_,
                "image_file_name": os.path.relpath(image_path, self.config.data_dir),
            }
            if self.config.features_granularity == "minimal":
                class_id = None
                if class_from_folder:
                    class_id = Path(os.path.dirname(image_path)).stem
                elif image_to_class is not None:
                    class_id = image_to_class[id_]

                if class_id is not None:
                    original_class_id = class_id
                    if self.config.manual_class_collapse and original_class_id not in DO_NOT_COLLAPSE:
                        ancestry_lines = class_id_to_anecstry_lines[class_id]

                        manual_collapse_class_ids_list = [[] for _ in range(len(ancestry_lines))]
                        for i, ancestry_line in enumerate(ancestry_lines):
                            for ancestor in ancestry_line:
                                if ancestor in MANUAL_COLLAPSE_CLASS_IDS:
                                    manual_collapse_class_ids_list[i].append(ancestor)
                        if all(
                            len(manual_collapse_class_ids) > 0
                            for manual_collapse_class_ids in manual_collapse_class_ids_list
                        ):
                            class_id = manual_collapse_class_ids_list[0][-1]

                    class_text = class_id_to_text[class_id]

                    example["class_id"] = class_id
                    example["class_text"] = class_text
                    if self.config.manual_class_collapse:
                        example["original_class_id"] = original_class_id

            yield id_, example


def flatten_hierarchy(node, path, class_id_to_anecstry_lines):
    class_id_to_anecstry_lines[node["id"]].append(deepcopy(path))
    path = deepcopy(path)
    path.append(node["id"])
    if "children" in node:
        for child in node["children"]:
            flatten_hierarchy(child, path, class_id_to_anecstry_lines)


def add_hierarchy_texts(class_id_to_text, node=None):
    if node is None:
        node = HIERARCHY

    if node["id"] not in class_id_to_text:
        class_id_to_text[node["id"]] = node["name"]

    if "children" in node:
        for child in node["children"]:
            add_hierarchy_texts(class_id_to_text, child)
