from typing import Any, Callable, Dict
from collections import defaultdict

import os
import json
from copy import deepcopy

import datasets

import logging

logger = logging.getLogger(__name__)


class BaseConfig(datasets.BuilderConfig):
    def __init__(self, reload_saved="", **kwargs):
        super().__init__(**kwargs)
        # reload_saved is a comma-seperated list of splits for which the data file
        # is already saved in the format of the dataset, and examples from it can be
        # loaded as is. For splits not in this list, the data file is processed
        # regularly by the dataset.
        self.reload_saved = reload_saved


def base_split_generators(_split_generators: Callable):
    def wrapped_func(*args, **kwargs) -> Dict[str, Any]:
        _self = args[0] if len(args) > 0 else kwargs["self"]
        dl_manager = args[1] if len(args) > 1 else kwargs["dl_manager"]

        original_data_files = deepcopy(_self.config.data_files)
        reload_saved = _self.config.reload_saved.split(",") if len(_self.config.reload_saved) > 0 else []
        for split in reload_saved:
            del _self.config.data_files[split]

        split_generators = _split_generators(*args, **kwargs)

        config_modified_data_files = {split: v for split, v in original_data_files.items() if split in reload_saved}
        modified_data_files = dl_manager.download_and_extract(config_modified_data_files)
        for paths in modified_data_files.values():
            for i, path in enumerate(paths):
                if os.path.isdir(path) and len(os.listdir(path)) == 1:
                    paths[i] = os.path.join(path, os.listdir(path)[0])

        split_generators.extend(
            [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "file_paths": paths,
                        "split": split,
                    },
                )
                for split, paths in modified_data_files.items()
            ]
        )

        return split_generators

    return wrapped_func


def base_generate_examples(_generate_examples: Callable):
    def wrapped_func(*args, **kwargs) -> Dict[str, Any]:
        _self = args[0] if len(args) > 0 else kwargs["self"]
        split = kwargs["split"]
        reload_saved = _self.config.reload_saved.split(",")
        if split in reload_saved:
            for file_path in kwargs["file_paths"]:
                logger.info(f"Loading {file_path} for {kwargs['split']} split")
                with open(file_path, "r") as f:
                    for line in f:
                        example = json.loads(line)
                        yield example["id"], example
        else:
            yield from _generate_examples(*args, **kwargs)

    return wrapped_func
