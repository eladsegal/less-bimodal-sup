from typing import Dict, Any, Optional, Union, List
from collections import defaultdict

import os
import sys
from copy import deepcopy
import psutil

from datasets import Dataset, DatasetDict
from src.data.datamodules.base_datamodule import BaseDataModule
from src.data.dataset_containers import ComplexDatasetContainer
from src.data.dataset_containers.hf_dataset_container import HfDatasetContainer
from src.data.dataset_containers.multi_dataset_container import MultiDatasetContainer
from transformers import PreTrainedTokenizerBase

from src.data.dataset_helper import DatasetHelper
from src.utils.inspection import find_in_class

import logging

logger = logging.getLogger(__name__)


class HfDataModule(BaseDataModule):
    def __init__(
        self,
        map_num_proc: Union[str, int] = 1,
        use_preprocessing_cache: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._map_num_proc = map_num_proc if map_num_proc != "max" else psutil.cpu_count()
        self._use_preprocessing_cache = use_preprocessing_cache

    def dataset_container_preprocessing(self) -> None:
        if isinstance(self.dataset_container, ComplexDatasetContainer):
            ready_datasets: DatasetDict = self.dataset_container[self.dataset_container.main_key].ready_datasets
        elif isinstance(self.dataset_container, MultiDatasetContainer):
            ready_datasets: DatasetDict = self.dataset_container.concatenated_datasets
        else:
            ready_datasets: DatasetDict = self.dataset_container.ready_datasets

        self.print_split_sizes(ready_datasets, "ready_datasets", "Split sizes before processing:")
        self.preprocessed_datasets = DatasetDict()
        for split, dataset in ready_datasets.items():
            dataset: Dataset
            self.preprocessed_datasets[split] = dataset.map(
                self._get_preprocessing_fn(split),
                fn_kwargs=self._get_preprocessing_kwargs(split),
                num_proc=max(1, self._map_num_proc),
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=self._use_preprocessing_cache,
                with_indices=True,
            )
        self.print_split_sizes(self.preprocessed_datasets, "preprocessed_datasets", "Split sizes after processing:")

        # Take care of the pidx column
        if self.preprocessed_datasets is not None:
            for split in self.preprocessed_datasets.keys():
                self.preprocessed_datasets[split] = self.preprocessed_datasets[split].add_column(
                    "pidx", [i for i in range(len(self.preprocessed_datasets[split]))]
                )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def print_split_sizes(self, dataset: DatasetDict, name: str, message: Optional[str] = None):
        if message is not None:
            logger.info(message)
        for split in dataset.keys():
            logger.info(f'{name}["{split}"]: {len(dataset[split])}')

    def _get_preprocessing_fn(self, split):
        if split == "test":
            return self.test_preprocessing
        elif split == "validation":
            return self.val_preprocessing
        elif split == "train":
            return self.train_preprocessing
        return self.general_preprocessing

    def _get_preprocessing_kwargs(self, split):
        if split == "test":
            preprocessing_kwargs = self.test_preprocessing_kwargs
        elif split == "validation":
            preprocessing_kwargs = self.val_preprocessing_kwargs
        elif split == "train":
            preprocessing_kwargs = self.train_preprocessing_kwargs
        else:
            preprocessing_kwargs = self.general_preprocessing_kwargs

        # The tokenizer changes after it's used with some arguments (e.g. max_length, truncation),
        # therefore we make a copy in order to have the same fingerprint when using .map.
        # This is cheaper than mapping when we can already load from the cache.
        # TODO: Open a github issue?
        for key, value in preprocessing_kwargs.items():
            if isinstance(value, PreTrainedTokenizerBase):
                preprocessing_kwargs[key] = deepcopy(value)

        preprocessing_kwargs["split"] = split
        return preprocessing_kwargs

    def _get_key_mapping(self, split):
        if split == "test":
            return self.test_key_mapping
        elif split == "validation":
            return self.val_key_mapping
        elif split == "train":
            return self.train_key_mapping
        return self.general_key_mapping

    @property
    def general_preprocessing_kwargs(self):
        raise NotImplementedError

    @property
    def general_key_mapping(self):
        raise NotImplementedError

    @property
    def general_preprocessing(self):
        func = find_in_class(type(self), "general_preprocessing", ignore_klass=HfDataModule)
        if func is not None:
            return func
        else:
            raise NotImplementedError

    # If not overriden, fall back
    @property
    def train_preprocessing_kwargs(self):
        return self.general_preprocessing_kwargs

    @property
    def train_key_mapping(self):
        return self.general_key_mapping

    @property
    def train_preprocessing(self):
        func = find_in_class(type(self), "train_preprocessing", ignore_klass=HfDataModule)
        return func if func is not None else self.general_preprocessing

    @property
    def val_preprocessing_kwargs(self):
        return self.train_preprocessing_kwargs

    @property
    def val_key_mapping(self):
        return self.train_key_mapping

    @property
    def val_preprocessing(self):
        func = find_in_class(type(self), "val_preprocessing", ignore_klass=HfDataModule)
        return func if func is not None else self.train_preprocessing

    @property
    def test_preprocessing_kwargs(self):
        return self.val_preprocessing_kwargs

    @property
    def test_key_mapping(self):
        return self.val_key_mapping

    @property
    def test_preprocessing(self):
        func = find_in_class(type(self), "test_preprocessing", ignore_klass=HfDataModule)
        return func if func is not None else self.val_preprocessing

    def prepare_example(
        self,
        example: Dict[str, Any],
        split="test",
        *,
        return_tensors=True,
        return_dataset_helper=True,
        preprocessing_kwargs: Optional[Dict[str, Any]] = None,
        collator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if preprocessing_kwargs is None:
            preprocessing_kwargs = {}
        if collator_kwargs is None:
            collator_kwargs = {}

        return self.prepare_examples(
            [example],
            split=split,
            return_tensors=return_tensors,
            return_dataset_helper=return_dataset_helper,
            preprocessing_kwargs=preprocessing_kwargs,
            collator_kwargs=collator_kwargs,
        )

    def prepare_examples(
        self,
        examples: Union[Dataset, Dict[str, Any], List[Dict[str, Any]]],
        split="test",
        *,
        return_tensors=True,
        return_dataset_helper=True,
        preprocessing_kwargs: Optional[Dict[str, Any]] = None,
        collator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if preprocessing_kwargs is None:
            preprocessing_kwargs = {}
        if collator_kwargs is None:
            collator_kwargs = {}

        if isinstance(examples, list):
            examples_dict = defaultdict(list)
            for example in examples:
                for key, value in example.items():
                    examples_dict[key].append(value)
            examples = Dataset.from_dict(examples_dict)
        elif isinstance(examples, dict):
            examples = Dataset.from_dict(examples)
        if not isinstance(examples, Dataset):
            raise ValueError(f"examples must be of type datasets.Dataset, got {type(examples)}")

        dataset_container = HfDatasetContainer(dataset_and_split=(examples, split))
        dataset_container.setup()
        examples = dataset_container.ready_datasets[split]

        # TODO: Not sure it should be permanent, but currently it's a workaround for overriding dataset_name for streamlit prediction:
        preprocessing_kwargs = {**self._get_preprocessing_kwargs(split), **preprocessing_kwargs}

        preprocessed_examples = examples.map(
            self._get_preprocessing_fn(split),
            fn_kwargs=preprocessing_kwargs,
            num_proc=max(1, self._map_num_proc),
            batched=True,
            remove_columns=examples.column_names,
            load_from_cache_file=self._use_preprocessing_cache,
            with_indices=True,
            keep_in_memory=True,
        )
        preprocessed_examples = preprocessed_examples.add_column(
            "pidx", [i for i in range(len(preprocessed_examples))]
        )
        result = preprocessed_examples

        if return_tensors:
            tensorized_examples = self.get_collate_fn(split, **collator_kwargs)(preprocessed_examples)
            result = tensorized_examples

        if return_dataset_helper:
            dataset_helper = DatasetHelper(examples, preprocessed_examples)
            return result, dataset_helper
        else:
            return result
