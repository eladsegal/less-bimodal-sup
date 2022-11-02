# Process multiple containers to a new one.
# Each of the datasets will have the same columns, and an additional source column will be added with the name of the dataset
# The purpose is to make it act like a single dataset

from typing import Dict, Any, List
from collections import defaultdict
from datasets import concatenate_datasets

import logging

logger = logging.getLogger(__name__)


class MultiDatasetContainer:
    def __init__(
        self,
        additional_kwargs: Dict[str, Any] = None,
        order: List[str] = None,
        **kwargs,
    ):
        super().__init__()

        self._internal_dict = dict(**kwargs) if order is None else {key: kwargs[key] for key in order}
        self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}

        from src.data.dataset_containers import ComplexDatasetContainer

        for value in self._internal_dict.values():
            if not isinstance(value, ComplexDatasetContainer):
                raise ValueError(f"Expected ComplexDatasetContainer, got {type(value)}")
            if "language" not in value or "vision" not in value or len(value) != 2:
                raise ValueError(f"Expected ComplexDatasetContainer to have language and vision datasets only")

    def __getitem__(self, key):
        return self._internal_dict[key]

    def __iter__(self):
        return iter(self._internal_dict)

    def __len__(self):
        return len(self._internal_dict)

    def __contains__(self, key):
        return key in self._internal_dict

    def keys(self):
        return self._internal_dict.keys()

    def items(self):
        return self._internal_dict.items()

    def values(self):
        return self._internal_dict.values()

    def setup(self, kwargs_per_container: Dict[str, Any] = None):
        from src.data.dataset_containers import HfDatasetContainer

        kwargs_per_container = kwargs_per_container if kwargs_per_container is not None else {}
        for key, dataset_container in self.items():
            kwargs = kwargs_per_container.get(key, {})
            if isinstance(dataset_container, HfDatasetContainer):
                kwargs["source_container_name"] = key
            dataset_container.setup(additional_kwargs=self.additional_kwargs, **kwargs)

        self.concatenated_datasets = self._concatenate_datasets()

    def _concatenate_datasets(self):
        logger.info("Concatenating datasets")

        datasets_per_split = defaultdict(dict)
        for key, dataset_container in self.items():
            for split, dataset in dataset_container["language"].ready_datasets.items():
                datasets_per_split[split][key] = dataset

        concatenated_dataset_per_split = {}
        for split, datasets in datasets_per_split.items():
            source_container_names = []
            shared_columns = None
            for key, dataset in datasets.items():
                source_container_names.extend([key] * len(dataset))
                shared_columns = (
                    set(dataset.column_names) if shared_columns is None else shared_columns & set(dataset.column_names)
                )

            for key in datasets.keys():
                datasets[key] = datasets[key].remove_columns(set(datasets[key].column_names) - shared_columns)

            concatenated_dataset = concatenate_datasets(datasets.values())
            concatenated_dataset = concatenated_dataset.add_column("source_container_name", source_container_names)
            concatenated_dataset_per_split[split] = concatenated_dataset

        return concatenated_dataset_per_split
