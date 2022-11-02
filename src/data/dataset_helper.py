from typing import Optional, List, Union
from dataclasses import dataclass
from collections import defaultdict

from datasets import Dataset

from src.data.dataset_containers import BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer


# TODO: Only suitable for HF currently. Need to allow more straightforward access with DatasetContainer
@dataclass
class DatasetHelper:
    split: str
    dataset_container: Union[BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer]
    preprocessed_dataset: Optional[Dataset] = None

    def __post_init__(self):
        self._mappings_per_source = {}
        if isinstance(self.dataset_container, MultiDatasetContainer):
            ids = self.dataset_container.concatenated_datasets[self.split]["id"]
            pidx_to_id = [ids[idx] for idx in self.preprocessed_dataset["idx"]]

            for source_container_name in self.dataset_container.keys():
                raw_dataset = self.get_raw_dataset(source_container_name)
                if raw_dataset is not None:
                    self._mappings_per_source[source_container_name] = _prepare_mappings(
                        raw_dataset,
                        self.preprocessed_dataset,
                        pidx_to_id=pidx_to_id,
                        source_container_name=source_container_name,
                    )
        else:
            raw_dataset = self.get_raw_dataset()
            if raw_dataset is not None:
                self._mappings_per_source[None] = _prepare_mappings(raw_dataset, self.preprocessed_dataset)

    def pidx_to_idx(self, pidx):
        return self._mappings_per_source[self.preprocessed_dataset[pidx].get("source_container_name")]["pidx_to_idx"][
            pidx
        ]

    def get_raw_examples(self, pidxs):
        return [
            self.get_raw_dataset(self.preprocessed_dataset[pidx].get("source_container_name"))[self.pidx_to_idx(pidx)]
            for pidx in pidxs
        ]

    def get_ready_examples(self, pidxs):
        return [
            self.get_ready_dataset(self.preprocessed_dataset[pidx].get("source_container_name"))[
                self.pidx_to_idx(pidx)
            ]
            for pidx in pidxs
        ]

    def get_example_by_id(self, id_: str, source_container_name=None):
        id_to_idx = self._mappings_per_source[source_container_name]["id_to_idx"]
        return self.get_raw_dataset(source_container_name)[id_to_idx[id_]]

    def get_examples_by_ids(self, ids: List[str], source_container_name=None):
        id_to_idx = self._mappings_per_source[source_container_name]["id_to_idx"]
        return self.get_raw_dataset(source_container_name)[[id_to_idx[id_] for id_ in ids]]

    def get_preprocessed_examples(self, pidxs):
        return self.preprocessed_dataset[pidxs]

    def get_raw_dataset(self, source_container_name=None):
        return _get_x_dataset("raw", self.dataset_container, self.split, source_container_name)

    def get_ready_dataset(self, source_container_name=None):
        return _get_x_dataset("ready", self.dataset_container, self.split, source_container_name)

    def get_dataset_name(self, source_container_name=None):
        return _get_dataset_name(self.dataset_container, source_container_name)


def _get_x_dataset(x, dataset_container, split, source_container_name=None):
    if isinstance(dataset_container, ComplexDatasetContainer):
        x_dataset = _get_x_dataset(x, dataset_container[dataset_container.main_key], split)
    elif isinstance(dataset_container, MultiDatasetContainer):
        x_dataset = _get_x_dataset(x, dataset_container[source_container_name], split)
    else:
        x_datasets = getattr(dataset_container, f"{x}_datasets")
        x_dataset = x_datasets[split] if split in x_datasets else None
    return x_dataset


def _get_dataset_name(dataset_container, source_container_name=None):
    if isinstance(dataset_container, ComplexDatasetContainer):
        dataset_name = _get_dataset_name(dataset_container[dataset_container.main_key])
    elif isinstance(dataset_container, MultiDatasetContainer):
        dataset_name = _get_dataset_name(dataset_container[source_container_name])
    else:
        dataset_name = dataset_container.dataset_name
    return dataset_name


def _prepare_mappings(raw_dataset, preprocessed_dataset, pidx_to_id=None, source_container_name=None):
    id_to_idx = {id_: idx for idx, id_ in enumerate(raw_dataset["id"])}
    mappings = {"id_to_idx": id_to_idx}

    if preprocessed_dataset is not None:
        if pidx_to_id is None:
            mappings["pidx_to_idx"] = {pidx: idx for pidx, idx in enumerate(preprocessed_dataset["idx"])}
        else:
            source_container_names = (
                preprocessed_dataset["source_container_name"]
                if "source_container_name" in preprocessed_dataset.column_names
                else defaultdict(lambda: None)
            )
            mappings["pidx_to_idx"] = {
                pidx: id_to_idx[pidx_to_id[pidx]]
                for pidx in range(len(preprocessed_dataset))
                if source_container_names[pidx] == source_container_name
            }

    return mappings
