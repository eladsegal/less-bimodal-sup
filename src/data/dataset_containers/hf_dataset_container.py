from typing import Dict, Any, Optional, Tuple, List

from datasets import load_dataset, Dataset, DatasetDict
from utils import patch

from src.data.dataset_containers import BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer
from src.data.hf_datasets.mappers import Mapper
from src.data.hf_datasets.mappers.imagenet_to_vqa_mapper import ImageNetToVqaMapper
from src.data.hf_datasets.mappers.conceptual_captions_labels_to_pretraining_mapper import (
    ConceptualCaptionsLabelsToPretrainingMapper,
)
from src.data.hf_datasets.mappers.imagenet_to_pretraining_mapper import (
    ImageNetToPretrainingMapper,
)
from utils.general import resolve_relative_paths


MAPPING_PRESETS = {
    ("vqa.default", "src.data.datamodules.finetuning_datamodule.FinetuningDataModule"): {
        "column_mapping": {
            "question": "text_input",
        },
        "mapper": None,
    },
    ("gqa.default", "src.data.datamodules.finetuning_datamodule.FinetuningDataModule"): {
        "column_mapping": {
            "question": "text_input",
        },
        "mapper": None,
    },
    ("nlvr2.default", "src.data.datamodules.finetuning_datamodule.FinetuningDataModule"): {
        "column_mapping": {
            "sentence": "text_input",
        },
        "mapper": None,
    },
    ("image_net_from_files.default", "src.data.datamodules.pretraining_datamodule.PretrainingDataModule",): {
        "column_mapping": None,
        "mapper": ImageNetToPretrainingMapper(),
    },
    ("conceptual_captions_labels.default", "src.data.datamodules.pretraining_datamodule.PretrainingDataModule",): {
        "column_mapping": None,
        "mapper": ConceptualCaptionsLabelsToPretrainingMapper(),
    },
    ("image_net_from_files.default", "src.data.datamodules.finetuning_datamodule.FinetuningDataModule"): {
        "column_mapping": None,
        "mapper": ImageNetToVqaMapper(),
    },
    ("wikipedia.20200501.en", "src.data.datamodules.pretraining_datamodule.PretrainingDataModule"): {
        "column_mapping": {
            "text": "text_input",
        },
        "mapper": None,
    },
}


class HfDatasetContainer(BaseDatasetContainer):
    def __init__(
        self,
        *,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        dataset_and_split: Optional[Tuple[Dataset, str]] = None,
        dataset_dict: Optional[Dataset] = None,
        id_column_name: Optional[str] = None,
        force_id_column: bool = True,
        use_idx_as_id: bool = False,
        column_mapping: Optional[Dict[str, str]] = None,
        mapper: Optional[Mapper] = None,
        target_datamodule: Optional[str] = None,
        map_num_proc: int = 1,
        use_mapping_cache: bool = True,
        reduced_train_size: Optional[float] = None,
        mapper_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if sum([load_dataset_kwargs is not None, dataset_and_split is not None, dataset_dict is not None]) != 1:
            raise ValueError("Only one of load_dataset_kwargs, dataset and dataset_dict can be specified")
        self._load_dataset_kwargs = load_dataset_kwargs
        self._dataset_and_split = dataset_and_split
        self._dataset_dict = dataset_dict

        if id_column_name is not None and use_idx_as_id:
            raise ValueError("id_column_name and use_idx_as_id cannot be used together")
        self._id_column_name = id_column_name
        self._force_id_column = force_id_column
        self._use_idx_as_id = use_idx_as_id

        self._column_mapping = column_mapping
        self._mapper = mapper
        self._target_datamodule = target_datamodule

        self._map_num_proc = map_num_proc
        self._use_mapping_cache = use_mapping_cache
        self._reduced_train_size = reduced_train_size

        self._mapper_kwargs = mapper_kwargs if mapper_kwargs is not None else {}

        if load_dataset_kwargs is not None:
            if load_dataset_kwargs["path"].endswith(".py"):
                load_dataset_kwargs["path"] = resolve_relative_paths(load_dataset_kwargs["path"])

            for kwarg in ["data_files", "data_dir"]:
                load_dataset_kwargs[kwarg] = (
                    resolve_relative_paths(load_dataset_kwargs[kwarg])
                    if load_dataset_kwargs.get(kwarg) is not None
                    else None
                )

    def _setup(
        self,
        parent_container: Optional[ComplexDatasetContainer] = None,
        additional_kwargs: Dict[str, Any] = None,
    ):
        if parent_container is not None and self._using_default_seed:
            self.seed = parent_container.seed

        map_num_proc = additional_kwargs["map_num_proc"] if "map_num_proc" in additional_kwargs else self._map_num_proc
        target_datamodule = (
            additional_kwargs["target_datamodule"]
            if "target_datamodule" in additional_kwargs
            else self._target_datamodule
        )

        if self._load_dataset_kwargs is not None:
            load_dataset_kwargs = self._load_dataset_kwargs
            split = load_dataset_kwargs.get("split")
            custom_split = load_dataset_kwargs.get("custom_split")
            if custom_split is not None:
                split = custom_split
                # this is needed because advanced split are not supported when passing a split dict to load_dataset
                self.raw_datasets = DatasetDict()
                for split_key, split_value in split.items():
                    load_dataset_kwargs_copy = dict(load_dataset_kwargs)
                    load_dataset_kwargs_copy.pop("split", None)
                    load_dataset_kwargs_copy.pop("custom_split", None)
                    self.raw_datasets[split_key] = load_dataset(**load_dataset_kwargs_copy, split=split_value)
                    load_dataset_kwargs.pop("download_mode", None)
            elif isinstance(split, str):
                self.raw_datasets = DatasetDict()
                self.raw_datasets[split] = load_dataset(**load_dataset_kwargs)
            else:
                self.raw_datasets = load_dataset(**load_dataset_kwargs)
        elif self._dataset_and_split is not None:
            self.raw_datasets = DatasetDict()
            self.raw_datasets[self._dataset_and_split[1]] = self._dataset_and_split[0]
        elif self._dataset_dict is not None:
            self.raw_datasets = self._dataset_dict

        assert (
            len(set(self.raw_datasets.keys()) & set(["train", "validation", "test"])) > 0
        ), 'The loaded dataset must have at least one of these splits: ["train", "validation", "test"]'

        if (
            "train" in self.raw_datasets
            and self._reduced_train_size is not None
            and self._reduced_train_size != 1
            and self._reduced_train_size != len(self.raw_datasets["train"])
        ):
            import torch

            if 0 <= self._reduced_train_size <= 1:
                train_number_of_examples = int(self._reduced_train_size * len(self.raw_datasets["train"]))
            elif self._reduced_train_size > 1 and self._reduced_train_size % 1.0 == 0:
                train_number_of_examples = int(self._reduced_train_size)
            else:
                assert False, f"Invalid reduced_train_size={self._reduced_train_size}"

            g = torch.Generator()
            g.manual_seed(self.seed)
            if self.dataset_name != "nlvr2.default":
                indices = torch.randperm(len(self.raw_datasets["train"]), generator=g)[
                    :train_number_of_examples
                ]  # https://github.com/pytorch/pytorch/issues/16897
                self.raw_datasets["train"] = self.raw_datasets["train"].select(indices)
            else:
                set_id_to_indices = {k: [] for k in self.raw_datasets["train"]["set_id"]}
                ordered_set_ids = list(set_id_to_indices.keys())

                for i, set_id in enumerate(self.raw_datasets["train"]["set_id"]):
                    set_id_to_indices[set_id].append(i)

                set_indices = torch.randperm(len(set_id_to_indices), generator=g)
                indices = []
                for set_index in set_indices:
                    if len(indices) >= train_number_of_examples:
                        break
                    indices.extend(set_id_to_indices[ordered_set_ids[set_index]])
                self.raw_datasets["train"] = self.raw_datasets["train"].select(indices)

        if self._force_id_column:
            # Take care of the id column
            for split_key in self.raw_datasets.keys():
                self.raw_datasets[split_key] = self._add_id_to_raw_dataset(self.raw_datasets[split_key])

        column_mapping = self._column_mapping
        mapper = self._mapper
        if column_mapping is None and mapper is None and target_datamodule is not None:
            dataset_name = self.dataset_name

            preset = MAPPING_PRESETS.get((dataset_name, target_datamodule))
            if preset is not None:
                column_mapping = preset["column_mapping"]
                mapper = preset["mapper"]

        self.ready_datasets = DatasetDict()
        for split_key in self.raw_datasets.keys():
            dataset = self.raw_datasets[split_key]
            if column_mapping is not None:
                dataset = dataset.rename_columns(column_mapping)

            if mapper is not None:
                dataset = dataset.map(
                    mapper.get_mapping_fn(split_key),
                    fn_kwargs=mapper.get_mapping_kwargs(split_key, dataset, **self._mapper_kwargs),
                    num_proc=max(1, map_num_proc),
                    batched=True,
                    remove_columns=dataset.column_names,
                    load_from_cache_file=self._use_mapping_cache,
                    with_indices=True,
                )

            self.ready_datasets[split_key] = dataset

    def _add_id_to_raw_dataset(self, raw_dataset: Dataset):
        if self._id_column_name is not None:
            if self._id_column_name != "id":
                if "id" in raw_dataset.column_names:
                    raise ValueError(
                        f"The dataset already has an id column. Please remove it before setting another id column."
                    )
                raw_dataset = raw_dataset.add_column("id", raw_dataset[self._id_column_name])
        elif self._use_idx_as_id:
            if "id" in raw_dataset.column_names:
                raise ValueError(
                    f"The dataset already has an id column. Please remove it before setting another id column."
                )
            raw_dataset = raw_dataset.add_column("id", [i for i in range(len(raw_dataset))])
        if "id" not in raw_dataset.column_names:
            raise ValueError("id column is missing. Please specify id_column_name or set use_idx_as_id=True")
        if len(set(raw_dataset["id"])) != len(raw_dataset):
            raise Exception("Values in the `id` column must be unique")
        return raw_dataset

    @property
    def dataset_name(self):
        raw_dataset = next(iter(self.raw_datasets.values()))
        dataset_name = raw_dataset.info.builder_name + "." + raw_dataset.info.config_name
        return dataset_name
