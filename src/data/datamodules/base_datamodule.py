from typing import Dict, Any, Optional, Union, List
from collections.abc import Mapping
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from pytorch_lightning.utilities import move_data_to_device
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only

from src.data.dataset_containers import (
    BaseDatasetContainer,
    ComplexDatasetContainer,
    MultiDatasetContainer,
    MultiDatasetContainer,
)
from utils.dot_notation import get_dot, set_dot

import logging

logger = logging.getLogger(__name__)


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_container: Union[BaseDatasetContainer, ComplexDatasetContainer, MultiDatasetContainer] = None,
        objective_format: Optional[str] = None,
        pin_memory: bool = True,
        dataloader_num_workers: int = 0,
        artifacts_dir: str = None,
        batch_size: int = 1,
        val_batch_size: Optional[int] = None,
        shuffle_training: bool = True,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._has_prepare_data_run = False
        self._has_setup_run = False

        self.dataset_container = dataset_container
        self._objective_format = objective_format

        self.batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._shuffle_training = shuffle_training
        self.seed = seed

        self.pin_memory = pin_memory
        self.dataloader_num_workers = dataloader_num_workers
        self.artifacts_dir = artifacts_dir

        # I would prefer passing it in the constructor,
        # but dataclasses typing in it makes it problematic for OmegaConf
        self._model_dict = None

        # Allow usage of some methods even without a PL trainer
        trainer = SimpleNamespace()
        trainer.world_size = 1
        trainer.max_epochs = 1
        trainer.current_epoch = 0
        trainer.global_rank = 0
        self.trainer = trainer

    def on_before_dataset_container_preprocessing(self) -> None:
        pass

    def dataset_container_preprocessing(self) -> None:
        pass

    def on_after_dataset_container_preprocessing(self) -> None:
        pass

    @property
    def model_kwargs(self) -> Dict[str, Any]:
        return {**self._model_kwargs, **self._fresh_model_kwargs}

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        return {}

    @property
    def _fresh_model_kwargs(self) -> Dict[str, Any]:
        return {}

    def prepare_data(self):
        super().prepare_data()
        if self._has_prepare_data_run:
            return
        else:
            self._has_prepare_data_run = True

    def setup(self, stage: Optional[str] = None, model_dict: Dict[str, Any] = None):
        super().setup(stage)
        if self._has_setup_run:
            return
        else:
            self._has_setup_run = True

        if model_dict is not None:
            hparams_dict = model_dict["hyper_parameters"]
            for key in self._model_kwargs.keys():
                if key in hparams_dict:
                    if hasattr(self, f"_{key}"):
                        setattr(self, f"_{key}", hparams_dict[key])
                    else:
                        setattr(self, key, hparams_dict[key])

        if self.dataset_container is not None:
            self.dataset_container.setup()

            self.on_before_dataset_container_preprocessing()
            self.dataset_container_preprocessing()
            self.on_after_dataset_container_preprocessing()

    def get_collate_fn(self, split=None):
        return None  # will use the default collate_fn

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int = 0) -> Any:
        if not (isinstance(batch, Mapping) and "tensor_keys" in batch):
            return move_data_to_device(batch, device)

        for k in batch["tensor_keys"]:
            exists, value = get_dot(batch, k)
            if exists:
                set_dot(batch, k, move_data_to_device(value, device))
        return batch

    def train_dataloader(self):
        if "train" not in self.preprocessed_datasets:
            return None

        shuffle = self._shuffle_training

        # https://stackoverflow.com/questions/59314174/is-pytorch-dataloader-iteration-order-stable
        dataset = self.preprocessed_datasets["train"]

        if shuffle:
            if rank_zero_only.rank == 0:
                logger.info(f"Reloading train dataloader with epoch {self.trainer.current_epoch}")
            # TODO: Replace all rank_zero_info with if rank_zero_only.rank and logger.info.
            # Reason: the module name in the log is always pytorch_lightning.utilities.distributed

            if self.trainer is not None:
                g = torch.Generator()
                g.manual_seed(self.seed + self.trainer.current_epoch)
            else:
                g = None
            permutation = torch.randperm(len(dataset), generator=g).tolist()
            shuffled_dataset = torch.utils.data.Subset(dataset, permutation)
        else:
            shuffled_dataset = dataset
        shuffle = False

        return DataLoader(
            shuffled_dataset,  # TODO: Change back to self.preprocessed_datasets["train"] once fault tolerant training is fixed
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.get_collate_fn("train"),
            shuffle=shuffle,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        if "validation" not in self.preprocessed_datasets:
            return None

        return DataLoader(
            self.preprocessed_datasets["validation"],
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.get_collate_fn("validation"),
        )

    def test_dataloader(self):
        if "test" not in self.preprocessed_datasets:
            return None

        return DataLoader(
            self.preprocessed_datasets["test"],
            batch_size=self.val_batch_size,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.get_collate_fn("test"),
        )

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass

    @property
    def val_batch_size(self):
        if self._val_batch_size is not None:
            return self._val_batch_size

        return self.batch_size

    def prepare_example(self, example: Dict[str, Any], split="test", return_tensors=True):
        raise NotImplementedError

    def prepare_examples(
        self,
        examples: Union[Dict[str, Any], List[Dict[str, Any]]],
        split="test",
        return_tensors=True,
    ):
        raise NotImplementedError

    def get_batch_sampler(self, klass, key, shuffle_per_epoch, dataset=None, **kwargs):
        if dataset is None:
            dataset = self.preprocessed_datasets[key]
        trainer = self.trainer

        if trainer.world_size <= 1:
            batch_sampler = klass(
                dataset=dataset,
                batch_size=self.batch_size,
                seed=self.seed,
                epoch=trainer.current_epoch,
                shuffle_per_epoch=shuffle_per_epoch,
                **kwargs,
            )
        else:
            batch_sampler = type("Custom", (klass, SpecializedDistributedSampler), {})(
                dataset=dataset,
                batch_size=self.batch_size,
                seed=self.seed,
                shuffle_per_epoch=shuffle_per_epoch,
                num_replicas=trainer.world_size,
                rank=trainer.global_rank,
                **kwargs,
            )
        return batch_sampler


class SpecializedDistributedSampler(DistributedSampler):
    # Copied and adapted from PyTorch DistributedSampler.
    def __init__(
        self,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        super().__init__(self, epoch=0, num_replicas=num_replicas, rank=rank, **kwargs)
