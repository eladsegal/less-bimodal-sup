from typing import Dict, Any, Optional, Union

import os
from functools import reduce

import torch
from torch.optim import Optimizer

import logging

from src.pl import MyLightningModule
from src.models.mixins import (
    MetricsMixin,
    DatasetHelperMixin,
    EvaluatorMixin,
    PredictionMixin,
    InitializationMixin,
    GradientsMixin,
)
from src.utils.instantiation import Instantiation
from utils.general import smart_open

logger = logging.getLogger(__name__)

OPTIMIZER_GROUPS_DIR = "optimizer_groups"


class BaseModel(
    MetricsMixin,
    DatasetHelperMixin,
    EvaluatorMixin,
    PredictionMixin,
    InitializationMixin,
    GradientsMixin,
    MyLightningModule,
):
    def __init__(
        self,
        optimizer: Optional[Instantiation[Optimizer]] = None,
        scheduler: Optional[Instantiation] = None,
        objective_format: Optional[str] = None,
        num_warmup_steps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        BaseModel.add_to_ignore_saving_in_kwargs("optimizer", kwargs)
        BaseModel.add_to_ignore_saving_in_kwargs("scheduler", kwargs)
        super().__init__(**kwargs)

        self._optimizer_instantiation = optimizer
        self._scheduler_instantiation = scheduler

        self._objective_format = objective_format
        self._warmup_value = num_warmup_steps

    def find_batch_size(self, batch: Dict[str, Any]) -> int:
        return len(batch["pidx"])

    def _get_optimizer_grouped_parameters(self, **optimizer_kwargs):
        optimizer_grouped_parameters = self.create_optimizer_group(
            filter(lambda n_p: n_p[1].requires_grad, self.named_parameters()),
            **optimizer_kwargs,
        )
        return optimizer_grouped_parameters

    def configure_optimizers(self):
        # optimizer
        optimizer_grouped_parameters = self._get_optimizer_grouped_parameters(**self._optimizer_instantiation.kwargs)
        optimizer = self._optimizer_instantiation(optimizer_grouped_parameters)

        files_to_upload = []
        for i, optimizer_group in enumerate(optimizer_grouped_parameters):
            params_set = set(optimizer_group["params"])
            names = [n for n, p in self.named_parameters() if p in params_set]
            file_name = optimizer.__class__.__name__ + "_" + optimizer_group.get("name", f"gp_{i}") + ".txt"
            file_path = os.path.join(OPTIMIZER_GROUPS_DIR, file_name)
            with smart_open(file_path, "w") as f:
                f.write(f"\n".join(names))
            files_to_upload.append(file_path)
        self.trainer.upload_files(files_to_upload)

        # learning rate scheduler
        if self._scheduler_instantiation is not None:
            lr_scheduler = self._scheduler_instantiation(
                optimizer=optimizer,
                trainer=self.trainer,
                pl_module=self,
            )
            output = ([optimizer], [{"scheduler": lr_scheduler, "interval": "step"}])
        else:
            output = optimizer

        return output

    @property
    def _num_warmup_steps(self) -> int:
        if self._warmup_value is None:
            return None
        if 0 < self._warmup_value < 1:
            # Convert float values to percentage of training steps to use as warmup
            return int(self._warmup_value * self.trainer.estimated_stepping_batches)
        return int(self._warmup_value)

    def create_optimizer_group(
        self, filtered_named_parameters, no_decay_suffixes=None, no_decay_classes=None, **optimizer_kwargs
    ):
        """
        Automatically create a group for params with no_decay such as bias and LayerNorm
        """
        filtered_named_parameters = list(filtered_named_parameters)
        if no_decay_suffixes is None:
            no_decay_suffixes = []
        if ".bias" not in no_decay_suffixes:
            no_decay_suffixes.append(".bias")

        if no_decay_classes is None:
            no_decay_classes = []
        if torch.nn.LayerNorm not in no_decay_classes:
            no_decay_classes.append(torch.nn.LayerNorm)
        no_decay_classes = tuple(no_decay_classes)

        decay_params = []
        no_decay_params = []
        for name, param in filtered_named_parameters:
            if (
                len(param.shape) <= 1
                or any(name.endswith(no_decay_suffix) for no_decay_suffix in no_decay_suffixes)
                or isinstance(self._get_module_from_name(name), no_decay_classes)
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = []
        optimizer_groups.append(
            {
                "params": decay_params,
                **optimizer_kwargs,
            }
        )

        no_decay_optimizer_group = {
            "params": no_decay_params,
            **{k: v for k, v in optimizer_kwargs.items() if k not in ["weight_decay", "name"]},
            "weight_decay": 0.0,
        }
        group_name = optimizer_kwargs.get("name")
        if group_name is not None:
            no_decay_optimizer_group["name"] = group_name + "_no_decay"
        optimizer_groups.append(no_decay_optimizer_group)

        return optimizer_groups

    def _get_module_from_name(self, full_name):
        names = full_name.split(".")
        result = reduce(getattr, names, self)
        if isinstance(result, torch.nn.Parameter):
            result = self._get_module_from_name(".".join(names[:-1]))
        return result
