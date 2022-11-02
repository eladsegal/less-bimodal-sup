from typing import Dict, List, Union, Optional

import os
import json
from pathlib import Path

import torch

from src.data.datamodules.base_datamodule import BaseDataModule
from utils.general import resolve_relative_paths

import logging

logger = logging.getLogger(__name__)


class AnswersLabelsMixin(BaseDataModule):
    def __init__(
        self,
        ans2label_current_path: str,  # Current means it is matching the current dataset
        ans2label_init: Union[str, Dict[str, int]] = None,
        merge_init_labels=False,
        pad_to_multiple_of: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pad_to_multiple_of = pad_to_multiple_of

        self._handle_answers_labels_init(ans2label_init)
        self._merge_init_labels = merge_init_labels
        self._classifier_classes_reorder = None

        self._handle_answers_labels_current(ans2label_current_path)
        self._ans2label = None

    def on_before_dataset_container_preprocessing(self):
        super().on_before_dataset_container_preprocessing()

        for variable_name, value in self._get_answers_labels().items():
            setattr(self, f"_{variable_name}", value)
        if len(self._ans2label_init) > 0:
            self._classifier_classes_reorder = {
                self._ans2label_init[ans]: self._ans2label[ans]
                for ans in self._ans2label
                if ans in self._ans2label_init
            }
        if self._pad_to_multiple_of is not None:
            diff = self._pad_to_multiple_of - len(self._ans2label) % self._pad_to_multiple_of
            for i in range(diff):
                self._ans2label[f"[PAD_{i}]"] = len(self._ans2label)
        logger.info(f"Number of answer labels: {len(self._ans2label)}")

    @property
    def _model_kwargs(self):
        model_kwargs = super()._model_kwargs
        model_kwargs["ans2label"] = self._ans2label

        return model_kwargs

    @property
    def _fresh_model_kwargs(self):
        fresh_model_kwargs = super()._fresh_model_kwargs

        if self._classifier_classes_reorder is not None:
            fresh_model_kwargs["classifier_classes_reorder"] = self._classifier_classes_reorder
        return fresh_model_kwargs

    def _handle_answers_labels_init(self, ans2label_init):
        def get_from_path(path, name: Optional[str] = None):
            obj = None
            if Path(path).suffix == ".ckpt":
                checkpoint = torch.load(path, map_location="cpu")
                if name in checkpoint["hyper_parameters"]:
                    obj = checkpoint["hyper_parameters"][name]
            else:
                if not os.path.isfile(path) and Path(path).suffix != ".json":
                    path = os.path.join(path, f"{name}.json")
                with open(path, mode="r") as f:
                    obj = json.load(f)
            if obj is None:
                raise Exception(f"Couldn't find {name} in {path}")
            return obj

        if ans2label_init is not None and isinstance(ans2label_init, str):
            ans2label_init = get_from_path(ans2label_init, "ans2label")

        self._ans2label_init = ans2label_init
        if self._ans2label_init is None:
            self._ans2label_init = {}

    def _handle_answers_labels_current(self, ans2label_current_path):
        ans2label_current_path = resolve_relative_paths(ans2label_current_path)
        with open(ans2label_current_path, mode="r") as f:
            self._ans2label_current = json.load(f)

    def _get_answers_labels(self):
        ans2label_current = self._ans2label_current

        if self._merge_init_labels:
            base_ans2label = dict(self._ans2label_init)
        else:
            base_ans2label = {}

        ans2label = {
            **base_ans2label,
            **{
                k: len(base_ans2label) + i
                for i, k in enumerate(x for x in ans2label_current.keys() if x not in base_ans2label)
            },
        }

        # This fixes self to use the right labels so the same instances would be dropped, independently of the labels init
        ans2label_current = {k: ans2label[k] for k in ans2label_current.keys()}

        output_dict = {
            "ans2label": ans2label,
            "ans2label_current": ans2label_current,
        }
        return output_dict
