from typing import Any, Dict, List

import inspect
from copy import deepcopy

import torch

from datasets.load import load_metric

from src.metrics import Metric

METRICS_INFO_PRESETS = {}

# TODO: Wasn't tested at all
class TorchMetricsWrapper(Metric):
    def __init__(self, torchmetric_klass, metrics_info: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)

        clean_kwargs = deepcopy(kwargs)
        for name in inspect.signature(Metric.__init__).parameters.keys():
            clean_kwargs.pop(name, None)

        self._torchmetric = torchmetric_klass(**clean_kwargs)
        self._torchmetric._to_sync = False

        if metrics_info is None:
            metrics_info = METRICS_INFO_PRESETS[torchmetric_klass]
        self._metrics_info = metrics_info  # We need to specify the outputs of the metric and their initial value
        for metric_info in metrics_info:
            self.add_state(
                metric_info["name"], default=torch.tensor(metric_info["default_value"]), dist_reduce_fx="sum"
            )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, *args, **kwargs):
        count = len(args[0] if len(args) > 0 else kwargs["preds"])

        self._torchmetric.reset()
        self._torchmetric.update(*args, **kwargs)
        metrics = self.compute()

        for metric_info in self._metrics_info:
            name = metric_info["name"]
            new_value = getattr(self, name) + (metrics[name] * count)
            setattr(self, name, new_value)
        self.count += count

    def compute(self):
        results = {}
        for metric_info in self._metrics_info:
            name = metric_info["name"]
            count = self.count.item()
            results[name] = getattr(self, name).item() / count if count > 0 else 0
        return results
