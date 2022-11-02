from typing import Any, Dict, List, Optional

import random
import time

import torch

from datasets.load import load_metric

from src.metrics import Metric

METRICS_INFO_PRESETS = {
    "squad": [{"name": "f1", "default_value": 0.0}, {"name": "exact_match", "default_value": 0.0}],
}


class HfMetric(Metric):
    def __init__(
        self, load_metric_kwargs: Dict[str, Any], metrics_info: Optional[List[Dict[str, Any]]] = None, **kwargs
    ):
        super().__init__(**kwargs)

        # experiment_id is used to prevent sync across processes, as we take care of it with src.metrics.metric
        self._hf_metric = load_metric(
            **load_metric_kwargs,
            keep_in_memory=True,
        )

        if metrics_info is None:
            metrics_info = METRICS_INFO_PRESETS[load_metric_kwargs["path"]]
        self._metrics_info = metrics_info  # We need to specify the outputs of the metric and their initial value
        for metric_info in metrics_info:
            self.add_state(
                metric_info["name"], default=torch.tensor(metric_info["default_value"]), dist_reduce_fx="sum"
            )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, references, **kwargs):
        count = len(predictions)
        metrics = self._hf_metric.compute(predictions=predictions, references=references, **kwargs)
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
