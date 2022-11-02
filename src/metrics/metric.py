from typing import Any, Callable, Optional
from collections import defaultdict
import functools

import torch
from torchmetrics import Metric as TorchMetric

from pytorch_lightning.trainer.supporters import TensorRunningAccum

from src.metrics import Result


class Metric(TorchMetric):
    """Improvements compared to torchmetrics.

    Main differences:
    - forward returns the accumulated result
    - option to get local result, without syncing
    - return names alongside the values
    """

    def __init__(
        self, name: str, prefix: Optional[str] = None, use_full_name: bool = True, window_length: Optional[int] = None
    ):
        super().__init__()
        self._prefix = prefix
        self._name = name
        self._use_full_name = use_full_name

        self._computed_per_sync_value = {key: None for key in [str(True), str(False)]}

        self._reset_kwargs = {}
        self._running_accums_per_sync_value = None
        self.window_length = window_length

    @property
    def window_length(self) -> Optional[int]:
        return self._window_length

    @window_length.setter
    def window_length(self, window_length: Optional[int] = None):
        self._window_length = window_length
        self._reset_running_accums()

    def _reset_running_accums(self):
        self._ignore_window = False
        if self._window_length is not None:
            if self._running_accums_per_sync_value is None:
                self._running_accums_per_sync_value = {
                    key: defaultdict(lambda: TensorRunningAccum(window_length=self._window_length))
                    for key in [str(True), str(False)]
                }
            else:
                for key in self._running_accums_per_sync_value.keys():
                    for k in self._running_accums_per_sync_value[key].keys():
                        self._running_accums_per_sync_value[key][k].reset(self._window_length)
        else:
            self._running_accums_per_sync_value = None

    @property
    def prefix(self):
        return self._prefix

    @property
    def name(self):
        return self._name

    def _wrap_update(self, update: Callable) -> Callable:
        update = super()._wrap_update(update)

        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            for key in [str(True), str(False)]:
                self._computed_per_sync_value[key] = None
            with torch.no_grad():
                return update(*args, **kwargs)

        return wrapped_func

    def _wrap_compute(self, compute: Callable) -> Callable:
        compute = super()._wrap_compute(compute)

        @functools.wraps(compute)
        def wrapped_func(*args: Any, sync=True, **kwargs: Any) -> Any:
            self._computed = self._computed_per_sync_value.get(str(sync))
            computation_needed = self._computed is None

            self._to_sync = sync
            computed = compute(*args, **kwargs)
            self._computed_per_sync_value[str(sync)] = computed
            self._computed = None

            single_metric = False
            if not isinstance(computed, dict):
                computed = {self.name: computed}
                single_metric = True

            if self._window_length is not None:
                if computation_needed:
                    for k, v in computed.items():
                        self._running_accums_per_sync_value[str(sync)][k].append(
                            torch.tensor(v) if not isinstance(v, torch.Tensor) else v
                        )
                computed = {k: self._running_accums_per_sync_value[str(sync)][k].mean() for k in computed.keys()}

            computed = {k: Result(log_name=self._get_log_name(k, single_metric), value=v) for k, v in computed.items()}

            return computed

        return wrapped_func

    def reset(self) -> None:
        super().reset()

        for key in [str(True), str(False)]:
            self._computed_per_sync_value[key] = None

        if self._reset_kwargs.get("reset_running_accums", True):
            self._reset_running_accums()

        self._reset_kwargs = {}

    def forward(self, *args, sync=True, **kwargs):
        if self.window_length is not None:
            self._reset_kwargs = {"reset_running_accums": False}
            self.reset()

        self.update(*args, **kwargs)
        return self.compute(sync=sync)

    def _get_log_name(self, metric_name, single_metric):
        slashed_prefix = f"{self.prefix}/" if self.prefix is not None and len(self.prefix) > 0 else ""
        if single_metric:
            return f"{slashed_prefix}{self.name}"
        else:
            if self._use_full_name:
                return f"{slashed_prefix}{self.name}_{metric_name}"
            else:
                return f"{slashed_prefix}{metric_name}"
