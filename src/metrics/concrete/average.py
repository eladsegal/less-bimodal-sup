from typing import Union, Optional
import torch
from src.metrics import Metric


class Average(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        value: Union[float, int, torch.Tensor],
        count: Optional[Union[float, int, torch.Tensor]] = None,
        already_averaged=False,
    ):
        value_is_not_averaged = isinstance(value, torch.Tensor) and value.dim() > 0 and len(value) > 1
        if value_is_not_averaged:
            assert value_is_not_averaged != already_averaged

        if count is None:
            count = len(value) if value_is_not_averaged else 1
        elif value_is_not_averaged:
            assert count == len(value)

        if already_averaged:
            self.value += value * count
        else:
            if isinstance(value, torch.Tensor):
                value = torch.sum(value)
            self.value += value
        self.count += count

    def compute(self):
        count = self.count.item()
        return self.value.item() / count if count > 0 else 0
