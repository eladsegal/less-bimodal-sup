from typing import Union
import torch
from src.metrics import Metric


class Sum(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value: Union[float, int, torch.Tensor]):
        self.value += value

    def compute(self):
        return self.value.item()
