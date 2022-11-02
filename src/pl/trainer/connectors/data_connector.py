from typing import Optional, Callable

import functools

from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler

from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.trainer.states import RunningStage


# Solves https://github.com/PyTorchLightning/pytorch-lightning/issues/10170
def _wrap_resolve_sampler(_resolve_sampler: Callable) -> Callable:
    @functools.wraps(_resolve_sampler)
    def wrapped_func(self, dataloader: DataLoader, shuffle: bool, mode: Optional[RunningStage] = None) -> Sampler:
        if isinstance(dataloader.sampler, RandomSampler):
            shuffle = True
        elif isinstance(dataloader.sampler, SequentialSampler):
            shuffle = False
        else:
            rank_zero_info(f"Couldn't find original shuffle value for the dataloader, using `shuffle={shuffle}`")

        return _resolve_sampler(dataloader, shuffle, mode)

    return wrapped_func
