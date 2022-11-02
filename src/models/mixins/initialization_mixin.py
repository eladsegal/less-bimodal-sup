from typing import List, Callable

import torch

from src.pl import MyLightningModule

import logging

logger = logging.getLogger(__name__)


SKIP = ["_metrics_dict"]


class InitializationMixin(MyLightningModule):
    def __init__(self, prevent_init_weights: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._prevent_init_weights = prevent_init_weights

    def apply_excluding(self, fn: Callable[["torch.nn.Module"], None], excluding: List[str]):
        """
        Exclude top-level modules by name from apply
        """
        for name, module in self.named_children():
            if name in SKIP:
                continue
            if name in excluding:
                logger.info(f"{self.__class__.__name__}: Excluded weights initialization for {name}")
                continue
            module.apply(fn)
            logger.info(f"{self.__class__.__name__}: Initialized weights for {name}")
        return self

    def init_weights(self, init_func, excluding):
        """
        `excluding` is expected to have names of pretrained members.

        Example:
        self.init_weights(excluding=["_language_model", "_vision_model"])
        """
        # TODO: For deepspeed this is not necessary as it takes care of initialization.
        #       See `no_init_weights` in `transformers.modeling_utils`.
        if self._prevent_init_weights:
            return
        # Initialize weights
        self.apply_excluding(init_func, excluding=excluding)
