from typing import Iterable, List, Union
import functools

from torch.optim import Optimizer
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning


class Freezer(BaseFinetuning):
    def __init__(self, fqns: List[str], train_bias_only: bool = False) -> None:
        super().__init__()
        self._fqns = fqns
        self._train_bias_only = train_bias_only

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        for fqn in self._fqns:
            for name, module in pl_module.named_modules():
                if len(name) == 0:
                    continue
                if name == fqn:
                    self.freeze(module, train_bias_only=self._train_bias_only)
                    # Switch to eval everything
                    force_eval(module)
                    break

    @staticmethod
    def freeze(
        modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True, train_bias_only: bool = False
    ) -> None:
        """Freezes the parameters of the provided modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: If True, leave the BatchNorm layers in training mode

        Returns:
            None
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and train_bn:
                BaseFinetuning.make_trainable(mod)
            else:
                # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
                for name, param in mod.named_parameters(recurse=False):
                    if train_bias_only and name == "bias":
                        continue
                    else:
                        param.requires_grad = False

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        pass


def force_eval(module):
    train_method = module.train

    @functools.wraps(train_method)
    def wrapped_func(*args, **kwargs):
        return train_method(False)

    module.train = wrapped_func
    module.eval()
