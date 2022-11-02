from typing import Any, Dict, List, Callable, Optional, Union
from collections.abc import Mapping

import math

import torch
from torch.optim import Optimizer

import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import TensorRunningAccum

from src.pl import MyLightningModule

import logging

logger = logging.getLogger(__name__)


SKIP = ["_metrics_dict"]


class GradientsMixin(MyLightningModule):
    def __init__(
        self,
        adaptive_gradient_clipping: Optional[Union[bool, Dict]] = None,  # Didn't see benefits from using this yet
        auto_clip: Optional[Union[bool, Dict]] = None,  # Didn't see benefits from using this yet
        **kwargs,
    ):
        super().__init__(**kwargs)

        if auto_clip and adaptive_gradient_clipping:
            raise ValueError("Both auto_clip and adaptive_gradient_clipping are set")

        self._adaptive_gradient_clipping = None
        if adaptive_gradient_clipping is not None:
            if isinstance(adaptive_gradient_clipping, bool) and adaptive_gradient_clipping:
                self._adaptive_gradient_clipping = {}
            elif isinstance(adaptive_gradient_clipping, Mapping):
                self._adaptive_gradient_clipping = adaptive_gradient_clipping

        self._auto_clip = None
        if auto_clip is not None:
            if isinstance(auto_clip, bool) and auto_clip:
                self._auto_clip = AutoClip()
            elif isinstance(auto_clip, Mapping):
                self._auto_clip = AutoClip(**auto_clip)

        self._ENABLE_SIMPLE_GRAD_NORM = True

        self._ENABLE_DETAILED_GRAD_NORMS = False
        if self._ENABLE_DETAILED_GRAD_NORMS:
            self._norms_dict = None

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ):
        if self._adaptive_gradient_clipping is not None:
            parameters = self.trainer.precision_plugin.main_params(optimizer)
            adaptive_clip_grad(parameters, **self._adaptive_gradient_clipping)
        elif self._auto_clip is not None:
            self._auto_clip(self, optimizer)
        else:
            super().configure_gradient_clipping(optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        if self._ENABLE_DETAILED_GRAD_NORMS:
            if self._norms_dict is not None:
                for v in self._norms_dict.values():
                    v.reset()

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        if self._ENABLE_DETAILED_GRAD_NORMS:
            norm_type = float(2)

            norms_dict, all_norms = get_module_norms(
                self, norm_type=norm_type, names_for_detailed_grad_norms=self._get_names_for_detailed_grad_norms()
            )

            if all_norms:
                norms_dict[f"grad_{norm_type}_norm/total"] = torch.tensor(all_norms).norm(norm_type).item()
                norms_dict[f"grad_{norm_type}_norm/max"] = torch.tensor(all_norms).max().item()

            if norms_dict:
                if self._norms_dict is None:
                    self._norms_dict = {
                        k: TensorRunningAccum(window_length=self.trainer.log_every_n_steps) for k in norms_dict
                    }

                for k, v in norms_dict.items():
                    self._norms_dict[k].append(torch.tensor(v))

                aggregated_norms = {}
                aggregated_norms.update(
                    {
                        k: round(v.mean().item(), 4)
                        for k, v in self._norms_dict.items()
                        if k.startswith(f"grad_{norm_type}_norm/total")
                    }
                )
                aggregated_norms.update(
                    {
                        k: round(v.max().item(), 4)
                        for k, v in self._norms_dict.items()
                        if k.startswith(f"grad_{norm_type}_norm/max")
                    }
                )
                self.log_dict(aggregated_norms, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        elif self._ENABLE_SIMPLE_GRAD_NORM:
            parameters = [p for p in self.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters])).item()
            self.log("gradient_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def _get_names_for_detailed_grad_norms(self):
        return []

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)

        if self._auto_clip is not None and "_auto_clip" in checkpoint:
            self._auto_clip.grad_history = checkpoint["_auto_clip"]["grad_history"]

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)

        if self._auto_clip is not None:
            checkpoint["_auto_clip"] = {"grad_history": self._auto_clip.grad_history}


def get_module_norms(module: torch.nn.Module, norm_type=2.0, name_prefix="", names_for_detailed_grad_norms=None):
    if names_for_detailed_grad_norms is None:
        names_for_detailed_grad_norms = []
    if module.__class__.__name__ in SKIP:
        return {}, []

    norms_dict = {}
    all_norms = []

    for name, m in module.named_children():
        if name in names_for_detailed_grad_norms:
            new_names_for_detailed_grad_norms = []
            for n in names_for_detailed_grad_norms:
                if "." in n:
                    new_names_for_detailed_grad_norms.append(".".join(n.split(".")[1:]))
            t = get_module_norms(
                m,
                norm_type=norm_type,
                name_prefix=name_prefix + name + ".",
                names_for_detailed_grad_norms=new_names_for_detailed_grad_norms,
            )
            norms_dict.update(t[0])
            all_norms.extend(t[1])
        else:
            m_norms = []
            for p in m.parameters():
                if p.grad is not None:
                    current_norm = p.grad.data.norm(norm_type).item()
                    m_norms.append(current_norm)
                    all_norms.append(current_norm)
            if m_norms:
                norms_dict[f"grad_{norm_type}_norm/total_{name_prefix}{name}"] = (
                    torch.tensor(m_norms).norm(norm_type).item()
                )
                norms_dict[f"grad_{norm_type}_norm/max_{name_prefix}{name}"] = torch.tensor(m_norms).max().item()
    return norms_dict, all_norms


# Copied from https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/utils/agc.py
def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


def adaptive_clip_grad(parameters, clip_factor=0.01, eps=1e-3, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()
        max_norm = unitwise_norm(p_data, norm_type=norm_type).clamp_(min=eps).mul_(clip_factor)
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
        new_grads = torch.where(grad_norm < max_norm, g_data, clipped_grad)
        p.grad.detach().copy_(new_grads)


# Copied from https://github.com/pseeth/autoclip/blob/master/autoclip.py
class AutoClip:
    def __init__(self, percentile=10):
        self.grad_history = []
        self.percentile = percentile

    def compute_grad_norm(self, parameters):
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        return total_norm

    def __call__(self, pl_module: "pl.LightningModule", optimizer: Optimizer):
        grad_norm = self.compute_grad_norm(get_parameters(pl_module, optimizer))
        if not math.isnan(grad_norm) and not math.isinf(grad_norm):
            self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, self.percentile) if len(self.grad_history) != 0 else 6.0
        torch.nn.utils.clip_grad_norm_(get_parameters(pl_module, optimizer), clip_value)


def get_parameters(pl_module: "pl.LightningModule", optimizer: Optimizer):
    return pl_module.trainer.precision_plugin.main_params(optimizer)
