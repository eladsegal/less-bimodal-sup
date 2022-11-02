import torch
from torch import nn


@torch.no_grad()
def init_weights_for_module(module):
    """Initialize the weights"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init_linear_weights(module.weight)
        if module.bias is not None:
            init_linear_bias(module.bias)
    elif isinstance(module, nn.Embedding):
        init_linear_weights(module.weight)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def init_linear_weights(tensor):
    tensor.data.normal_(mean=0.0, std=0.02)


def init_linear_bias(tensor):
    tensor.data.zero_()
