import math

import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._classifier(inputs)


class ShallowClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self._classifier = nn.Linear(input_dim, output_dim)


# Based on https://github.com/dandelin/ViLT/blob/5ec6ef9df56cefa490e65c02f00a83469dd37f93/vilt/modules/vilt_module.py#L67
# ==================================================================================================================== #
#                                                         ViLT                                                         #
# ==================================================================================================================== #
class ViltClassifier(Classifier):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self._classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )
