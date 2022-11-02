from typing import List, Union, Optional
import torch
from torch import nn

# Copied from https://github.com/allenai/allennlp/blob/3c1ac0329006cafcae36442957b80d3fe0cabca8/allennlp/modules/feedforward.py


class NoActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Feedforward(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations: Optional[Union[nn.Module, List[nn.Module]]] = None,
        dropout: Union[float, List[float]] = 0.0,
    ) -> None:

        super().__init__()

        no_activation = NoActivation()
        if activations is None:
            activations = no_activation

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers - 1  # type: ignore
        hidden_dims += [output_dim]

        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        else:
            for i, activation in enumerate(activations):
                if activation is None:
                    activations[i] = no_activation

        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore

        if len(hidden_dims) != num_layers:
            raise Exception(f"len(hidden_dims) ({len(hidden_dims) - 1}) + 1 != num_layers ({num_layers})")
        if len(activations) != num_layers:
            raise Exception(f"len(activations) ({len(activations)}) != num_layers ({num_layers})")
        if len(dropout) != num_layers:
            raise Exception(f"len(dropout) ({len(dropout)}) != num_layers ({num_layers})")
        self._activations = nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = nn.ModuleList(linear_layers)
        dropout_layers = [nn.Dropout(p=value) for value in dropout]
        self._dropout = nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self._input_dim = input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(self._linear_layers, self._activations, self._dropout):
            output = dropout(activation(layer(output)))
        return output
