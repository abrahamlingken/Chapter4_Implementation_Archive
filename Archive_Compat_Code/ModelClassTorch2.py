#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Archive-local subset of the legacy ModelClassTorch2 module.

The archived Chapter 4 workflow only needs the neural-network definition and
Xavier initialization helpers in order to:
1. load legacy ``model.pkl`` files serialized as ``ModelClassTorch2.Pinns``
2. instantiate the same network during archive-local retraining

The broader legacy training utilities depended on many modules that are not part
of this archive, so this standalone subset keeps only the pieces required for
model construction and inference.
"""

from __future__ import annotations

try:
    from .ImportFile import math, nn, torch
except ImportError:
    from ImportFile import math, nn, torch

pi = math.pi


class Swish(nn.Module):
    """Swish activation used by the archived PINN models."""

    def forward(self, x):
        return x * torch.sigmoid(x)


def activation(name: str) -> nn.Module:
    """Map legacy activation names to torch modules."""

    if name in ["tanh", "Tanh"]:
        return nn.Tanh()
    if name in ["relu", "ReLU"]:
        return nn.ReLU(inplace=True)
    if name in ["lrelu", "LReLU"]:
        return nn.LeakyReLU(inplace=True)
    if name in ["sigmoid", "Sigmoid"]:
        return nn.Sigmoid()
    if name in ["softplus", "Softplus"]:
        return nn.Softplus(beta=4)
    if name in ["celu", "CeLU"]:
        return nn.CELU()
    if name in ["swish", "Swish"]:
        return Swish()
    raise ValueError(f"Unknown activation function: {name}")


class Pinns(nn.Module):
    """Minimal legacy PINN network definition preserved for archive loading."""

    def __init__(
        self,
        input_dimension,
        output_dimension,
        network_properties,
        additional_models=None,
        solid_object=None,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(
            network_properties["regularization_parameter"]
        )
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.solid_object = solid_object
        self.additional_models = additional_models
        self.activation = activation(self.act_string)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


def init_xavier(model: nn.Module) -> None:
    """Apply the legacy Xavier initialization scheme."""

    def init_weights(module):
        if (
            isinstance(module, nn.Linear)
            and module.weight.requires_grad
            and module.bias.requires_grad
        ):
            gain = nn.init.calculate_gain("tanh")
            nn.init.xavier_uniform_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)

    model.apply(init_weights)


__all__ = ["Pinns", "Swish", "activation", "init_xavier", "pi"]
