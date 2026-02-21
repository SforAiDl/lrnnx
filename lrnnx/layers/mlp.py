"""
Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mlp.py
Copyright (c) 2024, Tri Dao.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class GatedMLP(nn.Module):
    """
    Gated Multi-Layer Perceptron.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to None.
        out_features (int, optional): Number of output features. Defaults to None.
        activation (callable, optional): Activation function to apply to the gate. Defaults to F.silu.
        bias (bool, optional): Whether to include bias terms in linear layers. Defaults to False.
        multiple_of (int, optional): Round hidden_features to be a multiple of this value. Defaults to 128.
        device (torch.device, optional): Device to place tensors on. Defaults to None.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to None.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation=F.silu,
        bias: bool = False,
        multiple_of: int = 128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        out_features = (
            out_features if out_features is not None else in_features
        )
        hidden_features = (
            hidden_features
            if hidden_features is not None
            else int(8 * in_features / 3)
        )
        hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )

        self.fc1 = nn.Linear(
            in_features, 2 * hidden_features, bias=bias, **factory_kwargs
        )
        self.activation = activation
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias, **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Gated MLP.

        Args:
            x (torch.Tensor): Input tensor of shape ``(..., in_features)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(..., out_features)``.
        """
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y