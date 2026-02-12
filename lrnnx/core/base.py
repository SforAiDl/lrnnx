"""
Base Model class for LRNNX.
"""

from abc import abstractmethod
from typing import Literal, Optional

from torch import Tensor
from torch.nn import Module

from .discretization import DISCRETIZE_FNS


class LRNN(Module):
    def __init__(
        self,
        discretization: Optional[
            Literal["zoh", "bilinear", "dirac", "async", "no_discretization"]
        ],
    ):
        """
        Base class for all LRNN models.

        Args
        ----
            discretization (Optional[Literal]): Discretization method to use, can be one of:
                - "zoh" for Zero-Order Hold
                - "bilinear" for Bilinear method
                - "dirac" for Dirac method
                - "async" for asynchronous discretization
                - "no_discretization" for no discretization
                - None for models that handle discretization internally
            Other parameters can be passed to the subclass.

        Each model must have a usage example in the documentation, like so:
        >>> from lrnnx.core import LRNN
        >>> my_lrnn = LRNN("zoh")
        >>> # create dummy input tensor and perform forward pass
        >>> # in subclass
        """
        super().__init__()
        if discretization is not None:
            assert (
                discretization in DISCRETIZE_FNS
            ), f"Discretization method {discretization} is not supported. Choose from {list(DISCRETIZE_FNS.keys())}."
            self.discretize_fn = DISCRETIZE_FNS[discretization]
        else:
            self.discretize_fn = None  # type: ignore[assignment]

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of through the LRNN.

        Args
        ----
            x (Tensor): Input tensor, ideally of shape (B, L, H).

            integration_timesteps (Tensor): Timesteps for async/event-driven
                discretization (Reference: https://arxiv.org/abs/2404.18508),
                ideally of shape (B, L). Only applicable for LTV models;
                LTI models ignore this parameter.

            lengths (Tensor): Lengths of sequences, ideally of shape (B,),
                this is required for bidirectional models.

        Returns
        -------
            Tensor: Output tensor, same shape as input (x), ideally (B, L, H).
        """

        raise NotImplementedError(
            "forward method must be implemented in the subclass."
        )
