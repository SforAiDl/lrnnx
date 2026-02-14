"""
Base class for LTV (Linear Time-Varying) models.

LTV models have time-varying dynamics, meaning the state transition matrices
(A, B, C, etc.) can change at each timestep based on the input. This makes
them more expressive than LTI models but prevents the use of FFT-based
convolution for training.

Key differences from LTI models:
- Cannot use convolution mode (FFT) since the kernel varies per timestep
- Support async/event-driven discretization with variable timesteps
- Must use scan for both training and inference
"""

from abc import abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch import Tensor

from lrnnx.core.base import LRNN


class LTV_LRNN(LRNN):
    """
    Base class for all LTV (Linear Time-Varying) LRNN models.

    Note:
        LTV models support async discretization for event-driven processing
        where timesteps between events may vary. This is specified via the
        ``integration_timesteps`` parameter in ``forward()``.

    Example:
        >>> from lrnnx.models.ltv import LTV_LRNN
        >>> my_lrnn = LTV_LRNN("zoh")
        >>> # create dummy input tensor and perform forward pass
        >>> # in subclass

    Args:
        discretization (Literal["zoh", "bilinear", "dirac", "async", "no_discretization"] | None): Discretization method to use.
            Can be one of "zoh", "bilinear", "dirac", "async", "no_discretization", or None for models that handle discretization internally.
            Other parameters can be passed to the subclass.
    """

    def __init__(
        self,
        discretization: Optional[
            Literal["zoh", "bilinear", "dirac", "async", "no_discretization"]
        ],
    ):
        super().__init__(discretization=discretization)

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        inference_cache: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Forward pass through the LTV model.

        Args:
            x (torch.Tensor): Input tensor, shape ``(B, L, H)``.
            integration_timesteps (torch.Tensor, optional): Timesteps for async/event-driven
                discretization (Reference: https://arxiv.org/abs/2404.18508),
                shape ``(B, L)``. If None, uniform timesteps are assumed. Defaults to None.
            lengths (torch.Tensor, optional): Lengths of sequences, shape ``(B,)``,
                required for variable-length sequences or bidirectional models. Defaults to None.
            inference_cache (dict, optional): Cache containing states and
                pre-computed values for efficient autoregressive generation.
                If provided during inference, enables incremental processing. Defaults to None.

        Returns:
            torch.Tensor: Output tensor, same shape as input (x), i.e., ``(B, L, H)``.
        """
        raise NotImplementedError(
            "forward method must be implemented in the subclass."
        )

    @abstractmethod
    def step(
        self,
        x: Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Performs a single recurrent step of the LTV model.

        This method is used for autoregressive inference, where inputs
        are processed one timestep at a time. Unlike LTI models, the
        dynamics may vary at each step based on the input.

        Args:
            x (torch.Tensor): Input at current timestep, shape ``(B, 1, H)``.
            inference_cache (Dict[str, Any]): Cache dictionary containing model
                states. This is the same format returned by ``allocate_inference_cache()``.
                The cache is updated in-place and also returned for convenience.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: A tuple containing:
                - y (torch.Tensor): Output at current timestep, shape ``(B, 1, H)``.
                - inference_cache (Dict[str, Any]): Updated cache dictionary.
        """
        raise NotImplementedError(
            "step method must be implemented in the subclass."
        )

    @abstractmethod
    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocates cache for efficient autoregressive inference.

        For LTV models, this typically includes:
        
        * Initial hidden state(s)
        * Any auxiliary states (e.g., convolution state for Mamba)
        * Metadata for tracking sequence position

        Args:
            batch_size (int): The batch size for inference.
            max_seqlen (int): Maximum sequence length to support.
            dtype (torch.dtype, optional): Data type for allocated tensors.
                If None, uses the model's default dtype. Defaults to None.
            **kwargs: Additional model-specific arguments.

        Returns:
            Dict[str, Any]: Cache dictionary that can be passed to ``forward()``.
                Should contain at minimum:
                - Model state tensors (e.g., "lrnn_state", "conv_state")
                - "seqlen_offset": Current position in the sequence
        """
        raise NotImplementedError(
            "allocate_inference_cache method must be implemented in the subclass."
        )