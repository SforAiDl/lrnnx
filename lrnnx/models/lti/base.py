"""
Base class for LTI models.
"""

from abc import abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.backends import opt_einsum

from lrnnx.core.base import LRNN


class LTI_LRNN(LRNN):
    def __init__(
        self,
        discretization: Literal[
            "zoh", "bilinear", "dirac", "no_discretization"
        ],
    ):
        """
        Base class for all LTI LRNN models.

        Args
        ----
            discretization (Literal): Discretization method to use, can be one of:
                - "zoh" for Zero-Order Hold
                - "bilinear" for Bilinear method
                - "dirac" for Dirac method
                - "no_discretization" for no discretization
            Other parameters can be passed to the subclass.

        Note: LTI models do not support async discretization as that requires
        time-varying dynamics. For async/event-driven models, use LTV models.

        Each model must have a usage example in the documentation, like so:
        >>> from lrnnx.models.lti import LTI_LRNN
        >>> my_lrnn = LTI_LRNN("zoh")
        >>> # create dummy input tensor and perform forward pass
        >>> # in subclass
        """
        # for optimal contractions
        assert opt_einsum.is_available()
        opt_einsum.strategy = "optimal"
        super().__init__(discretization=discretization)

    @abstractmethod
    def discretize(self) -> tuple[Tensor, Union[Tensor, float], Tensor]:
        """
        This function discretizes the A, B and C matrices,
        with a learned step-size delta. This could be done
        inside the `compute_kernel` method itself, but doing
        this explicitly outside allows for more flexibility
        later.

        Returns
        -------
            tuple[Tensor, Tensor, Tensor]: A tuple of tensors representing the
                discretized A, B, C matrices, ideally of shapes (B, N),
                (B, N, H) or float, and (B, H, N) respectively.
        """
        raise NotImplementedError(
            "discretize method must be implemented in the subclass."
        )

    @abstractmethod
    def compute_kernel(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """
        Computes the convolution kernel for efficient parallel processing.

        This function is only relevant for LTI models, for LTV
        models this will materialize a huge vector in-memory
        at every timestep, which is not efficient.
        Reference: https://github.com/kunibald413/aTENNuate/blob/15a27dab00d3bf2c27cbbbc3bd41a3d9196dca1e/attenuate/model.py#L30

        Args
        ----
            *args, **kwargs: Model-specific arguments (e.g., sequence length,
                discretized matrices). See subclass implementations for details.

        Returns
        -------
            tuple[Tensor, Tensor]: A tuple containing:
                - K (Tensor): Powers of A matrix (A^0, A^1, ..., A^{L-1}), shape (N, L)
                - B_norm (Tensor): Normalized input projection matrix, shape (N, H)
        """
        raise NotImplementedError(
            "compute_kernel method must be implemented in the subclass."
        )

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Performs a single recurrent step of the LTI model.

        This method is used for autoregressive inference, where inputs
        are processed one timestep at a time.

        Args
        ----
            x (torch.Tensor): Input at current timestep, shape (B, H)
            inference_cache (Dict[str, Any]): Cache dictionary from
                allocate_inference_cache() containing recurrent state and
                pre-computed matrices. Updated in-place and returned.

        Returns
        -------
            Tuple[torch.Tensor, Dict[str, Any]]: A tuple containing:
                - y (torch.Tensor): Output at current timestep, shape (B, H)
                - inference_cache (Dict[str, Any]): Updated cache dictionary
        """
        raise NotImplementedError(
            "step method must be implemented in the subclass."
        )

    @abstractmethod
    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int = 1,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocates initial state and caches matrices for efficient inference.

        For LTI models, the system matrices (A, B, C) are time-invariant,
        so they can be pre-computed once and reused for all timesteps during
        autoregressive generation.

        Args
        ----
            batch_size (int): The batch size for inference.
            max_seqlen (int): Maximum sequence length (unused for LTI, kept
                for interface consistency with LTV models).
            dtype (torch.dtype, optional): Data type for allocated tensors.
            **kwargs: Additional model-specific arguments.

        Returns
        -------
            Dict[str, Any]: Cache dictionary containing initial state and
                pre-computed matrices for use in step().
        """
        raise NotImplementedError(
            "allocate_inference_cache method must be implemented in the subclass."
        )
