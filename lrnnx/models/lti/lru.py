"""
Implementation of Linear Recurrent Unit (LRU) layer.
Paper: https://arxiv.org/abs/2303.06349.
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from lrnnx.core.convolution import opt_ssm_forward
from lrnnx.models.lti.base import LTI_LRNN


class LRU(LTI_LRNN):
    """
    Linear Recurrent Unit (LRU) layer.

    Paper: https://arxiv.org/abs/2303.06349

    Example:
        >>> model = LRU(d_model=64, d_state=64)
        >>> x = torch.randn(2, 128, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 128, 64])
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        r_min: float = 0,
        r_max: float = 1,
        max_phase: float = 2 * math.pi,
    ) -> None:
        """
        Initialize LRU layer.

        Args:
            d_model (int): Model dimension.
            d_state (int): State dimension.
            r_min (float, optional): Minimum radius for Lambda initialization. Defaults to 0.
            r_max (float, optional): Maximum radius for Lambda initialization. Defaults to 1.
            max_phase (float, optional): Maximum phase for Lambda initialization.
                Defaults to ``2 * math.pi``.
        """
        super().__init__(
            discretization="no_discretization"
        )  # discretization is unused in LRU
        self.d_model = d_model
        self.d_state = d_state

        self._init_parameters(d_model, d_state, r_min, r_max, max_phase)

    def _init_parameters(
        self,
        d_model: int,
        d_state: int,
        r_min: float = 0,
        r_max: float = 1,
        max_phase: float = 2 * math.pi,
    ) -> None:
        """
        Initialize parameters of the LRU layer.

        Args:
            d_model (int): Model dimension.
            d_state (int): State dimension.
            r_min (float, optional): Minimum radius for Lambda initialization. Defaults to 0.
            r_max (float, optional): Maximum radius for Lambda initialization. Defaults to 1.
            max_phase (float, optional): Maximum phase for Lambda initialization. Defaults to 2 * math.pi.
        """

        u1 = torch.rand(d_state)
        u2 = torch.rand(d_state)

        # nu_log and theta_log are used to init Lambda
        # values distributed uniformly on ring b/w r_min and r_max
        nu_log = torch.log(
            -0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )

        # phase b/w 0 and max_phase
        theta_log = torch.log(max_phase * u2)

        # glorot-initialized input/output projection matrices
        B_re = torch.randn(d_state, d_model) / (2 * d_model) ** 0.5
        B_im = torch.randn(d_state, d_model) / (2 * d_model) ** 0.5
        C_re = torch.randn(d_model, d_state) / d_state**0.5
        C_im = torch.randn(d_model, d_state) / d_state**0.5
        D = torch.randn(d_model)

        # normalization factor
        diag_lambda = torch.exp(-torch.exp(nu_log) + 1j * torch.exp(theta_log))
        # the original paper reports that setting gamma_log to
        # torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        # after every step also yields similar results to making it learnable
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))

        # register parameters
        self.nu_log = nn.Parameter(nu_log)
        self.theta_log = nn.Parameter(theta_log)
        self.B_re = nn.Parameter(B_re)
        self.B_im = nn.Parameter(B_im)
        self.C_re = nn.Parameter(C_re)
        self.C_im = nn.Parameter(C_im)
        self.D = nn.Parameter(D)
        self.gamma_log = nn.Parameter(gamma_log)

    def discretize(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        LRU uses no_discretization, so this acts like a prepare matrices method.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - A (torch.Tensor): Diagonal matrix of Lambda values, shape ``(N, N)``.
                - B (torch.Tensor): Complex input projection matrix, shape ``(N, H)``.
                - C (torch.Tensor): Complex output projection matrix, shape ``(H, N)``.
        """
        Lambda = torch.exp(
            -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
        )  # (N,)
        C_complex = self.C_re + 1j * self.C_im  # (H, N)
        B_complex = self.B_re + 1j * self.B_im  # (N, H)

        A = Lambda  # (N,)
        B = B_complex  # (N, H)
        C = C_complex  # (H, N)

        return A, B, C

    def compute_kernel(
        self, L: int, Lambda: Tensor, B_complex: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute Lambda and normalized B matrix for LRU.

        Args:
            L (int): Length of the input sequence.
            Lambda (torch.Tensor): Complex eigenvalues/diagonal elements, shape ``(N,)``.
            B_complex (torch.Tensor): Complex input projection matrix, shape ``(N, H)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Lambda (torch.Tensor): Complex eigenvalues/diagonal elements, shape ``(N,)``.
                - B_norm (torch.Tensor): Normalized complex input projection matrix, shape ``(N, H)``.
        """
        t_idx = torch.arange(
            L, dtype=torch.float32, device=Lambda.device
        )  # (L,)
        K = Lambda.unsqueeze(-1) ** t_idx.unsqueeze(0)  # (N, L)
        B_norm = B_complex * torch.exp(self.gamma_log).unsqueeze(-1)  # (N, H)

        return K, B_norm

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the LRU layer.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, L, H)``.
            integration_timesteps (torch.Tensor, optional): <To be implemented>. Defaults to None.
            lengths (torch.Tensor, optional): <To be implemented>. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, H)``.
        """

        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must be of shape (B, L, H), got {x.dim()}D tensor with shape {x.shape}"
            )
        L = x.shape[1]

        # prepare matrices
        Lambda, B_complex, C_complex = self.discretize()
        # compute kernel
        K, B_norm = self.compute_kernel(L, Lambda, B_complex)
        # convolve over input
        y_conv = opt_ssm_forward(x, K, B_norm, C_complex)
        # skip connection
        y = y_conv + x * self.D  # (B, L, H)

        return y

    def step(
        self,
        x: Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Single step inference for LRU layer.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, H)`` - single timestep.
            inference_cache (Dict[str, Any]): Cache from ``allocate_inference_cache()``
                containing "lrnn_state" and pre-computed matrices.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: A tuple containing:
                - y (torch.Tensor): Output tensor of shape ``(B, H)``.
                - inference_cache (Dict[str, Any]): Updated cache dictionary.
        """
        if x.dim() != 2:
            raise ValueError(
                f"Input tensor must be of shape (B, H), got {x.dim()}D tensor with shape {x.shape}"
            )

        state = inference_cache["lrnn_state"]

        # Extract cached matrices
        Lambda = inference_cache["Lambda"]
        B_norm = inference_cache["B_norm"]
        C_complex = inference_cache["C_complex"]

        # Recurrent update: x_t -> state_{t+1}
        # state_{t+1} = Lambda * state_t + B_norm @ u_t
        input_projection = torch.einsum(
            "nh,bh->bn", B_norm, x.to(B_norm.dtype)
        )  # (B, N)
        new_state = Lambda * state + input_projection  # (B, N)

        # Output computation: y_t = C @ state_t + D * x_t
        state_output = torch.einsum(
            "hn,bn->bh", C_complex, new_state
        ).real  # (B, H)
        y = state_output + x * self.D  # (B, H)

        inference_cache["lrnn_state"].copy_(new_state)
        return y, inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int = 1,
        dtype=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocate initial state and cached matrices for inference.

        Args:
            batch_size (int): Batch size.
            max_seqlen (int, optional): Maximum sequence length (unused, kept for
                interface consistency with LTV models). Defaults to 1.
            dtype (torch.dtype, optional): Data type for allocated tensors (unused). Defaults to None.
            **kwargs: Additional model-specific arguments.

        Returns:
            Dict[str, Any]: Cache dict with "lrnn_state" and
                pre-computed discrete matrices.
        """

        # Initialize state to zeros
        device = self.nu_log.device
        initial_state = torch.zeros(
            batch_size, self.d_state, dtype=torch.complex64, device=device
        )

        # Pre-compute and cache matrices (LTI - compute once)
        Lambda, B_complex, C_complex = self.discretize()
        B_norm = B_complex * torch.exp(self.gamma_log).unsqueeze(-1)

        return {
            "lrnn_state": initial_state,
            "Lambda": Lambda,
            "B_norm": B_norm,
            "C_complex": C_complex,
        }