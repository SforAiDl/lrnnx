"""
Basic S5 SSM.
Reference: https://openreview.net/forum?id=Ai8Hw3AXqks
"""

import math
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from lrnnx.core.convolution import opt_ssm_forward
from lrnnx.models.lti.base import LTI_LRNN


class S5(LTI_LRNN):
    """
    Basic S5 State Space Model.
    Reference: https://openreview.net/forum?id=Ai8Hw3AXqks

    Args:
        d_model (int): Model dimension.
        d_state (int): State dimension (P in the original paper).
        discretization (Literal["zoh", "bilinear", "dirac", "no_discretization"]): Discretization method to use.
        conj_sym (bool, optional): If True, uses conjugate symmetry for the state space model. Defaults to False.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,  # this is P in the paper, the actual state dimension of the system.
        discretization: Literal[
            "zoh",
            "bilinear",
            "dirac",
            "no_discretization",  # diff discretization methods available to the user.
        ],
        conj_sym: bool = False,  # if True, uses conjugate symmetry for the state space model.
    ):
        super().__init__(discretization=discretization)

        self.d_model = d_model
        self.d_state = d_state

        # just keeping this incase the tests use that
        self.hid_dim = d_model
        self.state_dim = d_state

        if conj_sym:
            raise NotImplementedError(
                "Conjugate symmetry is not implemented yet."
            )
        self.conj_sym = conj_sym  # if True, uses conjugate symmetry for the state space model.

        self._init_parameters()

    def _init_parameters(self):
        """
        Initializes the parameters of the S5 model.
        This method sets up the system matrix A, input matrix B, time step log_dt, and output weight matrix C.
        """
        init_parameter = lambda mat: Parameter(
            torch.tensor(mat, dtype=torch.float)
        )
        normal_parameter = lambda fan_in, shape: Parameter(
            torch.randn(*shape) * math.sqrt(2 / fan_in)
        )

        H = self.d_model  # hidden dimension (input)
        N = self.d_state  # state dimension (inner dimension for the SSM)

        # Creatiing real and imaginary parts of A
        A_real = 0.5 * np.ones(N)
        A_imag = math.pi * np.arange(N)

        # Stored in log for numerical stability.
        log_A_real = np.log(np.exp(A_real) - 1)  # inverse softplus
        # Stacking into complex diagonal format (Re, Im)
        A = np.stack([log_A_real, A_imag], axis=-1)

        # log spaced time scale
        log_dt = np.linspace(np.log(0.001), np.log(0.1), N)

        B = np.ones((N, H)) / math.sqrt(H)  # shape: (N, H)

        self.A = init_parameter(A)
        self.B = init_parameter(B)
        self.log_dt = init_parameter(log_dt)
        self.C = normal_parameter(N, (H, N, 2))
        self.D = normal_parameter(H, (H, H))  # output projection matrix

    def discretize(
        self,
    ) -> tuple[torch.Tensor, Union[torch.Tensor, float], torch.Tensor]:
        """
        Discretizes the continuous-time system matrices A and B using the specified discretization method.

        Returns:
            tuple[torch.Tensor, Union[torch.Tensor, float], torch.Tensor]: A tuple containing:
                - A_bar (torch.Tensor): Discretized system matrix A, shape ``(N,)``.
                - gamma_bar (Union[torch.Tensor, float]): Input normalizer, shape ``(N,)`` or a float.
                - C_complex (torch.Tensor): Complex output matrix C, shape ``(H, N)``.
        """
        log_A_real, A_imag = self.A.T  # (2, state_dim)
        dt = self.log_dt.exp()  # log time steps converted to real time.
        # They are stored in log-space during training as it provides numerical stability and allows the model
        # to learn a wide range of temporal scales, from very fast (small dt) to very slow (large dt) dynamics

        # Continous time complex A matrix
        A_complex = -F.softplus(log_A_real) + 1j * A_imag  # shape: (N,)

        # Discretize (LTI - no integration_timesteps)
        A_bar, gamma_bar = self.discretize_fn(
            A_complex, dt, None
        )  # (N,), (N,)

        # also prepare C matrix
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]  # (H, N)

        return A_bar, gamma_bar, C_complex

    def compute_kernel(
        self,
        L: int,
        A_bar: Tensor,
        gamma_bar: Union[Tensor, float],
    ):
        """
        Computes the kernel matrices for the S5 model: A^t and B_bar.

        Args:
            L (int): Length of the input sequence.
            A_bar (torch.Tensor): Discretized system matrix A, shape ``(N,)``.
            gamma_bar (Union[torch.Tensor, float]): Input normalizer, shape ``(N,)`` or a float.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A_power (torch.Tensor): Power of the discretized system matrix A, shape ``(N, L)``.
                - B_bar (torch.Tensor): Normalized input projection matrix, shape ``(N, H)``.
        """
        # Compute B_bar
        if isinstance(gamma_bar, float):
            B_bar = gamma_bar * self.B  # (N, H)
        else:
            assert (
                gamma_bar.dim() == 1
            ), f"gamma_bar should be 1D tensor, got {gamma_bar.dim()}D tensor"
            B_bar = gamma_bar.unsqueeze(-1) * self.B

        # Compute A^t
        lrange = torch.arange(L, device=A_bar.device)
        A_power = A_bar[:, None] ** lrange[None, :]
        return A_power, B_bar

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of the S5 SSM using FFT-based convolution.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, L, H)``.
            integration_timesteps (torch.Tensor, optional): Not used by S5 (LTI model).
                Kept for interface compatibility with LTV models. Defaults to None.
            lengths (torch.Tensor, optional): Lengths of the input sequences, shape ``(B,)``.
                TODO: Support bidirectional models. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, H)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must be of shape (B, L, H), got {x.dim()}D tensor with shape {x.shape}"
            )

        L = x.shape[1]

        A_bar, B_bar, C_complex = self.discretize()

        K, B_hat = self.compute_kernel(L, A_bar, B_bar)

        return opt_ssm_forward(x, K, B_hat, C_complex) + x @ self.D

    def step(
        self,
        x: torch.Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Performs a single recurrent step of the S5 model.

        Args:
            x (torch.Tensor): Input at current time step, shape ``(B, H)``.
            inference_cache (Dict[str, Any]): Cache from ``allocate_inference_cache()``
                containing "lrnn_state" and pre-computed matrices.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: Output y_t of shape ``(B, H)``
                and updated cache dictionary.
        """

        if x.dim() != 2:
            raise ValueError(
                f"Input tensor must be of shape (B, H), got {x.dim()}D tensor with shape {x.shape}"
            )

        state = inference_cache["lrnn_state"]

        # Extract cached matrices
        A_bar = inference_cache["A_bar"]
        B_bar = inference_cache["B_bar"]
        C_complex = inference_cache["C_complex"]

        # Recurrent update: x_t -> state_{t+1}
        # state_{t+1} = A_bar * state_t + B_bar @ u_t
        input_projection = torch.einsum("bh,nh->bn", x.to(B_bar.dtype), B_bar)
        new_state = A_bar * state + input_projection

        # Output computation: y_t = C @ state_t + D * u_t
        state_output = torch.einsum("hn,bn->bh", C_complex, new_state)
        y = state_output.real + x @ self.D  # (B, H)

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
        Allocates cache for inference.

        Args:
            batch_size (int): The batch size for the input data.
            max_seqlen (int, optional): Maximum sequence length (unused, kept for
                interface consistency with LTV models). Defaults to 1.
            dtype (torch.dtype, optional): Data type for allocated tensors (unused). Defaults to None.
            **kwargs: Additional model-specific arguments.

        Returns:
            Dict[str, Any]: Cache dict with "lrnn_state" and
                pre-computed discrete matrices.
        """
        # Initialize state to zeros
        device = self.A.device
        initial_state = torch.zeros(
            batch_size, self.d_state, dtype=torch.complex64, device=device
        )

        # Pre-compute and cache matrices (LTI - compute once)
        A_bar, gamma_bar, C_complex = self.discretize()

        if isinstance(gamma_bar, float):
            B_bar = gamma_bar * self.B.to(torch.complex64)
        else:
            assert (
                gamma_bar.dim() == 1
            ), f"gamma_bar should be 1D tensor, got {gamma_bar.dim()}D tensor"
            B_bar = gamma_bar.unsqueeze(-1) * self.B

        return {
            "lrnn_state": initial_state,
            "A_bar": A_bar,
            "B_bar": B_bar,
            "C_complex": C_complex,
        }