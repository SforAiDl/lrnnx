"""
Linear Oscillatory State-Space (LinOSS) layer.
Paper: https://arxiv.org/abs/2410.03943
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

from lrnnx.models.lti.base import LTI_LRNN
from lrnnx.ops.linoss_scan import linoss_scan_fn


class LinOSS(LTI_LRNN):
    """
    Linear Oscillatory State-Space (LinOSS) layer.

    A forced linear oscillator ``y'' = -A y + B u`` is written as a first
    order 2x2 system per state channel and integrated with either an implicit
    (``"im"``) or an implicit-explicit (``"imex"``) scheme. Both discretize to
    ``z_t = A_bar z_{t-1} + gamma_bar (B u_t)`` with a shared real 2x2
    ``A_bar``, which the associative scan in
    :func:`lrnnx.ops.linoss_scan.linoss_scan_fn` evaluates in parallel.

    Paper: https://arxiv.org/abs/2410.03943

    Example:
        >>> model = LinOSS(d_model=64, d_state=64, discretization="im")
        >>> x = torch.randn(2, 128, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 128, 64])
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        discretization: Literal["im", "imex"] = "im",
    ) -> None:
        """
        Initialize LinOSS layer.

        Args:
            d_model (int): Model dimension ``H``.
            d_state (int): Oscillator / SSM state dimension ``P``.
            discretization (Literal["im", "imex"], optional): Oscillatory
                discretization. Defaults to ``"im"``.
        """
        super().__init__(discretization=discretization)

        self.d_model = d_model
        self.d_state = d_state
        self.discretization = discretization

        self._init_parameters(d_model, d_state)

    def _init_parameters(self, d_model: int, d_state: int) -> None:
        """
        Initialize the parameters of the LinOSS layer.

        ``B`` and ``C`` are complex and stored as trailing ``(real, imag)``
        pairs so that they stay ordinary real parameters for the optimizer.

        Args:
            d_model (int): Model dimension ``H``.
            d_state (int): Oscillator / SSM state dimension ``P``.
        """
        uniform = lambda shape, std: Parameter(
            torch.empty(*shape).uniform_(-std, std)
        )

        # relu(A_diag) and sigmoid(steps) keep the oscillator stable.
        self.A_diag = Parameter(torch.rand(d_state))
        self.steps = Parameter(torch.rand(d_state))

        self.B = uniform((d_state, d_model, 2), 1.0 / math.sqrt(d_model))
        self.C = uniform((d_model, d_state, 2), 1.0 / math.sqrt(d_state))
        self.D = Parameter(torch.randn(d_model))

    def discretize(self) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        """
        Apply the ReLU / sigmoid constraints and the IM or IMEX discretization.

        Unlike the diagonal schemes used by the other LTI models, LinOSS
        discretizes to a 2x2 block, and ``B`` / ``C`` are used undiscretized.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``A_bar`` of shape ``(P, 4)``
                (row-major 2x2) and ``gamma_bar`` of shape ``(P, 2)``.
        """
        A_diag = torch.relu(self.A_diag)
        steps = torch.sigmoid(self.steps)
        # The LinOSS schemes always return a tensor multiplier, never a float.
        return self.discretize_fn(  # type: ignore[return-value]
            A_diag, steps, None
        )

    def compute_kernel(self, *args, **kwargs):
        """
        Not available for LinOSS.

        Raises:
            NotImplementedError: LinOSS is driven by a 2x2 associative scan,
                so it has no diagonal convolution kernel.
        """
        raise NotImplementedError(
            "LinOSS uses a 2x2 associative scan, not a convolution kernel."
        )

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parallel LinOSS forward pass via the 2x2 associative scan.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, L, H)``.
            integration_timesteps (torch.Tensor, optional): Not used by LinOSS
                (LTI model). Kept for interface compatibility with LTV models.
                Defaults to None.
            lengths (torch.Tensor, optional): Not used. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, H)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Input tensor must be of shape (B, L, H), got {x.dim()}D tensor with shape {x.shape}"
            )

        batch = x.shape[0]
        A_bar, gamma_bar = self.discretize()

        # The input, A_bar and the output are all real, so the complex
        # recurrence splits into two independent real scans - one for the real
        # and one for the imaginary part of Bu - stacked along the batch axis.
        # That keeps every matmul real: two GEMMs in and two out instead of the
        # four each a complex GEMM would need.
        Bu = torch.einsum("phc,blh->cbpl", self.B, x)  # (2, B, P, L)
        g1, g2 = (g.view(1, 1, -1, 1) for g in gamma_bar.unbind(-1))
        F = torch.stack([g1 * Bu, g2 * Bu], dim=-1)  # (2, B, P, L, 2)

        xs = linoss_scan_fn(A_bar, F.flatten(0, 1))  # (2B, P, L, 2)
        # Only the position component of the oscillator is read out.
        z = xs[..., 1].unflatten(0, (2, batch))  # (2, B, P, L)

        # Re(C z) = C_re z_re - C_im z_im
        y = torch.einsum("hp,bpl->blh", self.C[..., 0], z[0]) - torch.einsum(
            "hp,bpl->blh", self.C[..., 1], z[1]
        )
        return y + self.D * x

    def step(
        self,
        x: Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Perform a single recurrent step of the LinOSS layer.

        Args:
            x (torch.Tensor): Input at the current timestep, shape ``(B, H)``.
            inference_cache (Dict[str, Any]): Cache from
                ``allocate_inference_cache()`` containing "lrnn_state" and
                pre-computed matrices.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: A tuple containing:
                - y : Output at the current timestep, shape ``(B, H)``.
                - inference_cache : Updated cache dictionary.
        """
        if x.dim() != 2:
            raise ValueError(
                f"Input tensor must be of shape (B, H), got {x.dim()}D tensor with shape {x.shape}"
            )

        state = inference_cache["lrnn_state"]  # (B, P, 2), complex
        A_bar = inference_cache["A_bar"]
        gamma_bar = inference_cache["gamma_bar"]
        B_complex = inference_cache["B_complex"]
        C_complex = inference_cache["C_complex"]

        Bu = torch.einsum("ph,bh->bp", B_complex, x.to(B_complex.dtype))
        m11, m12, m21, m22 = A_bar.unbind(-1)
        v, z = state[..., 0], state[..., 1]

        # z_t = A_bar z_{t-1} + gamma_bar (B u_t)
        new_v = m11 * v + m12 * z + gamma_bar[:, 0] * Bu
        new_z = m21 * v + m22 * z + gamma_bar[:, 1] * Bu

        # y_t = Re(C z_t) + D u_t
        y = torch.einsum("hp,bp->bh", C_complex, new_z).real + self.D * x

        inference_cache["lrnn_state"].copy_(
            torch.stack([new_v, new_z], dim=-1)
        )
        return y, inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int = 1,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocate the initial state and cache the discretized matrices.

        Args:
            batch_size (int): Batch size for recurrent inference.
            max_seqlen (int, optional): Maximum sequence length (unused, kept
                for interface consistency with LTV models). Defaults to 1.
            dtype (torch.dtype, optional): Data type for allocated tensors
                (unused; the state is complex64). Defaults to None.
            **kwargs: Additional model-specific arguments.

        Returns:
            Dict[str, Any]: Cache dict with "lrnn_state" and the pre-computed
                discrete matrices.
        """
        device = self.A_diag.device
        initial_state = torch.zeros(
            batch_size, self.d_state, 2, dtype=torch.complex64, device=device
        )

        # Pre-compute and cache matrices (LTI - compute once)
        A_bar, gamma_bar = self.discretize()

        return {
            "lrnn_state": initial_state,
            "A_bar": A_bar,
            "gamma_bar": gamma_bar,
            "B_complex": torch.complex(self.B[..., 0], self.B[..., 1]),
            "C_complex": torch.complex(self.C[..., 0], self.C[..., 1]),
        }
