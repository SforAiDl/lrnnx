"""
S5 SSM with CUDA kernel acceleration.
Reference: https://openreview.net/forum?id=Ai8Hw3AXqks
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from lrnnx.models.ltv.base import LTV_LRNN
from lrnnx.ops.simplified_scan import s5_inner_fn, simplified_scan_fn
from lrnnx.utils.init import (
    init_CV,
    init_log_steps,
    init_VinvB,
    make_DPLR_HiPPO,
)

try:
    from lrnnx.ops.triton.simplified_state_update import (
        simplified_state_update,
    )
except ImportError:
    simplified_state_update = None


class S5(LTV_LRNN):
    """
    S5 SSM with CUDA kernel acceleration.
    Reference: https://openreview.net/forum?id=Ai8Hw3AXqks

    Example:
        >>> model = S5(d_model=64, d_state=64)
        >>> x = torch.randn(2, 128, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 128, 64])
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        discretization: Literal["bilinear", "zoh", "dirac"] = "zoh",
        conj_sym: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        use_fast_path: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initialize S5 model.

        Args:
            d_model (int): Model dimension.
            d_state (int): State dimension.
            discretization (Literal["bilinear", "zoh", "dirac"], optional):
                Discretization method. Defaults to ``"zoh"``.
            conj_sym (bool, optional): If True, uses conjugate symmetry for the
                state space model. Defaults to False.
            dt_min (float, optional): Minimum value for dt initialization. Defaults to 0.001.
            dt_max (float, optional): Maximum value for dt initialization. Defaults to 0.1.
            step_rescale (float, optional): Rescale factor for step size. Defaults to 1.0.
            use_fast_path (bool, optional): Whether to use fused CUDA kernels. Defaults to True.
            device (torch.device, optional): Device for parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type for parameters. Defaults to None.
        """
        super().__init__(discretization=discretization)

        self.d_model = d_model
        self.d_state = d_state
        self.conj_sym = conj_sym
        self.use_fast_path = use_fast_path
        self.discretization = discretization
        self.step_rescale = step_rescale

        H, P = d_model, d_state

        if conj_sym:
            Lambda, _, _, V, _ = make_DPLR_HiPPO(2 * P)
            Lambda = Lambda[:P]
            V = V[:, :P]
            Vinv = V.conj().T
            local_P = 2 * P
        else:
            Lambda, _, _, V, _ = make_DPLR_HiPPO(P)
            Vinv = np.linalg.inv(V)
            local_P = P

        A = np.stack([Lambda.real, Lambda.imag], axis=-1)
        self.A = Parameter(torch.tensor(A, dtype=torch.float32, device=device))
        self.B = Parameter(init_VinvB(Vinv, local_P, H).to(device=device))
        self.C = Parameter(init_CV(V, local_P, H).to(device=device))
        self.D = Parameter(torch.randn(H, device=device) * 0.1)
        self.log_dt = Parameter(
            init_log_steps(P, dt_min, dt_max).to(device=device)
        )

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        inference_cache: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Forward pass through S5.

        Args:
            x (torch.Tensor): Input tensor of shape ``(B, L, H)``.
            integration_timesteps (torch.Tensor, optional): Timesteps for async/event-driven discretization. Defaults to None.
            lengths (torch.Tensor, optional): Lengths of sequences, required for variable-length sequences. Defaults to None.
            inference_cache (Dict[str, Any], optional): Cache for autoregressive generation. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, H)``.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, L, H), got {x.dim()}D")

        batch, seqlen, _ = x.shape
        P = self.d_state

        if inference_cache is not None:
            seqlen_offset = inference_cache.get("seqlen_offset", 0)
            if seqlen_offset > 0:
                out, inference_cache = self.step(x, inference_cache)
                return out

        A = torch.complex(self.A[..., 0], self.A[..., 1])
        B_tilde = torch.complex(self.B[..., 0], self.B[..., 1])
        C_tilde = torch.complex(self.C[..., 0], self.C[..., 1])
        dt = self.log_dt.exp() * self.step_rescale

        x_t = x.transpose(1, 2)
        u = x_t.to(torch.complex64)
        delta = (
            dt.unsqueeze(0).unsqueeze(-1).expand(batch, P, seqlen).contiguous()
        )

        deltaA = None
        if integration_timesteps is not None:
            # Async/event mode: use separate deltaA for A discretization
            # integration_timesteps: (B, L) or (B, L, 1)
            if integration_timesteps.dim() == 3:
                integration_timesteps = integration_timesteps.squeeze(-1)
            deltaA = (
                integration_timesteps.to(delta.dtype)
                .unsqueeze(1)
                .expand(batch, P, seqlen)
                * dt.unsqueeze(0).unsqueeze(-1)
            ).contiguous()

        if self.use_fast_path and x.is_cuda:
            y = s5_inner_fn(
                u,
                delta,
                A,
                B_tilde,
                C_tilde,
                self.D.float(),
                deltaA=deltaA,
                discretization=self.discretization,
                conj_sym=self.conj_sym,
            )
        else:
            # Slow path: use simplified_scan_fn + manual post-processing
            y_complex = simplified_scan_fn(
                u,
                delta,
                A,
                B_tilde,
                C_tilde,
                deltaA=deltaA,
                discretization=self.discretization,
                return_last_state=False,
            )

            # Apply conjugate symmetry
            if self.conj_sym:
                y = 2.0 * y_complex.real
            else:
                y = y_complex.real

            # Skip connection: D * real(u)
            u_real = u.real if u.is_complex() else u
            y = y + self.D.float().unsqueeze(0).unsqueeze(-1) * u_real

        return y.transpose(1, 2).to(x.dtype)

    def step(
        self,
        x: Tensor,
        inference_cache: Dict[str, Any],
        integration_timesteps: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Performs a single recurrent step of S5.

        When the simplified_state_update Triton kernel is available and the
        tensors live on CUDA, the state is updated **in-place** via the kernel
        (which also fuses discretization, input projection, and output
        projection into a single launch).  Otherwise a pure-PyTorch fallback
        is used.

        Args:
            x (torch.Tensor): Input at current timestep, shape ``(B, 1, H)`` or ``(B, H)``.
            inference_cache (Dict[str, Any]): Cache dictionary containing SSM state and
                continuous-time parameters.
            integration_timesteps (torch.Tensor, optional): Optional per-step integration timesteps
                for event/async mode, shape ``(B,)`` or ``(B, 1)``. Defaults to None.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: A tuple containing:
                - y : Output tensor at the current timestep.
                - inference_cache : Updated cache dictionary.
        """
        if x.dim() == 3:
            x = x.squeeze(1)

        state = inference_cache[
            "ssm_state"
        ]  # (B, P) complex64, mutated in-place
        A = inference_cache["A"]  # (P,)   complex64
        B_tilde = inference_cache["B_tilde"]  # (P, H) complex64
        C_tilde = inference_cache["C_tilde"]  # (H, P) complex64
        dt = inference_cache["dt"]  # (P,)   float32

        # Optional per-step deltaA for event/async mode
        deltaA = None
        if integration_timesteps is not None:
            if integration_timesteps.dim() > 1:
                timestep = integration_timesteps.view(-1, 1)
            else:
                timestep = integration_timesteps.unsqueeze(-1)
            deltaA = timestep.to(dt.dtype) * dt.unsqueeze(0)  # (B, P)

        if simplified_state_update is not None and x.is_cuda:
            # Triton fast-path (in-place state update)
            y = simplified_state_update(
                state,
                x.float(),
                dt,
                A,
                B_tilde,
                C_tilde,
                D=None,  # D is 1-D here; handled below
                deltaA=deltaA,
                discretization=self.discretization,
                conj_sym=self.conj_sym,
            )
        else:
            dtA = (
                deltaA
                if deltaA is not None
                else dt.unsqueeze(0).expand(x.shape[0], -1)
            )

            if self.discretization == "bilinear":
                half_dtA_A = 0.5 * dtA * A
                A_bar = (1.0 + half_dtA_A) / (1.0 - half_dtA_A)
                gamma_bar = dt / (1.0 - 0.5 * dt * A)
            elif self.discretization == "zoh":
                A_bar = torch.exp(dtA * A)
                gamma_bar = (torch.exp(dt * A) - 1.0) / A
            elif self.discretization == "dirac":
                A_bar = torch.exp(dtA * A)
                gamma_bar = torch.ones_like(dt, dtype=A.dtype)
            else:
                raise ValueError(
                    f"Unknown discretization: {self.discretization}"
                )

            Bu = torch.einsum("ph,bh->bp", B_tilde, x.to(B_tilde.dtype))
            state.copy_(A_bar * state + gamma_bar * Bu)

            y_complex = torch.einsum("hp,bp->bh", C_tilde, state)
            y = 2.0 * y_complex.real if self.conj_sym else y_complex.real

        # Skip connection
        y = y + self.D * x

        inference_cache["seqlen_offset"] += 1

        return y.unsqueeze(1).to(x.dtype), inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocates cache for S5 autoregressive inference.

        Stores the **continuous-time** parameters so that
        simplified_state_update can fuse discretization into the kernel.

        Args:
            batch_size (int): The batch size for inference.
            max_seqlen (int): Maximum sequence length (unused, for interface consistency).
            dtype (torch.dtype, optional): Data type for allocated tensors. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Cache dictionary containing SSM state and continuous-time matrices.
        """
        device = self.A.device
        P = self.d_state

        A = torch.complex(self.A[..., 0], self.A[..., 1])  # (P,)
        B_tilde = torch.complex(self.B[..., 0], self.B[..., 1])  # (P, H)
        C_tilde = torch.complex(self.C[..., 0], self.C[..., 1])  # (H, P)
        dt = self.log_dt.exp() * self.step_rescale  # (P,)

        return {
            "ssm_state": torch.zeros(
                batch_size, P, dtype=torch.complex64, device=device
            ),
            "A": A,
            "B_tilde": B_tilde,
            "C_tilde": C_tilde,
            "dt": dt,
            "seqlen_offset": 0,
        }