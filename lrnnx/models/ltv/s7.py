"""
S7: Selective and Simplified State Space Layers for Sequence Modeling
https://arxiv.org/abs/2410.03464
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from lrnnx.models.ltv.base import LTV_LRNN
from lrnnx.ops.s7_scan import s7_inner_fn, s7_scan_fn
from lrnnx.utils.init import make_DPLR_HiPPO

try:
    from lrnnx.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class S7(LTV_LRNN):
    def __init__(
        self,
        d_model,
        d_state,
        J=1,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        """
        Initialize S7 layer.

        Args
        ----
            d_model (int): Model dimension
            use_fast_path (bool): Whether to use the CUDA fast path if available
            layer_idx (int, optional): Layer index for multi-layer models, used for caching
            device (torch.device, optional): Device for the model parameters
            dtype (torch.dtype, optional): Data type for the model parameters

        Example
        -------
        >>> model = S7(d_model=64)
        >>> x = torch.randn(2, 128, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 128, 64])
        """
        super().__init__(discretization="no_discretization")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        assert (
            d_state % J == 0
        ), "d_state must be divisible by J (number of blocks for initialization)"
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(
            self.d_model, self.d_model, bias=False, **factory_kwargs
        )

        # Lambda's HiPPO stuff
        base, _, _, _, _ = make_DPLR_HiPPO(d_state // J)
        self.base_params = torch.nn.Parameter(
            torch.from_numpy(base)
            .float()
            .repeat(
                J,
            ),
            requires_grad=True,
        )

        # x_proj outputs: A (N), B (H*N), C (H*N), D (H), bias (N)
        self.x_proj = nn.Linear(
            self.d_model,
            self.d_state
            + 2 * self.d_model * self.d_state
            + self.d_model
            + self.d_state,
            bias=False,
            **factory_kwargs,
        )

        # Gating projection
        self.gate_proj = nn.Linear(
            self.d_model, self.d_model, bias=False, **factory_kwargs
        )

        self._init_weights()

    def _init_weights(self):
        # A portion should start small so A_bar ~= -1 initially
        # B, C small initialization, D starts near 1
        # The original paper does not mention exact initialization details
        nn.init.normal_(self.x_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        inference_cache: Optional[Dict[str, Any]] = None,
    ):
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, seqlen, dim), got {hidden_states.dim()}D"
            )

        batch, seqlen, dim = hidden_states.shape

        if inference_cache is not None:
            seqlen_offset = inference_cache.get("seqlen_offset", 0)
            if seqlen_offset > 0:
                out, inference_cache = self.step(
                    hidden_states, inference_cache
                )
                return out

        if self.use_fast_path:
            out = s7_inner_fn(
                hidden_states,
                self.in_proj.weight,
                self.x_proj.weight,
                self.gate_proj.weight,
                self.d_state,
                self.base_params,
            )
        else:
            # Slow path: explicit math
            x = self.in_proj(hidden_states)

            x_dbl = self.x_proj(rearrange(x, "b l d -> (b l) d"))
            A, B, C, D, bias = torch.split(
                x_dbl,
                [
                    self.d_state,
                    self.d_model * self.d_state,
                    self.d_model * self.d_state,
                    self.d_model,
                    self.d_state,
                ],
                dim=-1,
            )

            A = rearrange(A, "(b l) n -> b n l", l=seqlen)
            A = A + self.base_params.unsqueeze(0).unsqueeze(
                -1
            )  # Add HiPPO initialization to A
            A = A.contiguous()
            B = rearrange(
                B, "(b l) (h n) -> b n h l", l=seqlen, n=self.d_state
            ).contiguous()
            C = rearrange(
                C, "(b l) (h n) -> b h n l", l=seqlen, n=self.d_state
            ).contiguous()
            D_tv = rearrange(D, "(b l) h -> b h l", l=seqlen)
            bias = rearrange(bias, "(b l) n -> b n l", l=seqlen)

            u = rearrange(x, "b l h -> b h l")

            y = s7_scan_fn(u, A, B, C, bias)
            y = y + D_tv * u

            y = rearrange(y, "b h l -> b l h")
            gate = torch.sigmoid(self.gate_proj(F.gelu(y)))
            y = gate * y

            out = y + hidden_states

        return out

    def step(
        self,
        hidden_states: Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        ssm_state = inference_cache["lrnn_state"]
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1

        x_in = hidden_states.squeeze(1)
        x = self.in_proj(x_in)

        x_dbl = self.x_proj(x)
        A, B, C, D, bias = torch.split(
            x_dbl,
            [
                self.d_state,
                self.d_model * self.d_state,
                self.d_model * self.d_state,
                self.d_model,
                self.d_state,
            ],
            dim=-1,
        )

        A = A + self.base_params  # Add HiPPO initialization to A
        B = rearrange(B, "b (h n) -> b h n", n=self.d_state)
        C = rearrange(C, "b (h n) -> b h n", n=self.d_state)

        # Pre-compute Bu = B^T @ x + bias  (reduction over dim)
        Bu = torch.einsum("bhn,bh->bn", B, x) + bias  # (batch, dstate)

        if selective_state_update is not None:
            # Map S7 tensors to the unified selective_state_update interface:
            # The kernel computes A_bar from dt via DISCRETIZATION="s7",
            # uses identity dB (Bu already contains B^T@x+bias),
            # and returns the raw new state values (C_kernel=1).
            batch = Bu.shape[0]
            state_3d = ssm_state.unsqueeze(-1)
            A_kernel = torch.zeros(
                self.d_state, 1, device=x.device, dtype=x.dtype
            )
            BC_ones = torch.ones(batch, 1, device=x.device, dtype=x.dtype)
            new_state_vals = selective_state_update(
                state_3d,
                Bu,
                A,
                A_kernel,
                BC_ones,
                BC_ones,
                discretization="s7",
            )
        else:
            # Pure PyTorch fallback
            A_sq_half = A * A + 0.5
            A_bar = 1.0 - 1.0 / A_sq_half
            new_state_vals = A_bar * ssm_state + Bu
            ssm_state.copy_(new_state_vals)

        # Post-compute: y = C @ new_state + D * x
        y = torch.einsum("bhn,bn->bh", C, new_state_vals) + D * x

        gate = torch.sigmoid(self.gate_proj(F.gelu(y)))
        y = gate * y
        out = y + x_in

        inference_cache["seqlen_offset"] += 1

        return out.unsqueeze(1).to(dtype), inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        device = self.in_proj.weight.device
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {
            "lrnn_state": ssm_state,
            "seqlen_offset": 0,
        }
