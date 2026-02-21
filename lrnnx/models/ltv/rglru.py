"""
RG-LRU (Recurrent Gated Linear Recurrent Unit) block.
https://arxiv.org/abs/2402.19427
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from lrnnx.models.ltv.base import LTV_LRNN
from lrnnx.ops.rglru_scan import rglru_inner_fn, rglru_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from lrnnx.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


class RGLRU(LTV_LRNN):
    """
    RG-LRU block following the Griffin architecture.

    Example:
        >>> model = RGLRU(d_model=64, d_state=1, d_conv=4)
        >>> x = torch.randn(2, 128, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 128, 64])
    """

    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        expand: int = 1,
        c: float = 8.0,
        a_init_range: Tuple[float, float] = (0.9, 0.999),
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        """
        Initialize RG-LRU block.

        Args:
            d_model (int): Model dimension.
            d_conv (int, optional): Temporal convolution kernel size. Defaults to 4.
            expand (int, optional): Expansion factor for inner dimension. Defaults to 1.
            c (float, optional): Fixed scalar for recurrent gate scaling. Defaults to 8.0.
            a_init_range (Tuple[float, float], optional): Tuple ``(lo, hi)`` so *a* is
                initialised in ``[lo, hi]`` in ``(0, 1)``. Defaults to ``(0.9, 0.999)``.
            conv_bias (bool, optional): Whether the Conv1D uses a bias term. Defaults to True.
            bias (bool, optional): Whether Linear projections use bias. Defaults to False.
            use_fast_path (bool, optional): Use the fused CUDA kernel when available. Defaults to True.
            layer_idx (int, optional): Layer index (for multi-layer caching). Defaults to None.
            device (torch.device, optional): Device for parameters. Defaults to None.
            dtype (torch.dtype, optional): Data type for parameters. Defaults to None.
        """
        # RG-LRU handles discretisation internally
        super().__init__(discretization=None)
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_conv = d_conv
        self.expand = expand
        self.dstate = 1
        self.d_inner = int(self.expand * self.d_model)
        self.c = c
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # Stream 1: Linear -> GeLU
        self.gate_proj = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )

        # Stream 2: Linear -> Conv1D -> RG-LRU
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )

        # Recurrent / input gate projections
        self.recurrent_gate_proj = nn.Linear(
            self.d_inner, self.d_inner, bias=True, **factory_kwargs
        )
        self.input_gate_proj = nn.Linear(
            self.d_inner, self.d_inner, bias=True, **factory_kwargs
        )

        # Learnable recurrence base a in (0, 1), shape (d_inner, d_state)
        a_lo, a_hi = a_init_range
        a_init = a_lo + (a_hi - a_lo) * torch.rand(
            self.d_inner, self.dstate, **factory_kwargs
        )
        self.a_log = nn.Parameter(torch.log(a_init))
        self.a_log._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(
        self,
        hidden_states: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        inference_cache: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Forward pass through the RG-LRU block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, L, D)``.
            integration_timesteps (torch.Tensor, optional): *Unused* - kept for LTV interface compat. Defaults to None.
            lengths (torch.Tensor, optional): *Unused* - kept for interface compatibility. Defaults to None.
            inference_cache (Dict[str, Any], optional): Cache dict for autoregressive generation. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``(B, L, D)``.
        """
        batch, seqlen, dim = hidden_states.shape

        if inference_cache is not None:
            seqlen_offset = inference_cache.get("seqlen_offset", 0)
            if seqlen_offset > 0:
                out, inference_cache = self.step(
                    hidden_states, inference_cache
                )
                return out

        # Stream 1: gate path
        gate = F.gelu(self.gate_proj(hidden_states))  # (B, L, D_inner)

        # Stream 2: conv -> RG-LRU
        x = self.in_proj(hidden_states)  # (B, L, D_inner)
        x = rearrange(x, "b l d -> b d l")

        # Learnable base in (0, 1)
        a = torch.sigmoid(self.a_log)  # (d_inner, d_state)

        if (
            self.use_fast_path
            and causal_conv1d_fn is not None
            and inference_cache is None
        ):
            out = rglru_inner_fn(
                x,
                self.conv1d.weight,
                self.conv1d.bias,
                a,
                self.recurrent_gate_proj.weight,
                self.recurrent_gate_proj.bias,
                self.input_gate_proj.weight,
                self.input_gate_proj.bias,
                self.out_proj.weight,
                self.out_proj.bias,
                gate,
                c=self.c,
            )
        else:
            # Causal temporal convolution
            if causal_conv1d_fn is not None:
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=None,
                )
            else:
                x = self.conv1d(x)[..., :seqlen]

            # Update conv cache if present
            conv_state = (
                inference_cache.get("conv_state") if inference_cache else None
            )
            if conv_state is not None:
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

            # Gate projections  (B, D_inner, L) -> transpose -> project -> back
            x_BLD = rearrange(x, "b d l -> (b l) d")
            recurrent_gate = torch.sigmoid(self.recurrent_gate_proj(x_BLD))
            input_gate = torch.sigmoid(self.input_gate_proj(x_BLD))
            recurrent_gate = rearrange(
                recurrent_gate, "(b l) d -> b d l", l=seqlen
            ).contiguous()
            input_gate = rearrange(
                input_gate, "(b l) d -> b d l", l=seqlen
            ).contiguous()

            ssm_state = (
                inference_cache.get("lrnn_state") if inference_cache else None
            )

            # Manual path: gating + scan
            delta = (self.c * recurrent_gate).float().contiguous()
            u_gated = (input_gate * x).float().contiguous()

            y = rglru_scan_fn(
                u_gated,
                delta,
                a,
                return_last_state=ssm_state is not None,
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)

            y = rearrange(y, "b d l -> b l d")

            # Merge streams and project out
            out = self.out_proj(gate * y)
        return out

    def step(
        self,
        hidden_states: Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Single recurrent step for autoregressive inference.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape ``(B, 1, D)``.
            inference_cache (Dict[str, Any]): Must contain conv_state, lrnn_state,
                and seqlen_offset.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: Tuple containing:
                - out : Output tensor of shape ``(B, 1, D)``.
                - inference_cache : Updated cache dictionary.
        """
        conv_state = inference_cache["conv_state"]
        ssm_state = inference_cache["lrnn_state"]
        dtype = hidden_states.dtype

        assert (
            hidden_states.shape[1] == 1
        ), "step() supports single-token decoding only"
        x_in = hidden_states.squeeze(1)  # (B, D)

        # Stream 1
        gate = F.gelu(self.gate_proj(x_in))  # (B, D_inner)

        # Stream 2
        x = self.in_proj(x_in)  # (B, D_inner)

        # Conv step
        if causal_conv1d_update is not None:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None,
            )
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"),
                dim=-1,
            )
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias

        # Gate projections
        recurrent_gate = torch.sigmoid(
            self.recurrent_gate_proj(x)
        )  # (B, D_inner)
        input_gate = torch.sigmoid(self.input_gate_proj(x))  # (B, D_inner)

        a = torch.sigmoid(self.a_log)  # (d_inner, d_state)

        # Pre-compute gate and gated input for the RG-LRU recurrence
        gate_val = self.c * recurrent_gate  # (B, D_inner)
        u_gated = input_gate * x  # (B, D_inner)

        if selective_state_update is not None:
            # Triton fused path: pass pre-computed gate via dt, gated input via x,
            # A (base in (0,1)) and identity B/C.
            B_ones = torch.ones(
                u_gated.shape[0],
                self.dstate,
                device=u_gated.device,
                dtype=u_gated.dtype,
            )
            C_ones = torch.ones_like(B_ones)
            y = selective_state_update(
                ssm_state,
                u_gated,
                gate_val,
                a,
                B_ones,
                C_ones,
                dt_bias=None,
                dt_softplus=False,
                discretization="rglru",
            )
        else:
            # fallback
            # a: (D, N), gate_val: (B, D) -> a_bar: (B, D, N)
            a_bar = a.unsqueeze(0).pow(gate_val.unsqueeze(-1))  # (B, D, N)
            sqrt_term = torch.sqrt(1.0 - a_bar * a_bar)
            new_state = a_bar * ssm_state + sqrt_term * u_gated.unsqueeze(-1)
            y = new_state.sum(dim=-1)  # (B, D_inner) - sum over dstate
            ssm_state.copy_(new_state)

        # Merge and project out
        out = self.out_proj(gate * y)

        inference_cache["conv_state"] = conv_state
        inference_cache["lrnn_state"] = ssm_state
        inference_cache["seqlen_offset"] = (
            inference_cache.get("seqlen_offset", 0) + 1
        )

        return out.unsqueeze(1).to(dtype), inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocate cache for autoregressive inference.

        Args:
            batch_size (int): Batch size.
            max_seqlen (int): Unused, kept for interface consistency.
            dtype (torch.dtype, optional): Data type for cache tensors. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Cache dictionary containing "conv_state", "ssm_state", and "seqlen_offset".
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype

        conv_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_state = torch.zeros(
            batch_size,
            self.d_inner,
            self.dstate,
            device=device,
            dtype=ssm_dtype,
        )
        return {
            "conv_state": conv_state,
            "lrnn_state": ssm_state,
            "seqlen_offset": 0,
        }