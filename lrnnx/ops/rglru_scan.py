"""
RG-LRU (Recurrent Gated Linear Recurrent Unit) Scan Operation.
This module exposes 2 levels of the scan similar to Mamba.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

import selective_scan_cuda

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import (
        causal_conv1d_bwd_function,
        causal_conv1d_fwd_function,
    )
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None

from lrnnx.ops.torch import custom_bwd, custom_fwd


class RGLRUScanFn(torch.autograd.Function):
    """
    Thin autograd wrapper around the RGLRU CUDA kernel.

    All gating pre-computations must be done *before* calling this.
    """

    @staticmethod
    def forward(
        ctx,
        u,
        delta,
        A,
        return_last_state=False,
    ):
        """
        Forward pass for the RG-LRU Scan CUDA kernel.

        Args:
            ctx (Any): Autograd context.
            u (torch.Tensor): Pre-gated input of shape ``(batch, dim, seqlen)`` in float32.
            delta (torch.Tensor): Pre-computed exponent of shape ``(batch, dim, seqlen)`` in float32.
            A (torch.Tensor): Learnable recurrence base in (0, 1), shape ``(dim, dstate)``.
            return_last_state (bool, optional): Whether to return the last hidden state. Defaults to False.
        
        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The output tensor, and optionally the last state.
        """
        if not u.is_contiguous():
            u = u.contiguous()
        if not delta.is_contiguous():
            delta = delta.contiguous()
        if not A.is_contiguous():
            A = A.contiguous()

        dim, dstate = A.shape

        # Identity B / C  (time-invariant, not learnable)
        B = torch.ones(
            dim, dstate, dtype=u.dtype, device=u.device
        ).contiguous()
        C = torch.ones(
            dim, dstate, dtype=u.dtype, device=u.device
        ).contiguous()

        out, x, *_ = selective_scan_cuda.fwd(
            u,
            delta,
            A,
            B,
            C,
            None,  # D
            None,  # z
            None,  # delta_bias
            None,  # deltaA
            False,  # delta_softplus
            "rglru",
        )

        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)

        ctx.save_for_backward(u, delta, A, B, C, out, x)

        if return_last_state:
            return out, last_state
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, out, x = ctx.saved_tensors

        if not dout.is_contiguous():
            dout = dout.contiguous()

        du, ddelta, dA, *_ = selective_scan_cuda.bwd(
            u,
            delta,
            A,
            B,
            C,
            None,  # D
            None,  # z
            None,  # delta_bias
            None,  # deltaA
            dout,
            x,
            out,
            None,  # dz
            False,  # delta_softplus
            False,  # recompute_out
            "rglru",
        )

        return (
            du,
            ddelta,
            dA,
            None,  # return_last_state
        )


def rglru_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RG-LRU scan - thin CUDA kernel wrapper.

    All inputs must already be in float32.

    Args:
        u (torch.Tensor): Pre-gated input of shape ``(batch, dim, seqlen)``.
        delta (torch.Tensor): Pre-computed exponent of shape ``(batch, dim, seqlen)``.
        A (torch.Tensor): Learnable recurrence base in (0, 1), shape ``(dim, dstate)``.
        return_last_state (bool, optional): Whether to return last hidden state. Defaults to False.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: 
            - out (torch.Tensor): Output tensor of shape ``(batch, dim, seqlen)``.
            - last_state (torch.Tensor, optional): If ``return_last_state`` is True, shape ``(batch, dim, dstate)``.
    """
    return RGLRUScanFn.apply(u, delta, A, return_last_state)


def rglru_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Reference RG-LRU scan (pure PyTorch, sequential loop).

    Args:
        u (torch.Tensor): Pre-gated input of shape ``(batch, dim, seqlen)`` in float32.
        delta (torch.Tensor): Pre-computed exponent of shape ``(batch, dim, seqlen)`` in float32.
        A (torch.Tensor): Learnable recurrence base in (0, 1), shape ``(dim, dstate)`` in float32.
        return_last_state (bool, optional): Whether to return last hidden state. Defaults to False.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            - out (torch.Tensor): Output tensor of shape ``(batch, dim, seqlen)``.
            - last_state (torch.Tensor, optional): If ``return_last_state`` is True, shape ``(batch, dim, dstate)``.
    """
    dtype_in = u.dtype
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    u = u.float()
    delta = delta.float()
    A = A.float()

    # A_bar = A ^ delta  via log-space for stability
    log_A = torch.log(A)  # (dim, dstate)
    # delta: (B, D, L) -> (B, D, 1, L);  log_A: (D, N) -> (1, D, N, 1)
    A_bar = torch.exp(
        delta.unsqueeze(2) * log_A.unsqueeze(0).unsqueeze(-1)
    )  # (B, D, N, L)

    sqrt_term = torch.sqrt(1.0 - A_bar * A_bar)  # (B, D, N, L)

    # B_u = sqrt(1 - A_bar^2) * u
    B_u = sqrt_term * u.unsqueeze(2)  # (B, D, N, L)

    # Sequential scan
    h = torch.zeros(batch, dim, dstate, dtype=torch.float32, device=u.device)
    hs = []
    for t in range(seqlen):
        h = A_bar[:, :, :, t] * h + B_u[:, :, :, t]
        hs.append(h)

    last_state = h
    h_seq = torch.stack(hs, dim=-1)  # (B, D, N, L)

    # C = identity ones → y = sum over dstate
    y = h_seq.sum(dim=2)  # (B, D, L)

    out = y.to(dtype_in)
    if return_last_state:
        return out, last_state
    return out


class RGLRUInnerFn(torch.autograd.Function):
    """
    RG-LRU inner function: conv1d + gate projections + gating + scan + output.

    Performs:
        x              = causal_conv1d(x_pre_conv)
        recurrent_gate = sigmoid(x @ W_r^T + b_r)
        input_gate     = sigmoid(x @ W_i^T + b_i)
        delta          = c x recurrent_gate
        u_gated        = input_gate x x
        y              = rglru_scan(u_gated, delta, a)
        out            = (gate x y) @ W_out^T + b_out
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x,
        conv1d_weight,
        conv1d_bias,
        a,
        recurrent_gate_weight,
        recurrent_gate_bias,
        input_gate_weight,
        input_gate_bias,
        out_proj_weight,
        out_proj_bias,
        gate,
        c=8.0,
    ):
        """
        Forward pass for the RG-LRU inner function.

        Args:
            ctx (Any): Autograd context.
            x (torch.Tensor): Input before conv of shape ``(batch, dim, seqlen)``.
            conv1d_weight (torch.Tensor): Conv1d weight of shape ``(dim, 1, kernel_size)``.
            conv1d_bias (torch.Tensor | None): Conv1d bias of shape ``(dim,)`` or None.
            a (torch.Tensor): Learnable recurrence base in (0, 1), shape ``(dim,)`` or ``(dim, dstate)``.
            recurrent_gate_weight (torch.Tensor): Recurrent gate weight of shape ``(dim, dim)``.
            recurrent_gate_bias (torch.Tensor): Recurrent gate bias of shape ``(dim,)``.
            input_gate_weight (torch.Tensor): Input gate weight of shape ``(dim, dim)``.
            input_gate_bias (torch.Tensor): Input gate bias of shape ``(dim,)``.
            out_proj_weight (torch.Tensor): Output projection weight of shape ``(d_model, dim)``.
            out_proj_bias (torch.Tensor | None): Output projection bias of shape ``(d_model,)`` or None.
            gate (torch.Tensor): Stream-1 gate of shape ``(batch, seqlen, dim)``.
            c (float, optional): Fixed scalar constant. Defaults to 8.0.

        Returns:
            torch.Tensor: The projected output tensor.
        """
        assert (
            causal_conv1d_fn is not None
        ), "causal_conv1d_cuda is not available. Please install causal-conv1d."
        dtype_in = x.dtype
        batch = x.shape[0]
        L = x.shape[-1]

        a_was_1d = a.dim() == 1
        if a_was_1d:
            a = a.unsqueeze(-1)  # (dim,) -> (dim, 1)

        if x.stride(-1) != 1:
            x = x.contiguous()
        a_f = a.float().contiguous()

        # Causal conv1d
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        conv1d_bias = (
            conv1d_bias.contiguous() if conv1d_bias is not None else None
        )
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, False
        )
        x_f = conv1d_out.float()

        # Gate projections: conv1d_out (B, D, L) → (BL, D) → linear → sigmoid
        x_flat = rearrange(x_f, "b d l -> (b l) d")  # (BL, D)

        rg_pre = F.linear(
            x_flat,
            recurrent_gate_weight.float(),
            recurrent_gate_bias.float(),
        )
        recurrent_gate = torch.sigmoid(rg_pre)  # (BL, D)

        ig_pre = F.linear(
            x_flat,
            input_gate_weight.float(),
            input_gate_bias.float(),
        )
        input_gate = torch.sigmoid(ig_pre)  # (BL, D)

        recurrent_gate_bdl = rearrange(
            recurrent_gate, "(b l) d -> b d l", b=batch
        ).contiguous()
        input_gate_bdl = rearrange(
            input_gate, "(b l) d -> b d l", b=batch
        ).contiguous()

        delta = (c * recurrent_gate_bdl).contiguous()  # (B, D, L)
        u_gated = (input_gate_bdl * x_f).contiguous()  # (B, D, L)

        dim, dstate = a_f.shape

        # Identity B / C placeholders
        B = torch.ones(dim, dstate, dtype=torch.float32, device=x.device)
        C = torch.ones(dim, dstate, dtype=torch.float32, device=x.device)

        out, x_states, *_ = selective_scan_cuda.fwd(
            u_gated,
            delta,
            a_f,
            B,
            C,
            None,  # D
            None,  # z
            None,  # delta_bias
            None,  # deltaA
            False,  # delta_softplus
            "rglru",
        )

        # Merge with stream-1 gate and project out
        y = rearrange(out, "b d l -> b l d")  # (B, L, D)
        gate_f = gate.float()
        result = F.linear(
            gate_f * y,
            out_proj_weight.float(),
            out_proj_bias.float() if out_proj_bias is not None else None,
        )

        ctx.save_for_backward(
            x,  # pre-conv input for conv1d backward
            conv1d_weight,
            conv1d_bias,
            conv1d_out,
            a_f,
            recurrent_gate,
            input_gate,  # (BL, D) each
            u_gated,
            delta,
            out,
            x_states,
            B,
            C,
            out_proj_weight.float(),
            gate_f,
            recurrent_gate_weight.float(),
            input_gate_weight.float(),
        )
        ctx.c = c
        ctx.a_was_1d = a_was_1d
        ctx.dtype_in = dtype_in
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.batch = batch
        ctx.L = L

        return result.to(dtype_in)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        (
            x_pre_conv,
            conv1d_weight,
            conv1d_bias,
            conv1d_out,
            a_f,
            recurrent_gate,
            input_gate,
            u_gated,
            delta,
            scan_out,
            x_states,
            B,
            C,
            out_proj_weight,
            gate_f,
            rg_weight,
            ig_weight,
        ) = ctx.saved_tensors
        c = ctx.c
        dtype_in = ctx.dtype_in
        batch = ctx.batch
        L = ctx.L

        dout = dout.float()
        if not dout.is_contiguous():
            dout = dout.contiguous()

        x_f = conv1d_out.float()

        y = rearrange(scan_out, "b d l -> b l d")  # (B, L, D)
        gate_y = gate_f * y  # (B, L, D)

        dout_2d = rearrange(dout, "b l e -> (b l) e")  # (BL, D_model)
        gate_y_2d = rearrange(gate_y, "b l d -> (b l) d")  # (BL, D)

        d_gate_y_2d = dout_2d @ out_proj_weight  # (BL, D)
        d_out_proj_weight = dout_2d.t() @ gate_y_2d  # (D_model, D)
        d_out_proj_bias = (
            dout_2d.sum(0) if not ctx.out_proj_bias_is_None else None
        )

        # Backward through gate * y
        d_gate_y = rearrange(d_gate_y_2d, "(b l) d -> b l d", b=batch)
        d_gate = d_gate_y * y  # (B, L, D)
        dy = d_gate_y * gate_f  # (B, L, D)

        # Backward through rearrange → CUDA scan backward
        dout_scan = rearrange(dy, "b l d -> b d l").contiguous()

        du_gated, ddelta, dA, *_ = selective_scan_cuda.bwd(
            u_gated,
            delta,
            a_f,
            B,
            C,
            None,  # D
            None,  # z
            None,  # delta_bias
            None,  # deltaA
            dout_scan,
            x_states,
            scan_out,
            None,  # dz
            False,  # delta_softplus
            False,  # recompute_out
            "rglru",
        )

        # Chain rule: u_gated = input_gate_bdl * x_f
        input_gate_bdl = rearrange(input_gate, "(b l) d -> b d l", b=batch)
        dconv1d_out = du_gated * input_gate_bdl  # (B, D, L)
        d_input_gate_bdl = du_gated * x_f  # (B, D, L)

        # Chain rule: delta = c * recurrent_gate_bdl
        d_recurrent_gate_bdl = ddelta * c  # (B, D, L)

        # Reshape to (BL, D) for sigmoid + linear backward
        d_input_gate_2d = rearrange(d_input_gate_bdl, "b d l -> (b l) d")
        d_recurrent_gate_2d = rearrange(
            d_recurrent_gate_bdl, "b d l -> (b l) d"
        )

        # Backward through sigmoid
        d_ig_pre = d_input_gate_2d * input_gate * (1 - input_gate)
        d_rg_pre = d_recurrent_gate_2d * recurrent_gate * (1 - recurrent_gate)

        # Backward through linear projections
        x_flat = rearrange(x_f, "b d l -> (b l) d")

        d_ig_weight = d_ig_pre.t() @ x_flat  # (D, D)
        d_ig_bias = d_ig_pre.sum(0)  # (D,)
        dconv1d_out_from_ig = d_ig_pre @ ig_weight  # (BL, D)

        d_rg_weight = d_rg_pre.t() @ x_flat  # (D, D)
        d_rg_bias = d_rg_pre.sum(0)  # (D,)
        dconv1d_out_from_rg = d_rg_pre @ rg_weight  # (BL, D)

        # Total dconv1d_out
        dconv1d_out_from_proj = rearrange(
            dconv1d_out_from_ig + dconv1d_out_from_rg,
            "(b l) d -> b d l",
            b=batch,
        )
        dconv1d_out = dconv1d_out + dconv1d_out_from_proj  # (B, D, L)

        # Backward through causal conv1d
        dx = torch.empty_like(x_pre_conv)
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x_pre_conv,
            conv1d_weight,
            conv1d_bias,
            dconv1d_out.to(x_pre_conv.dtype),
            None,
            None,
            None,
            dx,
            False,
            False,
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")

        da = dA
        if ctx.a_was_1d:
            da = da.squeeze(-1)

        return (
            dx.to(dtype_in),  # x
            dconv1d_weight.to(dtype_in),  # conv1d_weight
            (
                dconv1d_bias.to(dtype_in) if dconv1d_bias is not None else None
            ),  # conv1d_bias
            da,  # a
            d_rg_weight.to(dtype_in),  # recurrent_gate_weight
            d_rg_bias.to(dtype_in),  # recurrent_gate_bias
            d_ig_weight.to(dtype_in),  # input_gate_weight
            d_ig_bias.to(dtype_in),  # input_gate_bias
            d_out_proj_weight.to(dtype_in),  # out_proj_weight
            (
                d_out_proj_bias.to(dtype_in)
                if d_out_proj_bias is not None
                else None
            ),  # out_proj_bias
            d_gate.to(dtype_in),  # gate
            None,  # c
        )


def rglru_inner_fn(
    x: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    a: torch.Tensor,
    recurrent_gate_weight: torch.Tensor,
    recurrent_gate_bias: torch.Tensor,
    input_gate_weight: torch.Tensor,
    input_gate_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    gate: torch.Tensor,
    c: float = 8.0,
) -> torch.Tensor:
    """
    RG-LRU inner function (CUDA).

    Computes conv1d, gate projections, gating, scan, and output projection::

        x_conv         = causal_conv1d(x)
        recurrent_gate = sigmoid(x_conv @ W_r^T + b_r)
        input_gate     = sigmoid(x_conv @ W_i^T + b_i)
        delta          = c x recurrent_gate
        u_gated        = input_gate x x_conv
        y              = rglru_scan(u_gated, delta, a)
        out            = (gate x y) @ W_out^T + b_out

    Args:
        x (torch.Tensor): Input before conv, shape ``(batch, dim, seqlen)``.
        conv1d_weight (torch.Tensor): Conv1d weight, shape ``(dim, 1, kernel_size)``.
        conv1d_bias (torch.Tensor | None): Conv1d bias, shape ``(dim,)`` or None.
        a (torch.Tensor): Learnable recurrence base in (0, 1), shape ``(dim,)`` or ``(dim, dstate)``.
        recurrent_gate_weight (torch.Tensor): Recurrent gate weight, shape ``(dim, dim)``.
        recurrent_gate_bias (torch.Tensor): Recurrent gate bias, shape ``(dim,)``.
        input_gate_weight (torch.Tensor): Input gate weight, shape ``(dim, dim)``.
        input_gate_bias (torch.Tensor): Input gate bias, shape ``(dim,)``.
        out_proj_weight (torch.Tensor): Output projection weight, shape ``(d_model, dim)``.
        out_proj_bias (torch.Tensor | None): Output projection bias, shape ``(d_model,)`` or None.
        gate (torch.Tensor): Stream-1 gate, shape ``(batch, seqlen, dim)``.
        c (float, optional): Fixed scalar constant. Defaults to 8.0.

    Returns:
        torch.Tensor: Output tensor of shape ``(batch, seqlen, d_model)``.
    """
    return RGLRUInnerFn.apply(
        x,
        conv1d_weight,
        conv1d_bias,
        a,
        recurrent_gate_weight,
        recurrent_gate_bias,
        input_gate_weight,
        input_gate_bias,
        out_proj_weight,
        out_proj_bias,
        gate,
        c,
    )


def rglru_inner_ref(
    x: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    a: torch.Tensor,
    recurrent_gate_weight: torch.Tensor,
    recurrent_gate_bias: torch.Tensor,
    input_gate_weight: torch.Tensor,
    input_gate_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    gate: torch.Tensor,
    c: float = 8.0,
) -> torch.Tensor:
    """
    Reference RG-LRU inner function (pure PyTorch).

    Computes:
        x_conv         = conv1d(x)[..., :L]
        recurrent_gate = sigmoid(x_conv @ W_r^T + b_r)
        input_gate     = sigmoid(x_conv @ W_i^T + b_i)
        gate_t         = c x recurrent_gate_t
        A_bar_t        = a ** gate_t
        h_t            = A_bar_t x h_{t-1} + sqrt(1 - A_bar_t**2) x (input_gate_t x u_t)
        y_t            = sum_n h_n,t
        out            = (gate x y) @ W_out^T + b_out

    Args:
        x (torch.Tensor): Input before conv, shape ``(batch, dim, seqlen)``.
        conv1d_weight (torch.Tensor): Conv1d weight, shape ``(dim, 1, kernel_size)``.
        conv1d_bias (torch.Tensor | None): Conv1d bias, shape ``(dim,)`` or None.
        a (torch.Tensor): Learnable recurrence base in (0, 1), shape ``(dim,)`` or ``(dim, dstate)``.
        recurrent_gate_weight (torch.Tensor): Recurrent gate weight, shape ``(dim, dim)``.
        recurrent_gate_bias (torch.Tensor): Recurrent gate bias, shape ``(dim,)``.
        input_gate_weight (torch.Tensor): Input gate weight, shape ``(dim, dim)``.
        input_gate_bias (torch.Tensor): Input gate bias, shape ``(dim,)``.
        out_proj_weight (torch.Tensor): Output projection weight, shape ``(d_model, dim)``.
        out_proj_bias (torch.Tensor | None): Output projection bias, shape ``(d_model,)`` or None.
        gate (torch.Tensor): Stream-1 gate, shape ``(batch, seqlen, dim)``.
        c (float, optional): Fixed scalar constant. Defaults to 8.0.
        
    Returns:
        torch.Tensor: Output tensor of shape ``(batch, seqlen, d_model)``.
    """
    dtype_in = x.dtype
    L = x.shape[-1]

    x = x.float()
    a = a.float()
    gate = gate.float()

    if a.dim() == 1:
        a = a.unsqueeze(-1)  # (dim,) -> (dim, 1)

    # Conv1d (depthwise, causal via padding + truncation)
    conv1d_weight_f = conv1d_weight.float()
    d_conv = conv1d_weight.shape[-1]
    x_padded = F.pad(x, (d_conv - 1, 0))
    x_conv = F.conv1d(
        x_padded,
        conv1d_weight_f,
        conv1d_bias.float() if conv1d_bias is not None else None,
        groups=x.shape[1],
    )

    # Gate projections
    batch = x.shape[0]
    x_flat = rearrange(x_conv, "b d l -> (b l) d")
    recurrent_gate = torch.sigmoid(
        F.linear(
            x_flat,
            recurrent_gate_weight.float(),
            recurrent_gate_bias.float(),
        )
    )
    input_gate = torch.sigmoid(
        F.linear(
            x_flat,
            input_gate_weight.float(),
            input_gate_bias.float(),
        )
    )
    recurrent_gate = rearrange(
        recurrent_gate, "(b l) d -> b d l", b=batch
    ).contiguous()
    input_gate = rearrange(
        input_gate, "(b l) d -> b d l", b=batch
    ).contiguous()

    # Projections
    delta = c * recurrent_gate  # (B, D, L)
    u_gated = input_gate * x_conv  # (B, D, L)

    result = rglru_scan_ref(u_gated, delta, a)

    # Merge with stream-1 gate and project out
    y = rearrange(result, "b d l -> b l d")
    out = F.linear(
        gate * y,
        out_proj_weight.float(),
        out_proj_bias.float() if out_proj_bias is not None else None,
    )
    return out.to(dtype_in)