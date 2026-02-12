"""
S7 Scan Operation.
This module exposes 2 levels of the scan similar to Mamba.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

import selective_scan_cuda
from lrnnx.ops.torch import custom_bwd, custom_fwd


class S7ScanFn(torch.autograd.Function):
    """Autograd function for S7 scan with time-varying A, B, C."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        u,
        A,
        B,
        C,
        bias=None,
        return_last_state=False,
    ):
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]

        if not u.is_contiguous():
            u = u.contiguous()

        Bu = torch.einsum("bnhl,bhl->bnl", B.float(), u.float())
        if bias is not None:
            Bu = Bu + bias.float()
        Bu = Bu.contiguous()

        delta = A.contiguous()
        A_kernel = torch.zeros(
            dstate, 1, dtype=Bu.dtype, device=Bu.device
        ).contiguous()
        B_kernel = torch.ones(
            dstate, 1, dtype=Bu.dtype, device=Bu.device
        ).contiguous()
        C_kernel = torch.ones(
            dstate, 1, dtype=Bu.dtype, device=Bu.device
        ).contiguous()

        out_kernel, x_kernel, *_ = selective_scan_cuda.fwd(
            Bu,
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
            None,
            None,
            None,
            None,
            False,
            "s7",
        )

        y = torch.einsum("bhnl,bnl->bhl", C.float(), out_kernel.float())
        last_state = out_kernel[:, :, -1]

        ctx.save_for_backward(
            u,
            B,
            C,
            bias,
            Bu,
            out_kernel,
            x_kernel,
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
        )
        if return_last_state:
            return y.to(u.dtype), last_state
        return y.to(u.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        (
            u,
            B,
            C,
            bias,
            Bu,
            out_kernel,
            x_kernel,
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
        ) = ctx.saved_tensors

        dout = dout.float()
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        dC = torch.einsum("bhl,bnl->bhnl", dout, out_kernel.float())
        dout_kernel = torch.einsum(
            "bhnl,bhl->bnl", C.float(), dout
        ).contiguous()

        dBu, ddelta, *_ = selective_scan_cuda.bwd(
            Bu.contiguous(),
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
            None,
            None,
            None,
            None,
            dout_kernel,
            x_kernel,
            out_kernel,
            None,
            False,
            False,
            "s7",
        )

        dA = ddelta
        dB = torch.einsum("bnl,bhl->bnhl", dBu, u.float())
        du = torch.einsum("bnhl,bnl->bhl", B.float(), dBu)
        dbias = dBu if bias is not None else None

        return (
            du.to(u.dtype),
            dA,
            dB,
            dC,
            dbias,
            None,  # return_last_state
        )


def s7_scan_fn(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor | None = None,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    S7 scan using CUDA kernel.

    Args:
        u: Input (batch, dim, seqlen) float32
        A: Time-varying state transition (batch, dstate, seqlen) float32
        B: Time-varying input projection (batch, dstate, dim, seqlen) float32
        C: Time-varying output projection (batch, dim, dstate, seqlen) float32
        bias: Optional LTV bias (batch, dstate, seqlen) float32
        return_last_state: Whether to return last hidden state

    Returns:
        out: Output (batch, dim, seqlen) float32
        last_state: If return_last_state=True, (batch, dstate) float32
    """
    return S7ScanFn.apply(
        u,
        A,
        B,
        C,
        bias,
        return_last_state,
    )


def s7_scan_ref(
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: torch.Tensor | None = None,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of S7 scan (pure PyTorch).

    Args:
        u: Input (batch, dim, seqlen) float32
        A: Time-varying state transition (batch, dstate, seqlen) float32
        B: Time-varying projection (batch, dstate, dim, seqlen) float32
        C: Time-varying projection (batch, dim, dstate, seqlen) float32
        bias: Optional LTV bias (batch, dstate, seqlen) float32
        return_last_state: Whether to return the last hidden state

    Returns:
        out: Output (batch, dim, seqlen) float32
        last_state: If return_last_state=True, (batch, dstate) float32
    """
    dtype_in = u.dtype
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    u = u.float()
    A = A.float()
    B = B.float()
    C = C.float()

    A_sq_half = A * A + 0.5
    A_bar = 1.0 - 1.0 / A_sq_half  # (batch, dstate, seqlen)

    Bu = torch.einsum("bnhl,bhl->bnl", B, u)
    if bias is not None:
        Bu = Bu + bias

    x = torch.zeros((batch, dstate), dtype=torch.float32, device=u.device)
    xs = []

    for t in range(seqlen):
        x = A_bar[:, :, t] * x + Bu[:, :, t]
        xs.append(x)

    last_state = x
    x_seq = torch.stack(xs, dim=-1)
    y = torch.einsum("bhnl,bnl->bhl", C, x_seq)

    out = y.to(dtype_in)

    if return_last_state:
        return out, last_state
    return out


class S7InnerFn(torch.autograd.Function):
    """Fused S7 inner function with custom forward and backward."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        hidden_states,
        in_proj_weight,
        x_proj_weight,
        gate_proj_weight,
        d_state,
        base_params,
    ):
        batch, seqlen, d_model = hidden_states.shape

        if hidden_states.stride(-1) != 1:
            hidden_states = hidden_states.contiguous()

        # in_proj: (B, L, D) @ (D, D)^T -> (B, L, D)
        x = F.linear(hidden_states, in_proj_weight)

        # x_proj: (BL, D) @ (D, N + 2*D*N + D)^T -> (BL, N + 2*D*N + D)
        x_dbl = F.linear(rearrange(x, "b l d -> (b l) d"), x_proj_weight)

        # Split into A, B, C, D, bias
        A, B, C, D, bias = torch.split(
            x_dbl,
            [d_state, d_model * d_state, d_model * d_state, d_model, d_state],
            dim=-1,
        )

        A = rearrange(A, "(b l) n -> b n l", l=seqlen)
        A = A + base_params.unsqueeze(0).unsqueeze(
            -1
        )  # Add HiPPO initialization to A
        B = rearrange(B, "(b l) (h n) -> b n h l", l=seqlen, n=d_state)
        C = rearrange(
            C, "(b l) (h n) -> b h n l", l=seqlen, n=d_state
        ).contiguous()
        D_tv = rearrange(D, "(b l) h -> b h l", l=seqlen).contiguous()
        bias = rearrange(bias, "(b l) n -> b n l", l=seqlen)
        u = rearrange(x, "b l h -> b h l")

        Bu = torch.einsum("bnhl,bhl->bnl", B.float(), u.float())
        if bias is not None:
            Bu = Bu + bias.float()
        Bu = Bu.contiguous()

        # Run CUDA kernel with dim=N, dstate=1
        delta = A.contiguous()
        A_kernel = torch.zeros(d_state, 1, dtype=Bu.dtype, device=Bu.device)
        B_kernel = torch.ones(
            batch, 1, 1, seqlen, dtype=Bu.dtype, device=Bu.device
        ).contiguous()
        C_kernel = torch.ones(
            batch, 1, 1, seqlen, dtype=Bu.dtype, device=Bu.device
        ).contiguous()

        out_kernel, x_kernel, *_ = selective_scan_cuda.fwd(
            Bu,
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
            None,
            None,
            None,
            None,
            False,
            "s7",
        )

        # Project output: y = C @ out_kernel
        y = torch.einsum("bhnl,bnl->bhl", C.float(), out_kernel.float())

        # Apply D (time-varying skip)
        y = y + D_tv.float() * u.float()

        # Gating: gate_proj -> sigmoid, then gate * y
        y_t = rearrange(y, "b h l -> b l h")
        gelu_y_t = F.gelu(y_t)
        gate = torch.sigmoid(F.linear(gelu_y_t, gate_proj_weight))
        y_gated = gate * y_t

        # Residual
        out = y_gated + hidden_states

        ctx.save_for_backward(
            hidden_states,
            in_proj_weight,
            x_proj_weight,
            gate_proj_weight,
            x,
            x_dbl,
            A,
            B,
            C,
            D_tv,
            u,
            Bu,
            out_kernel,
            x_kernel,
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
            y,
            y_t,
            gate,
            gelu_y_t,
        )
        ctx.d_state = d_state

        return out.to(hidden_states.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        (
            hidden_states,
            in_proj_weight,
            x_proj_weight,
            gate_proj_weight,
            x,
            x_dbl,
            A,
            B,
            C,
            D_tv,
            u,
            Bu,
            out_kernel,
            x_kernel,
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
            y,
            y_t,
            gate,
            gelu_y_t,
        ) = ctx.saved_tensors
        d_state = ctx.d_state
        batch, seqlen, d_model = hidden_states.shape

        dout = dout.float()
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        dy_gated = dout
        dhidden_states = dout.clone()

        # Gradient through gating: y_gated = gate * y_t
        dgate = dy_gated * y_t
        dy_t = dy_gated * gate

        # Gradient through gate = sigmoid(gate_proj(gelu(y_t)))
        dsigmoid = gate * (1.0 - gate)
        dgate_pre = dgate * dsigmoid

        # Gradient through gate_proj
        dgate_proj_weight = dgate_pre.reshape(
            -1, d_model
        ).t() @ gelu_y_t.reshape(-1, d_model)
        dgelu_y_t = F.linear(dgate_pre, gate_proj_weight.t())

        # Gradient through GELU: gelu'(x) = 0.5 * (1 + tanh(k)) + 0.5 * x * sech^2(k) * (sqrt(2/pi) * (1 + 3*0.044715*x^2))
        # where k = sqrt(2/pi) * (x + 0.044715 * x^3)
        # Simplified: use the fact that gelu(x) ≈ x * sigmoid(1.702 * x) for approximation
        # But for exact gradient, compute directly
        k = 0.7978845608 * (
            y_t + 0.044715 * y_t * y_t * y_t
        )  # sqrt(2/pi) ≈ 0.7978845608
        tanh_k = torch.tanh(k)
        sech2_k = 1.0 - tanh_k * tanh_k
        dgelu = 0.5 * (1.0 + tanh_k) + 0.5 * y_t * sech2_k * 0.7978845608 * (
            1.0 + 3.0 * 0.044715 * y_t * y_t
        )
        dy_t_from_gate = dgelu_y_t * dgelu + dy_t

        dy = rearrange(dy_t_from_gate, "b l h -> b h l")

        # Gradient through D skip: y = y_inner + D_tv * u
        dD_tv = dy * u.float()
        dy_inner = dy
        du_from_D = D_tv.float() * dy

        # Gradient through C projection: y_inner = C @ out_kernel
        dC = torch.einsum("bhl,bnl->bhnl", dy_inner, out_kernel.float())
        dout_kernel = torch.einsum(
            "bhnl,bhl->bnl", C.float(), dy_inner
        ).contiguous()

        # Backward through CUDA kernel
        dBu, ddelta, *_ = selective_scan_cuda.bwd(
            Bu.contiguous(),
            delta,
            A_kernel,
            B_kernel,
            C_kernel,
            None,
            None,
            None,
            None,
            dout_kernel,
            x_kernel,
            out_kernel,
            None,
            False,
            False,
            "s7",
        )

        dA = ddelta
        dbase_params = torch.sum(
            dA, dim=(0, 2)
        )  # Sum over batch and time dimensions for base_params gradient
        dB = torch.einsum("bnl,bhl->bnhl", dBu, u.float())
        dbias = dBu
        du_from_B = torch.einsum("bnhl,bnl->bhl", B.float(), dBu)
        du = du_from_B + du_from_D

        dx = rearrange(du, "b h l -> b l h")

        # Reconstruct x_dbl gradients
        dA_flat = rearrange(dA, "b n l -> (b l) n")
        dB_flat = rearrange(dB, "b n h l -> (b l) (h n)")
        dC_flat = rearrange(dC, "b h n l -> (b l) (h n)")
        dD_flat = rearrange(dD_tv, "b h l -> (b l) h")
        dbias_flat = rearrange(dbias, "b n l -> (b l) n")
        dx_dbl = torch.cat(
            [dA_flat, dB_flat, dC_flat, dD_flat, dbias_flat], dim=-1
        )

        # Gradient through x_proj
        dx_proj_weight = dx_dbl.t() @ rearrange(x, "b l d -> (b l) d")
        dx_from_proj = F.linear(dx_dbl, x_proj_weight.t())
        dx_from_proj = rearrange(dx_from_proj, "(b l) d -> b l d", l=seqlen)
        dx = dx + dx_from_proj

        # Gradient through in_proj (no GELU here)
        din_proj_weight = dx.reshape(-1, d_model).t() @ hidden_states.reshape(
            -1, d_model
        )
        dhidden_states = dhidden_states + F.linear(dx, in_proj_weight.t())

        return (
            dhidden_states.to(hidden_states.dtype),
            din_proj_weight,
            dx_proj_weight,
            dgate_proj_weight,
            None,  # d_state
            dbase_params,
        )


def s7_inner_fn(
    hidden_states,
    in_proj_weight,
    x_proj_weight,
    gate_proj_weight,
    d_state,
    base_params,
):
    """Fused S7 inner function using CUDA kernel."""
    return S7InnerFn.apply(
        hidden_states,
        in_proj_weight,
        x_proj_weight,
        gate_proj_weight,
        d_state,
        base_params,
    )


def s7_inner_ref(
    hidden_states,
    in_proj_weight,
    x_proj_weight,
    gate_proj_weight,
    d_state,
    base_params,
):
    """Reference S7 inner function (pure PyTorch)."""
    batch, seqlen, d_model = hidden_states.shape

    x = F.linear(hidden_states, in_proj_weight)

    x_dbl = F.linear(rearrange(x, "b l d -> (b l) d"), x_proj_weight)
    A, B, C, D, bias = torch.split(
        x_dbl,
        [d_state, d_model * d_state, d_model * d_state, d_model, d_state],
        dim=-1,
    )

    A = rearrange(A, "(b l) n -> b n l", l=seqlen) + base_params.unsqueeze(
        0
    ).unsqueeze(
        -1
    )  # Add HiPPO initialization to A
    B = rearrange(B, "(b l) (h n) -> b n h l", l=seqlen, n=d_state)
    C = rearrange(C, "(b l) (h n) -> b h n l", l=seqlen, n=d_state)
    D_tv = rearrange(D, "(b l) h -> b h l", l=seqlen)
    bias = rearrange(bias, "(b l) n -> b n l", l=seqlen)

    u = rearrange(x, "b l h -> b h l")

    y = s7_scan_ref(u, A, B, C, bias=bias)
    y = y + D_tv * u

    y_t = rearrange(y, "b h l -> b l h")
    gate = torch.sigmoid(F.linear(F.gelu(y_t), gate_proj_weight))
    y_gated = gate * y_t

    out = y_gated + hidden_states
    return out
