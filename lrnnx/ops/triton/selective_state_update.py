# Copyright (c) 2024, Tri Dao, Albert Gu.
# Modified by SAiDL.

"""We want triton==2.1.0 or triton==2.2.0 or triton==2.3.0 for this"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange, repeat

from lrnnx.ops.triton.softplus import softplus


@triton.heuristics(
    {"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None}
)
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics({"HAS_DELTAA": lambda args: args["deltaA_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    deltaA_ptr,
    out_ptr,
    state_batch_indices_ptr,
    # Matrix dimensions
    batch,
    nheads,
    dim,
    dstate,
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_deltaA_batch,
    stride_deltaA_head,
    stride_deltaA_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DELTAA: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    DISCRETIZATION: tl.constexpr,
):
    """
    Triton JIT kernel for the selective state update.
    """
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    if HAS_STATE_BATCH_INDICES:
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr)
        state_ptr += (
            state_batch_idx * stride_state_batch + pid_h * stride_state_head
        )
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += (
        pid_b * stride_B_batch
        + (pid_h // nheads_ngroups_ratio) * stride_B_group
    )
    C_ptr += (
        pid_b * stride_C_batch
        + (pid_h // nheads_ngroups_ratio) * stride_C_group
    )
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    if HAS_DELTAA:
        deltaA_ptr += pid_b * stride_deltaA_batch + pid_h * stride_deltaA_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim
        + offs_n[None, :] * stride_state_dstate
    )
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (
        offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
    )
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    if HAS_DELTAA:
        deltaA_ptrs = deltaA_ptr + offs_m * stride_deltaA_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    state = tl.load(
        state_ptrs,
        mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
        other=0.0,
    )
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if not TIE_HDIM:
        A = tl.load(
            A_ptrs,
            mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
            other=-1.0,
        ).to(tl.float32)
    else:
        A = tl.load(A_ptr).to(tl.float32)

    if HAS_DELTAA:
        if not TIE_HDIM:
            deltaA = tl.load(deltaA_ptrs, mask=offs_m < dim, other=0.0).to(
                tl.float32
            )
            if DISCRETIZATION == 4:
                dA = tl.exp(deltaA[:, None] * tl.log(A))
            else:
                dA = tl.exp(A * deltaA[:, None])
        else:
            deltaA = tl.load(deltaA_ptr).to(tl.float32)
            if DISCRETIZATION == 4:
                dA = tl.exp(deltaA * tl.log(A))
            else:
                dA = tl.exp(A * deltaA)
    else:
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(
                    tl.float32
                )
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, softplus(dt), dt)
            if DISCRETIZATION == 4:
                dA = tl.exp(dt[:, None] * tl.log(A))
            elif DISCRETIZATION == 5:  # S7: A_bar = 1 - 1/(A_raw² + 0.5)
                dt_sq_half = dt[:, None] * dt[:, None] + 0.5
                dA = 1.0 - 1.0 / dt_sq_half
            else:
                dA = tl.exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, softplus(dt), dt)
            if DISCRETIZATION == 4:
                dA = tl.exp(dt * tl.log(A))
            elif DISCRETIZATION == 5:  # S7
                dt_sq_half = dt * dt + 0.5
                dA = 1.0 - 1.0 / dt_sq_half
            else:
                dA = tl.exp(A * dt)

    if not TIE_HDIM:
        if not HAS_DELTAA:
            pass
        else:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(
                    tl.float32
                )
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, softplus(dt), dt)
    else:
        if not HAS_DELTAA:
            pass
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = tl.where(dt <= 20.0, softplus(dt), dt)

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    if DISCRETIZATION == 0:
        if not TIE_HDIM:
            A_dt = A * dt[:, None]
            expm1_A_dt = tl.exp(A_dt) - 1.0
            B_tilde = expm1_A_dt / A
            dB = B[None, :] * B_tilde
        else:
            A_dt = A * dt
            expm1_A_dt = tl.exp(A_dt) - 1.0
            B_tilde = expm1_A_dt / A
            dB = B * B_tilde
    elif DISCRETIZATION == 1:
        if not TIE_HDIM:
            v = 0.5 * dt[:, None] * A
            den_inv = 1.0 / (1.0 - v)
            dB = B[None, :] * den_inv * dt[:, None]
        else:
            v = 0.5 * dt * A
            den_inv = 1.0 / (1.0 - v)
            dB = B * den_inv * dt
    elif DISCRETIZATION == 2:
        if not TIE_HDIM:
            dB = B[None, :]  # (1, dstate), will broadcast with x[:, None]
        else:
            dB = B
    elif DISCRETIZATION == 4:
        sqrt_term = tl.sqrt(1.0 - dA * dA)
        if not TIE_HDIM:
            dB = B[None, :] * sqrt_term
        else:
            dB = B * sqrt_term
    elif DISCRETIZATION == 5:  # S7: identity (Bu pre-computed)
        if not TIE_HDIM:
            dB = B[None, :]
        else:
            dB = B
    else:
        if not TIE_HDIM:
            dB = B[None, :] * dt[:, None]
        else:
            dB = B * dt

    state = state * dA + dB * x[:, None]
    tl.store(
        state_ptrs,
        state,
        mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
    )
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    deltaA=None,
    state_batch_indices=None,
    discretization="mamba",
):
    """
    Triton-accelerated single-step state update for selective state space models.

    Args:
        state (torch.Tensor): Hidden state of shape ``(batch, dim, dstate)`` or ``(batch, nheads, dim, dstate)``.
        x (torch.Tensor): Input tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``.
        dt (torch.Tensor): Timestep tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``.
        A (torch.Tensor): State transition matrix of shape ``(dim, dstate)`` or ``(nheads, dim, dstate)``.
        B (torch.Tensor): Input projection matrix of shape ``(batch, dstate)`` or ``(batch, ngroups, dstate)``.
        C (torch.Tensor): Output projection matrix of shape ``(batch, dstate)`` or ``(batch, ngroups, dstate)``.
        D (torch.Tensor, optional): Skip connection vector of shape ``(dim,)`` or ``(nheads, dim)``. Defaults to None.
        z (torch.Tensor, optional): Gating tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``. Defaults to None.
        dt_bias (torch.Tensor, optional): Bias for dt of shape ``(dim,)`` or ``(nheads, dim)``. Defaults to None.
        dt_softplus (bool, optional): Whether to apply softplus to dt. Defaults to False.
        deltaA (torch.Tensor, optional): Timestep for A discretization (dtA) in asymmetric mode, shape ``(batch, dim)`` or ``(batch, nheads, dim)``. Defaults to None.
        state_batch_indices (torch.Tensor, optional): Indices to select states for the batch, shape ``(batch,)``. Defaults to None.
        discretization (str, optional): Discretization method ('zoh', 'bilinear', 'dirac', 'mamba', 'rglru', 's7'). Defaults to "mamba".

    Returns:
        torch.Tensor: The output tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``.
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if deltaA is not None and deltaA.dim() == 2:
        deltaA = deltaA.unsqueeze(1)
    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    if x.shape != (batch, nheads, dim):
        print(f"{state.shape} {x.shape} {batch} {nheads} {dim}")
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if deltaA is not None:
        assert deltaA.shape == (batch, nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)

    # map discretization string to integer
    disc_map = {
        "zoh": 0,
        "bilinear": 1,
        "dirac": 2,
        "mamba": 3,
        "rglru": 4,
        "s7": 5,
    }
    disc_int = disc_map.get(discretization.lower(), 3)

    out = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (
        (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    )
    deltaA_strides = (
        (deltaA.stride(0), deltaA.stride(1), deltaA.stride(2))
        if deltaA is not None
        else (0, 0, 0)
    )
    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    BLOCK_SIZE_M, num_warps = (
        (32, 4)
        if dstate <= 16
        else (
            (16, 4)
            if dstate <= 32
            else (
                (8, 4)
                if dstate <= 64
                else ((4, 4) if dstate <= 128 else ((4, 8)))
            )
        )
    )
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is not None and dt_bias.stride(-1) == 0)
    )
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            deltaA,
            out,
            state_batch_indices,
            batch,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            *(
                (dt_bias.stride(0), dt_bias.stride(1))
                if dt_bias is not None
                else (0, 0)
            ),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            *((D.stride(0), D.stride(1)) if D is not None else (0, 0)),
            z_strides[0],
            z_strides[1],
            z_strides[2],
            deltaA_strides[0],
            deltaA_strides[1],
            deltaA_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            DISCRETIZATION=disc_int,
            num_warps=num_warps,
        )
    if not has_heads:
        out = out.squeeze(1)
    return out


def selective_state_update_ref(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    deltaA=None,
    discretization="mamba",
):
    """
    Reference (pure PyTorch) implementation of the single-step selective state update.

    Args:
        state (torch.Tensor): Hidden state of shape ``(batch, dim, dstate)`` or ``(batch, nheads, dim, dstate)``.
        x (torch.Tensor): Input tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``.
        dt (torch.Tensor): Timestep tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``.
        A (torch.Tensor): State transition matrix of shape ``(dim, dstate)`` or ``(nheads, dim, dstate)``.
        B (torch.Tensor): Input projection matrix of shape ``(batch, dstate)`` or ``(batch, ngroups, dstate)``.
        C (torch.Tensor): Output projection matrix of shape ``(batch, dstate)`` or ``(batch, ngroups, dstate)``.
        D (torch.Tensor, optional): Skip connection vector of shape ``(dim,)`` or ``(nheads, dim)``. Defaults to None.
        z (torch.Tensor, optional): Gating tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``. Defaults to None.
        dt_bias (torch.Tensor, optional): Bias for dt of shape ``(dim,)`` or ``(nheads, dim)``. Defaults to None.
        dt_softplus (bool, optional): Whether to apply softplus to dt. Defaults to False.
        deltaA (torch.Tensor, optional): Timestep for A discretization (dtA) in asymmetric mode, shape ``(batch, dim)`` or ``(batch, nheads, dim)``. Defaults to None.
        discretization (str, optional): Discretization method ('zoh', 'bilinear', 'dirac', 'mamba', 'rglru', 's7'). Defaults to "mamba".

    Returns:
        torch.Tensor: The output tensor of shape ``(batch, dim)`` or ``(batch, nheads, dim)``.
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if deltaA is not None and deltaA.dim() == 2:
        deltaA = deltaA.unsqueeze(1)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if deltaA is not None:
        assert deltaA.shape == (batch, nheads, dim)

    dt = dt + dt_bias if dt_bias is not None else dt
    dt = F.softplus(dt) if dt_softplus else dt

    if discretization.lower() == "rglru":
        gate = rearrange(
            deltaA if deltaA is not None else dt, "b h d -> b h d 1"
        )
        dA = A**gate  # (batch, nheads, dim, dstate)
    elif discretization.lower() == "s7":
        dt_exp = rearrange(dt, "b h d -> b h d 1")
        dA = 1.0 - 1.0 / (dt_exp * dt_exp + 0.5)
    elif deltaA is not None:
        dA = torch.exp(
            rearrange(deltaA, "b h d -> b h d 1") * A
        )  # (batch, nheads, dim, dstate)
    else:
        dA = torch.exp(
            rearrange(dt, "b h d -> b h d 1") * A
        )  # (batch, nheads, dim, dstate)
    B = repeat(
        B, "b g n -> b (g h) n", h=nheads // ngroups
    )  # (batch, nheads, dstate)
    C = repeat(
        C, "b g n -> b (g h) n", h=nheads // ngroups
    )  # (batch, nheads, dstate)

    dt_expanded = rearrange(dt, "b h d -> b h d 1")
    B_expanded = rearrange(B, "b h n -> b h 1 n")

    if discretization.lower() == "zoh":
        A_dt = dt_expanded * A  # (batch, nheads, dim, dstate)
        expm1_A_dt = torch.exp(A_dt) - 1.0
        B_tilde = expm1_A_dt / A  # (batch, nheads, dim, dstate)
        dB = B_expanded * B_tilde
    elif discretization.lower() == "bilinear":
        v = 0.5 * dt_expanded * A
        den_inv = 1.0 / (1.0 - v)
        dB = B_expanded * den_inv * dt_expanded
    elif discretization.lower() in ("dirac", "s7"):
        dB = B_expanded.expand_as(dA)
    elif discretization.lower() == "rglru":
        sqrt_term = torch.sqrt(1.0 - dA * dA)
        dB = B_expanded * sqrt_term
    else:
        dB = dt_expanded * B_expanded

    state.copy_(
        state * dA + dB * rearrange(x, "b h d -> b h d 1")
    )  # (batch, dim, dstate)
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out