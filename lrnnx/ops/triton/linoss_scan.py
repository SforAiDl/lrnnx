"""
Triton 2x2 time-invariant associative scan for LinOSS.

The recurrence is ``x_t = A x_{t-1} + F_t`` with a single real 2x2 ``A``
shared across time for every state channel. A sequence is scanned in chunks
of at most ``MAX_BLOCK_L`` timesteps with a carry threaded between them,
which reproduces ``jax.lax.associative_scan`` over the whole sequence for
any length.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One program owns a whole (batch, state) sequence, so the grid already has
# batch * d_state programs to fill the device. Benchmarks across shapes then
# favour a single warp per program with a narrow chunk: it maximises the work
# each thread does between the scan's cross-lane shuffles.
MAX_BLOCK_L = 256
MIN_BLOCK_L = 32
NUM_WARPS = 1

# Low precision inputs are accumulated in fp32; fp64 keeps its own precision.
_ACC_TYPES = {
    torch.float16: tl.float32,
    torch.bfloat16: tl.float32,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}
_ACC_TORCH_TYPES = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


@triton.jit
def _combine_2x2(
    a_m11,
    a_m12,
    a_m21,
    a_m22,
    a_f1,
    a_f2,
    b_m11,
    b_m12,
    b_m21,
    b_m22,
    b_f1,
    b_f2,
):
    """
    Compose the affine maps ``x -> M x + f`` of two scan elements.

    ``tl.associative_scan`` passes the element that comes first in scan order
    as ``a`` and the one that follows it as ``b`` - also with
    ``reverse=True``, where scan order runs backwards - so the composition is
    ``M_b (M_a x + f_a) + f_b``.
    """
    new_m11 = b_m11 * a_m11 + b_m12 * a_m21
    new_m12 = b_m11 * a_m12 + b_m12 * a_m22
    new_m21 = b_m21 * a_m11 + b_m22 * a_m21
    new_m22 = b_m21 * a_m12 + b_m22 * a_m22

    new_f1 = b_m11 * a_f1 + b_m12 * a_f2 + b_f1
    new_f2 = b_m21 * a_f1 + b_m22 * a_f2 + b_f2

    return new_m11, new_m12, new_m21, new_m22, new_f1, new_f2


@triton.jit
def _linoss_scan_fwd_kernel(
    M_ptr,
    F_ptr,
    out_ptr,
    seqlen,
    stride_m_p,
    stride_f_b,
    stride_f_p,
    stride_f_l,
    stride_o_b,
    stride_o_p,
    stride_o_l,
    ACC_TYPE: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """One program scans the full sequence of one ``(batch, state)`` pair."""
    pid_b = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)

    m_base = M_ptr + pid_p * stride_m_p
    m11 = tl.load(m_base + 0).to(ACC_TYPE)
    m12 = tl.load(m_base + 1).to(ACC_TYPE)
    m21 = tl.load(m_base + 2).to(ACC_TYPE)
    m22 = tl.load(m_base + 3).to(ACC_TYPE)

    f_base = F_ptr + pid_b * stride_f_b + pid_p * stride_f_p
    o_base = out_ptr + pid_b * stride_o_b + pid_p * stride_o_p

    offs = tl.arange(0, BLOCK_L)
    is_first = offs == 0

    carry_f1 = tl.full((), 0.0, dtype=ACC_TYPE)
    carry_f2 = tl.full((), 0.0, dtype=ACC_TYPE)

    for start in range(0, seqlen, BLOCK_L):
        offs_l = start + offs
        mask = offs_l < seqlen

        f_ptrs = f_base + offs_l * stride_f_l
        f1 = tl.load(f_ptrs + 0, mask=mask, other=0.0).to(ACC_TYPE)
        f2 = tl.load(f_ptrs + 1, mask=mask, other=0.0).to(ACC_TYPE)

        # Fold the previous chunk's last state into this chunk's first one.
        f1 = tl.where(is_first, m11 * carry_f1 + m12 * carry_f2 + f1, f1)
        f2 = tl.where(is_first, m21 * carry_f1 + m22 * carry_f2 + f2, f2)

        # Out-of-range lanes carry the monoid identity (I, 0) so that they
        # cannot perturb the valid prefix of the scan.
        a11 = tl.where(mask, m11, 1.0)
        a12 = tl.where(mask, m12, 0.0)
        a21 = tl.where(mask, m21, 0.0)
        a22 = tl.where(mask, m22, 1.0)

        _, _, _, _, s_f1, s_f2 = tl.associative_scan(
            (a11, a12, a21, a22, f1, f2), 0, _combine_2x2
        )

        o_ptrs = o_base + offs_l * stride_o_l
        tl.store(o_ptrs + 0, s_f1, mask=mask)
        tl.store(o_ptrs + 1, s_f2, mask=mask)

        is_last = offs == tl.minimum(BLOCK_L, seqlen - start) - 1
        carry_f1 = tl.sum(tl.where(is_last, s_f1, 0.0))
        carry_f2 = tl.sum(tl.where(is_last, s_f2, 0.0))


@triton.jit
def _linoss_scan_bwd_kernel(
    M_ptr,
    xs_ptr,
    dout_ptr,
    dF_ptr,
    dM_ptr,
    seqlen,
    stride_m_p,
    stride_x_b,
    stride_x_p,
    stride_x_l,
    stride_do_b,
    stride_do_p,
    stride_do_l,
    stride_df_b,
    stride_df_p,
    stride_df_l,
    stride_dm_p,
    ACC_TYPE: tl.constexpr,
    HAS_DM: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Reverse scan with ``A^T``.

    ``g_t = A^T g_{t+1} + dout_t`` gives ``dF_t = g_t`` and
    ``dA = sum_t g_t x_{t-1}^T`` with ``x_{-1} = 0``. ``A`` is shared across
    the batch, so ``dA`` is reduced with atomics.
    """
    pid_b = tl.program_id(axis=0)
    pid_p = tl.program_id(axis=1)

    # A^T, again packed row-major.
    m_base = M_ptr + pid_p * stride_m_p
    t11 = tl.load(m_base + 0).to(ACC_TYPE)
    t12 = tl.load(m_base + 2).to(ACC_TYPE)
    t21 = tl.load(m_base + 1).to(ACC_TYPE)
    t22 = tl.load(m_base + 3).to(ACC_TYPE)

    do_base = dout_ptr + pid_b * stride_do_b + pid_p * stride_do_p
    df_base = dF_ptr + pid_b * stride_df_b + pid_p * stride_df_p
    xs_base = xs_ptr + pid_b * stride_x_b + pid_p * stride_x_p

    offs = tl.arange(0, BLOCK_L)
    is_first = offs == 0

    carry_g1 = tl.full((), 0.0, dtype=ACC_TYPE)
    carry_g2 = tl.full((), 0.0, dtype=ACC_TYPE)
    dm11 = tl.full((), 0.0, dtype=ACC_TYPE)
    dm12 = tl.full((), 0.0, dtype=ACC_TYPE)
    dm21 = tl.full((), 0.0, dtype=ACC_TYPE)
    dm22 = tl.full((), 0.0, dtype=ACC_TYPE)

    num_chunks = tl.cdiv(seqlen, BLOCK_L)
    for rev_chunk in range(num_chunks):
        start = (num_chunks - 1 - rev_chunk) * BLOCK_L
        offs_l = start + offs
        mask = offs_l < seqlen

        do_ptrs = do_base + offs_l * stride_do_l
        d1 = tl.load(do_ptrs + 0, mask=mask, other=0.0).to(ACC_TYPE)
        d2 = tl.load(do_ptrs + 1, mask=mask, other=0.0).to(ACC_TYPE)

        # Fold the following chunk's leading gradient into this chunk's last
        # element (the first one the reverse scan visits).
        is_last = offs == tl.minimum(BLOCK_L, seqlen - start) - 1
        d1 = tl.where(is_last, t11 * carry_g1 + t12 * carry_g2 + d1, d1)
        d2 = tl.where(is_last, t21 * carry_g1 + t22 * carry_g2 + d2, d2)

        a11 = tl.where(mask, t11, 1.0)
        a12 = tl.where(mask, t12, 0.0)
        a21 = tl.where(mask, t21, 0.0)
        a22 = tl.where(mask, t22, 1.0)

        _, _, _, _, g1, g2 = tl.associative_scan(
            (a11, a12, a21, a22, d1, d2), 0, _combine_2x2, reverse=True
        )

        df_ptrs = df_base + offs_l * stride_df_l
        tl.store(df_ptrs + 0, g1, mask=mask)
        tl.store(df_ptrs + 1, g2, mask=mask)

        carry_g1 = tl.sum(tl.where(is_first, g1, 0.0))
        carry_g2 = tl.sum(tl.where(is_first, g2, 0.0))

        if HAS_DM:
            prev_ptrs = xs_base + (offs_l - 1) * stride_x_l
            prev_mask = mask & (offs_l > 0)
            x1 = tl.load(prev_ptrs + 0, mask=prev_mask, other=0.0).to(ACC_TYPE)
            x2 = tl.load(prev_ptrs + 1, mask=prev_mask, other=0.0).to(ACC_TYPE)
            dm11 += tl.sum(g1 * x1)
            dm12 += tl.sum(g1 * x2)
            dm21 += tl.sum(g2 * x1)
            dm22 += tl.sum(g2 * x2)

    if HAS_DM:
        dm_base = dM_ptr + pid_p * stride_dm_p
        tl.atomic_add(dm_base + 0, dm11)
        tl.atomic_add(dm_base + 1, dm12)
        tl.atomic_add(dm_base + 2, dm21)
        tl.atomic_add(dm_base + 3, dm22)


def _check_inputs(
    M: torch.Tensor, F: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate shapes/dtypes, returning contiguous ``M`` and ``F``."""
    if M.dim() != 2 or M.shape[-1] != 4:
        raise ValueError(f"M must have shape (P, 4), got {tuple(M.shape)}")
    if F.dim() != 4 or F.shape[-1] != 2:
        raise ValueError(
            f"F must have shape (B, P, L, 2), got {tuple(F.shape)}"
        )
    if F.shape[1] != M.shape[0]:
        raise ValueError(
            f"M and F state dims must match, got P={M.shape[0]} vs {F.shape[1]}"
        )
    if M.is_complex() or F.is_complex():
        raise ValueError(
            "The Triton scan is real-valued; split complex forcing into its "
            "real and imaginary parts before calling it."
        )
    if M.dtype not in _ACC_TYPES or F.dtype not in _ACC_TYPES:
        raise ValueError(
            f"Unsupported dtypes M={M.dtype}, F={F.dtype}; expected one of "
            f"{list(_ACC_TYPES)}"
        )
    return M.contiguous(), F.contiguous()


def _launch_config(
    M: torch.Tensor, F: torch.Tensor, seqlen: int
) -> tuple[object, torch.dtype, int]:
    """Pick the accumulator type and chunk width for a launch."""
    acc = torch.promote_types(M.dtype, F.dtype)
    block_l = max(
        MIN_BLOCK_L, min(MAX_BLOCK_L, triton.next_power_of_2(seqlen))
    )
    return _ACC_TYPES[acc], _ACC_TORCH_TYPES[acc], block_l


def linoss_scan_triton_fwd(M: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Run the forward 2x2 scan on the GPU.

    Args:
        M (torch.Tensor): ``(P, 4)`` real transitions, 2x2 packed row-major.
        F (torch.Tensor): ``(B, P, L, 2)`` real forcing vectors.

    Returns:
        torch.Tensor: Scanned states ``xs`` of shape ``(B, P, L, 2)``.
    """
    M, F = _check_inputs(M, F)
    batch, P, seqlen, _ = F.shape
    out_F = torch.empty_like(F)
    if seqlen == 0:
        return out_F

    acc_type, _, block_l = _launch_config(M, F, seqlen)
    _linoss_scan_fwd_kernel[(batch, P)](
        M,
        F,
        out_F,
        seqlen,
        M.stride(0),
        F.stride(0),
        F.stride(1),
        F.stride(2),
        out_F.stride(0),
        out_F.stride(1),
        out_F.stride(2),
        ACC_TYPE=acc_type,
        BLOCK_L=block_l,
        num_warps=NUM_WARPS,
    )
    return out_F


def linoss_scan_triton_bwd(
    M: torch.Tensor,
    xs: torch.Tensor,
    d_out: torch.Tensor,
    needs_dM: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the reverse 2x2 scan that produces the scan's gradients.

    Args:
        M (torch.Tensor): ``(P, 4)`` real transitions.
        xs (torch.Tensor): Forward scanned states, ``(B, P, L, 2)``.
        d_out (torch.Tensor): Upstream gradient w.r.t. ``xs``, same shape.
        needs_dM (bool, optional): Whether ``dM`` is required. When False the
            kernel skips the outer-product reduction. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``dM`` of shape ``(P, 4)`` and
            ``dF`` of shape ``(B, P, L, 2)``.
    """
    M, xs = _check_inputs(M, xs)
    d_out = d_out.contiguous()
    batch, P, seqlen, _ = xs.shape
    dF = torch.empty_like(xs)

    acc_type, acc_torch_type, block_l = _launch_config(M, xs, seqlen)
    dM = torch.zeros(P, 4, device=M.device, dtype=acc_torch_type)
    if seqlen == 0:
        return dM.to(M.dtype), dF

    _linoss_scan_bwd_kernel[(batch, P)](
        M,
        xs,
        d_out,
        dF,
        dM,
        seqlen,
        M.stride(0),
        xs.stride(0),
        xs.stride(1),
        xs.stride(2),
        d_out.stride(0),
        d_out.stride(1),
        d_out.stride(2),
        dF.stride(0),
        dF.stride(1),
        dF.stride(2),
        dM.stride(0),
        ACC_TYPE=acc_type,
        HAS_DM=needs_dM,
        BLOCK_L=block_l,
        num_warps=NUM_WARPS,
    )
    return dM.to(M.dtype), dF
