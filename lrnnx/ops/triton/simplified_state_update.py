"""
Triton kernel for single-step state update of S5-style SSMs.
This is to simplified_scan what selective_state_update is to selective_scan.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_DELTAA": lambda args: args["deltaA_ptr"] is not None})
@triton.jit
def _simplified_state_update_kernel(
    # Pointers
    state_re_ptr,
    state_im_ptr,
    x_ptr,
    dt_ptr,
    A_re_ptr,
    A_im_ptr,
    B_re_ptr,
    B_im_ptr,
    C_re_ptr,
    C_im_ptr,
    D_ptr,
    deltaA_ptr,
    out_ptr,
    # Dimensions
    batch,
    H,  # input / output dim
    P,  # state dim
    # Strides - state (batch, P)
    stride_state_batch,
    stride_state_p,
    # Strides - x (batch, H)
    stride_x_batch,
    stride_x_h,
    # Strides - dt (batch, P)  or  (P,)
    stride_dt_batch,
    stride_dt_p,
    # Strides - A (P,)
    stride_A_p,
    # Strides - B (P, H)
    stride_B_p,
    stride_B_h,
    # Strides - C (H, P)
    stride_C_h,
    stride_C_p,
    # Strides - D (H, H) or None
    stride_D_h_out,
    stride_D_h_in,
    # Strides - deltaA (batch, P) or None
    stride_deltaA_batch,
    stride_deltaA_p,
    # Strides - out (batch, H)
    stride_out_batch,
    stride_out_h,
    # Meta
    CONJ_SYM: tl.constexpr,
    BLOCK_SIZE_P: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_DELTAA: tl.constexpr,
    DISCRETIZATION: tl.constexpr,  # 0=bilinear, 1=zoh, 2=dirac
):
    """
    Triton JIT kernel for the simplified state update. 
    Each program handles one batch element and a tile of H output dims.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offs_p = tl.arange(0, BLOCK_SIZE_P)  # state dim
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(
        0, BLOCK_SIZE_H
    )  # output dim tile
    mask_p = offs_p < P
    mask_h = offs_h < H

    state_re_ptrs = (
        state_re_ptr + pid_b * stride_state_batch + offs_p * stride_state_p
    )
    state_im_ptrs = (
        state_im_ptr + pid_b * stride_state_batch + offs_p * stride_state_p
    )
    s_re = tl.load(state_re_ptrs, mask=mask_p, other=0.0).to(tl.float32)
    s_im = tl.load(state_im_ptrs, mask=mask_p, other=0.0).to(tl.float32)

    x_ptrs = x_ptr + pid_b * stride_x_batch + offs_h * stride_x_h
    x_val = tl.load(x_ptrs, mask=mask_h, other=0.0).to(
        tl.float32
    )  # (BLOCK_SIZE_H,)

    A_re = tl.load(A_re_ptr + offs_p * stride_A_p, mask=mask_p, other=0.0).to(
        tl.float32
    )
    A_im = tl.load(A_im_ptr + offs_p * stride_A_p, mask=mask_p, other=0.0).to(
        tl.float32
    )

    dt = tl.load(
        dt_ptr + pid_b * stride_dt_batch + offs_p * stride_dt_p,
        mask=mask_p,
        other=0.0,
    ).to(tl.float32)

    if HAS_DELTAA:
        dtA = tl.load(
            deltaA_ptr
            + pid_b * stride_deltaA_batch
            + offs_p * stride_deltaA_p,
            mask=mask_p,
            other=0.0,
        ).to(tl.float32)
    else:
        dtA = dt

    # Discretize A: A_bar = discretize(A_complex, dtA)
    if DISCRETIZATION == 0:  # bilinear
        # A_bar = (1 + 0.5*dtA*A) / (1 - 0.5*dtA*A)
        half_dtA = 0.5 * dtA
        num_re = 1.0 + half_dtA * A_re
        num_im = half_dtA * A_im
        den_re = 1.0 - half_dtA * A_re
        den_im = -half_dtA * A_im
        # complex division: (num_re + j*num_im) / (den_re + j*den_im)
        den_mag_sq = den_re * den_re + den_im * den_im
        den_mag_sq = tl.where(den_mag_sq == 0.0, 1e-12, den_mag_sq)
        A_bar_re = (num_re * den_re + num_im * den_im) / den_mag_sq
        A_bar_im = (num_im * den_re - num_re * den_im) / den_mag_sq
    elif DISCRETIZATION == 1:  # zoh
        # A_bar = exp(dtA * A)   where A is complex
        # exp(a + jb) = exp(a) * (cos(b) + j*sin(b))
        exp_real = tl.exp(dtA * A_re)
        angle = dtA * A_im
        A_bar_re = exp_real * tl.cos(angle)
        A_bar_im = exp_real * tl.sin(angle)
    else:  # dirac  (DISCRETIZATION == 2)
        exp_real = tl.exp(dtA * A_re)
        angle = dtA * A_im
        A_bar_re = exp_real * tl.cos(angle)
        A_bar_im = exp_real * tl.sin(angle)

    # Discretize B: B_bar scalar per state dim
    if DISCRETIZATION == 0:  # bilinear
        # gamma_bar = dt / (1 - 0.5*dt*A)   (complex)
        half_dt = 0.5 * dt
        gden_re = 1.0 - half_dt * A_re
        gden_im = -half_dt * A_im
        gden_mag_sq = gden_re * gden_re + gden_im * gden_im
        gden_mag_sq = tl.where(gden_mag_sq == 0.0, 1e-12, gden_mag_sq)
        # dt is real, so numerator is (dt, 0)
        gamma_re = (dt * gden_re) / gden_mag_sq
        gamma_im = (-dt * gden_im) / gden_mag_sq  # 0*gden_re - dt*gden_im
    elif DISCRETIZATION == 1:  # zoh
        # gamma_bar = (exp(dt*A) - 1) / A   (complex)
        exp_re = tl.exp(dt * A_re)
        ang = dt * A_im
        expm1_re = exp_re * tl.cos(ang) - 1.0
        expm1_im = exp_re * tl.sin(ang)
        # divide by A (complex): (expm1) / (A_re + j*A_im)
        A_mag_sq = A_re * A_re + A_im * A_im
        A_mag_sq = tl.where(A_mag_sq == 0.0, 1e-12, A_mag_sq)
        gamma_re = (expm1_re * A_re + expm1_im * A_im) / A_mag_sq
        gamma_im = (expm1_im * A_re - expm1_re * A_im) / A_mag_sq
    else:  # dirac
        gamma_re = tl.full(offs_p.shape, 1.0, dtype=tl.float32)
        gamma_im = tl.full(offs_p.shape, 0.0, dtype=tl.float32)

    # Compute Bu = B @ x  (complex (P,) result)
    # We accumulate over H tiles
    Bu_re = tl.zeros((BLOCK_SIZE_P,), dtype=tl.float32)
    Bu_im = tl.zeros((BLOCK_SIZE_P,), dtype=tl.float32)

    for h_start in range(0, H, BLOCK_SIZE_H):
        h_offs = h_start + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offs < H
        x_tile = tl.load(
            x_ptr + pid_b * stride_x_batch + h_offs * stride_x_h,
            mask=h_mask,
            other=0.0,
        ).to(
            tl.float32
        )  # (BLOCK_SIZE_H,)
        # B_re/im: (P, H) -> load tile (BLOCK_SIZE_P, BLOCK_SIZE_H)
        B_re_tile = tl.load(
            B_re_ptr
            + offs_p[:, None] * stride_B_p
            + h_offs[None, :] * stride_B_h,
            mask=mask_p[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        B_im_tile = tl.load(
            B_im_ptr
            + offs_p[:, None] * stride_B_p
            + h_offs[None, :] * stride_B_h,
            mask=mask_p[:, None] & h_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        Bu_re += tl.sum(B_re_tile * x_tile[None, :], axis=1)
        Bu_im += tl.sum(B_im_tile * x_tile[None, :], axis=1)

    Bbaru_re = gamma_re * Bu_re - gamma_im * Bu_im
    Bbaru_im = gamma_re * Bu_im + gamma_im * Bu_re

    new_s_re = A_bar_re * s_re - A_bar_im * s_im + Bbaru_re
    new_s_im = A_bar_re * s_im + A_bar_im * s_re + Bbaru_im

    tl.store(state_re_ptrs, new_s_re, mask=mask_p)
    tl.store(state_im_ptrs, new_s_im, mask=mask_p)

    C_re_tile = tl.load(
        C_re_ptr + offs_h[:, None] * stride_C_h + offs_p[None, :] * stride_C_p,
        mask=mask_h[:, None] & mask_p[None, :],
        other=0.0,
    ).to(tl.float32)
    C_im_tile = tl.load(
        C_im_ptr + offs_h[:, None] * stride_C_h + offs_p[None, :] * stride_C_p,
        mask=mask_h[:, None] & mask_p[None, :],
        other=0.0,
    ).to(tl.float32)

    y_tile = tl.sum(
        C_re_tile * new_s_re[None, :] - C_im_tile * new_s_im[None, :], axis=1
    )

    if CONJ_SYM:
        y_tile = 2.0 * y_tile

    if HAS_D:
        Dx = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_SIZE_H):
            h_offs = h_start + tl.arange(0, BLOCK_SIZE_H)
            h_mask_inner = h_offs < H
            x_tile = tl.load(
                x_ptr + pid_b * stride_x_batch + h_offs * stride_x_h,
                mask=h_mask_inner,
                other=0.0,
            ).to(tl.float32)
            D_tile = tl.load(
                D_ptr
                + offs_h[:, None] * stride_D_h_out
                + h_offs[None, :] * stride_D_h_in,
                mask=mask_h[:, None] & h_mask_inner[None, :],
                other=0.0,
            ).to(tl.float32)
            Dx += tl.sum(D_tile * x_tile[None, :], axis=1)
        y_tile += Dx

    out_ptrs = out_ptr + pid_b * stride_out_batch + offs_h * stride_out_h
    tl.store(out_ptrs, y_tile, mask=mask_h)


def simplified_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    deltaA: torch.Tensor | None = None,
    discretization: str = "bilinear",
    conj_sym: bool = True,
) -> torch.Tensor:
    """
    Triton-accelerated single-step state update for S5-style (simplified) SSMs.

    Args:
        state (torch.Tensor): Complex hidden state of shape ``(batch, P)``, dtype ``complex64``. **Modified in-place**.
        x (torch.Tensor): Real input at current timestep of shape ``(batch, H)``, dtype ``float32``.
        dt (torch.Tensor): Real timestep of shape ``(batch, P)`` or ``(P,)``, dtype ``float32``.
        A (torch.Tensor): Complex eigenvalues of shape ``(P,)``, dtype ``complex64``.
        B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype ``complex64``.
        C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype ``complex64``.
        D (torch.Tensor, optional): Real skip connection matrix of shape ``(H, H)``, dtype ``float32``. Defaults to None.
        deltaA (torch.Tensor, optional): Optional separate timestep for A discretization of shape ``(batch, P)``, dtype ``float32``. If None, ``dt`` is used for both A and B. Defaults to None.
        discretization (str, optional): Discretization method ('bilinear', 'zoh', or 'dirac'). Defaults to "bilinear".
        conj_sym (bool, optional): If True, output is 2 * Re(...), else Re(...). Defaults to True.

    Returns:
        torch.Tensor: Real output tensor of shape ``(batch, H)``, dtype ``float32``.
    """
    assert state.is_complex(), "state must be complex64"
    assert A.is_complex(), "A must be complex64"
    assert B.is_complex(), "B must be complex64"
    assert C.is_complex(), "C must be complex64"
    assert not x.is_complex(), "x must be real (float32)"

    batch, P = state.shape
    H = x.shape[1]
    assert x.shape == (batch, H)
    assert A.shape == (P,)
    assert B.shape == (P, H)
    assert C.shape == (H, P)
    if D is not None:
        assert D.shape == (H, H)
    if dt.dim() == 1:
        assert dt.shape == (P,)
        dt = dt.unsqueeze(0).expand(batch, -1).contiguous()
    assert dt.shape == (batch, P)
    if deltaA is not None:
        if deltaA.dim() == 1:
            deltaA = deltaA.unsqueeze(0).expand(batch, -1).contiguous()
        assert deltaA.shape == (batch, P)

    disc_map = {"bilinear": 0, "zoh": 1, "dirac": 2}
    disc_int = disc_map.get(discretization.lower())
    if disc_int is None:
        raise ValueError(
            f"discretization must be one of {list(disc_map)}, got '{discretization}'"
        )

    # View complex tensors as their underlying float storage (real, imag contiguous pairs)
    # complex64 -> view_as_real -> (..., 2) float32
    state_ri = torch.view_as_real(state)  # (batch, P, 2)
    A_ri = torch.view_as_real(A)  # (P, 2)
    B_ri = torch.view_as_real(B)  # (P, H, 2)
    C_ri = torch.view_as_real(C)  # (H, P, 2)

    state_re = state_ri[..., 0].contiguous()  # (batch, P)
    state_im = state_ri[..., 1].contiguous()  # (batch, P)
    A_re = A_ri[..., 0].contiguous()  # (P,)
    A_im = A_ri[..., 1].contiguous()  # (P,)
    B_re = B_ri[..., 0].contiguous()  # (P, H)
    B_im = B_ri[..., 1].contiguous()  # (P, H)
    C_re = C_ri[..., 0].contiguous()  # (H, P)
    C_im = C_ri[..., 1].contiguous()  # (H, P)

    x = x.contiguous()
    dt = dt.contiguous()

    out = torch.empty(batch, H, device=x.device, dtype=torch.float32)

    # Determine block sizes
    BLOCK_SIZE_P = triton.next_power_of_2(P)
    BLOCK_SIZE_H = min(triton.next_power_of_2(H), 128)

    grid = (batch, triton.cdiv(H, BLOCK_SIZE_H))

    with torch.cuda.device(x.device.index):
        _simplified_state_update_kernel[grid](
            state_re,
            state_im,
            x,
            dt,
            A_re,
            A_im,
            B_re,
            B_im,
            C_re,
            C_im,
            D,
            deltaA,
            out,
            # dims
            batch,
            H,
            P,
            # state strides
            state_re.stride(0),
            state_re.stride(1),
            # x strides
            x.stride(0),
            x.stride(1),
            # dt strides
            dt.stride(0),
            dt.stride(1),
            # A strides
            A_re.stride(0),
            # B strides
            B_re.stride(0),
            B_re.stride(1),
            # C strides
            C_re.stride(0),
            C_re.stride(1),
            # D strides
            *(D.stride(0), D.stride(1)) if D is not None else (0, 0),
            # deltaA strides
            *(
                (deltaA.stride(0), deltaA.stride(1))
                if deltaA is not None
                else (0, 0)
            ),
            # out strides
            out.stride(0),
            out.stride(1),
            # meta
            CONJ_SYM=conj_sym,
            BLOCK_SIZE_P=BLOCK_SIZE_P,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            DISCRETIZATION=disc_int,
        )

    # Write updated state back into the original complex tensor
    state.real.copy_(state_re)
    state.imag.copy_(state_im)

    return out


def simplified_state_update_ref(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    deltaA: torch.Tensor | None = None,
    discretization: str = "bilinear",
    conj_sym: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch reference for a single-step S5 state update.

    Args:
        state (torch.Tensor): Complex hidden state of shape ``(batch, P)``, dtype ``complex64``. **Modified in-place**.
        x (torch.Tensor): Real input at current timestep of shape ``(batch, H)``, dtype ``float32``.
        dt (torch.Tensor): Real timestep of shape ``(batch, P)`` or ``(P,)``, dtype ``float32``.
        A (torch.Tensor): Complex eigenvalues of shape ``(P,)``, dtype ``complex64``.
        B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype ``complex64``.
        C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype ``complex64``.
        D (torch.Tensor, optional): Real skip connection matrix of shape ``(H, H)``, dtype ``float32``. Defaults to None.
        deltaA (torch.Tensor, optional): Optional separate timestep for A discretization of shape ``(batch, P)``, dtype ``float32``. Defaults to None.
        discretization (str, optional): Discretization method ('bilinear', 'zoh', or 'dirac'). Defaults to "bilinear".
        conj_sym (bool, optional): If True, output is 2 * Re(...), else Re(...). Defaults to True.

    Returns:
        torch.Tensor: Real output tensor of shape ``(batch, H)``, dtype ``float32``.
    """
    assert state.is_complex()
    assert A.is_complex()
    assert B.is_complex()
    assert C.is_complex()

    batch, P = state.shape
    H = x.shape[1]

    # Expand dt / deltaA to (batch, P) if needed
    if dt.dim() == 1:
        dt = dt.unsqueeze(0).expand(batch, -1)
    dtA = deltaA if deltaA is not None else dt
    if dtA.dim() == 1:
        dtA = dtA.unsqueeze(0).expand(batch, -1)

    A_complex = A  # (P,)

    if discretization.lower() == "bilinear":
        half_dtA_A = 0.5 * dtA * A_complex  # (batch, P)
        A_bar = (1.0 + half_dtA_A) / (1.0 - half_dtA_A)
    elif discretization.lower() == "zoh":
        A_bar = torch.exp(dtA * A_complex)
    elif discretization.lower() == "dirac":
        A_bar = torch.exp(dtA * A_complex)
    else:
        raise ValueError(f"Unknown discretization: {discretization}")

    if discretization.lower() == "bilinear":
        gamma_bar = dt / (1.0 - 0.5 * dt * A_complex)  # (batch, P) complex
    elif discretization.lower() == "zoh":
        gamma_bar = (torch.exp(dt * A_complex) - 1.0) / A_complex
    elif discretization.lower() == "dirac":
        gamma_bar = torch.ones_like(dt, dtype=A.dtype)
    else:
        raise ValueError(f"Unknown discretization: {discretization}")

    Bu = torch.einsum("ph,bh->bp", B, x.to(B.dtype))

    state.copy_(A_bar * state + gamma_bar * Bu)

    y_complex = torch.einsum("hp,bp->bh", C, state)
    y = y_complex.real
    if conj_sym:
        y = 2.0 * y

    if D is not None:
        y = y + x @ D.T  # (batch, H) @ (H, H)^T -> (batch, H)

    return y