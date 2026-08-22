"""
LinOSS 2x2 time-invariant scan.

Exposes a Triton CUDA kernel with a PyTorch Kogge-Stone reference that is
used on CPU and whenever Triton is unavailable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from lrnnx.ops.torch import custom_bwd, custom_fwd

try:
    from lrnnx.ops.triton.linoss_scan import (
        linoss_scan_triton_bwd,
        linoss_scan_triton_fwd,
    )
except ImportError:  # pragma: no cover - triton is an optional dependency
    linoss_scan_triton_fwd = None  # type: ignore[assignment]
    linoss_scan_triton_bwd = None  # type: ignore[assignment]


def _square_2x2(M: Tensor) -> Tensor:
    """
    Square every 2x2 matrix packed row-major in the last dim of ``M``.

    Args:
        M (torch.Tensor): Matrices of shape ``(P, 4)``.

    Returns:
        torch.Tensor: ``M @ M``, shape ``(P, 4)``.
    """
    m11, m12, m21, m22 = M.unbind(-1)
    trace = m11 + m22
    return torch.stack(
        [
            m11 * m11 + m12 * m21,
            m12 * trace,
            m21 * trace,
            m12 * m21 + m22 * m22,
        ],
        dim=-1,
    )


def _apply_2x2(M: Tensor, v: Tensor) -> Tensor:
    """
    Apply the per-state 2x2 matrices ``M`` to a batch of 2-vectors.

    Args:
        M (torch.Tensor): Matrices of shape ``(P, 4)``, row-major.
        v (torch.Tensor): Vectors of shape ``(B, P, L, 2)``.

    Returns:
        torch.Tensor: ``M v``, shape ``(B, P, L, 2)``.
    """
    m11, m12, m21, m22 = (M[:, i].view(1, -1, 1) for i in range(4))
    v1, v2 = v[..., 0], v[..., 1]
    return torch.stack([m11 * v1 + m12 * v2, m21 * v1 + m22 * v2], dim=-1)


def linoss_scan_ref(M: Tensor, F: Tensor) -> Tensor:
    """
    Kogge-Stone reference for ``x_t = M x_{t-1} + F_t`` with ``x_{-1} = 0``.

    ``M`` is time-invariant, so at the doubling round with stride ``s`` every
    position the scan still updates already aggregates exactly ``s`` elements
    and therefore shares the matrix ``M ** s``. Only that single ``(P, 4)``
    matrix has to be carried, never one per timestep.

    Args:
        M (torch.Tensor): Real 2x2 transitions of shape ``(P, 4)``, row-major.
        F (torch.Tensor): Forcing terms of shape ``(B, P, L, 2)``, real or
            complex.

    Returns:
        torch.Tensor: Scanned states of shape ``(B, P, L, 2)``.
    """
    L = F.shape[2]
    if L <= 1:
        return F.clone()

    xs = F
    M_pow = M
    step = 1
    while step < L:
        tail = _apply_2x2(M_pow, xs[:, :, :-step]) + xs[:, :, step:]
        xs = torch.cat([xs[:, :, :step], tail], dim=2)
        M_pow = _square_2x2(M_pow)
        step *= 2
    return xs


def _linoss_scan_bwd_ref(
    M: Tensor, xs: Tensor, d_out: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Analytic gradients of :func:`linoss_scan_ref`, mirroring the Triton kernel.

    The adjoint recurrence is ``g_t = M^T g_{t+1} + d_out_t``, which gives
    ``dF_t = g_t`` and ``dM = sum_{b,t} g_t x_{t-1}^T`` with ``x_{-1} = 0``.

    Args:
        M (torch.Tensor): Real 2x2 transitions of shape ``(P, 4)``.
        xs (torch.Tensor): Forward scanned states of shape ``(B, P, L, 2)``.
        d_out (torch.Tensor): Upstream gradient w.r.t. ``xs``, same shape.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``dM`` of shape ``(P, 4)`` and
            ``dF`` of shape ``(B, P, L, 2)``.
    """
    M_T = M[:, [0, 2, 1, 3]]
    dF = linoss_scan_ref(M_T, d_out.flip(2)).flip(2)

    x_prev = torch.cat([torch.zeros_like(xs[:, :, :1]), xs[:, :, :-1]], dim=2)
    # For complex states PyTorch stores dL/dRe + i dL/dIm, so the real
    # parameter gradient is Re(conj(g) x_prev).
    g = dF.conj() if dF.is_complex() else dF
    dM = torch.einsum("bpli,bplj->pij", g, x_prev).reshape(-1, 4)
    if dM.is_complex():
        dM = dM.real
    return dM.to(M.dtype), dF


def _check_shapes(M: Tensor, F: Tensor) -> None:
    """
    Validate the scan's argument shapes on every backend.

    Args:
        M (torch.Tensor): Real 2x2 transitions, expected shape ``(P, 4)``.
        F (torch.Tensor): Forcing terms, expected shape ``(B, P, L, 2)``.

    Raises:
        ValueError: If a shape is malformed or ``M`` is not real.
    """
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
    if M.is_complex():
        raise ValueError(f"M must be real, got dtype {M.dtype}")


def _triton_available(M: Tensor, F: Tensor) -> bool:
    """Whether the Triton path can serve this pair of inputs."""
    return (
        linoss_scan_triton_fwd is not None
        and linoss_scan_triton_bwd is not None
        and M.is_cuda
        and F.is_cuda
    )


def _scan_triton_fwd(M: Tensor, F: Tensor) -> Tensor:
    """Forward scan on the GPU; complex forcing runs as two real scans."""
    if F.is_complex():
        return torch.complex(
            linoss_scan_triton_fwd(M, F.real.contiguous()),
            linoss_scan_triton_fwd(M, F.imag.contiguous()),
        )
    return linoss_scan_triton_fwd(M, F)


def _scan_triton_bwd(
    M: Tensor, xs: Tensor, d_out: Tensor, needs_dM: bool
) -> Tuple[Tensor, Tensor]:
    """Reverse scan on the GPU; complex states run as two real scans."""
    if xs.is_complex():
        dM_re, dF_re = linoss_scan_triton_bwd(
            M, xs.real.contiguous(), d_out.real.contiguous(), needs_dM
        )
        dM_im, dF_im = linoss_scan_triton_bwd(
            M, xs.imag.contiguous(), d_out.imag.contiguous(), needs_dM
        )
        return dM_re + dM_im, torch.complex(dF_re, dF_im)
    return linoss_scan_triton_bwd(M, xs, d_out, needs_dM)


class LinOSSScanFn(torch.autograd.Function):
    """Autograd wrapper around the LinOSS 2x2 scan."""

    @staticmethod
    @custom_fwd
    def forward(ctx, M: Tensor, F: Tensor) -> Tensor:
        """
        Forward pass of the 2x2 scan.

        Args:
            ctx (Any): Autograd context.
            M (torch.Tensor): Real 2x2 transitions of shape ``(P, 4)``.
            F (torch.Tensor): Forcing terms of shape ``(B, P, L, 2)``.

        Returns:
            torch.Tensor: Scanned states of shape ``(B, P, L, 2)``.
        """
        _check_shapes(M, F)
        use_triton = _triton_available(M, F)
        xs = _scan_triton_fwd(M, F) if use_triton else linoss_scan_ref(M, F)
        ctx.save_for_backward(M, xs)
        ctx.use_triton = use_triton
        return xs

    @staticmethod
    @custom_bwd
    def backward(
        ctx, d_out: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Backward pass of the 2x2 scan.

        Args:
            ctx (Any): Autograd context.
            d_out (torch.Tensor): Gradient w.r.t. the scanned states.

        Returns:
            tuple: Gradients ``(dM, dF)``, each None when not required.
        """
        M, xs = ctx.saved_tensors
        needs_dM, needs_dF = ctx.needs_input_grad[:2]
        if ctx.use_triton:
            dM, dF = _scan_triton_bwd(M, xs, d_out.contiguous(), needs_dM)
        else:
            dM, dF = _linoss_scan_bwd_ref(M, xs, d_out)
        return dM if needs_dM else None, dF if needs_dF else None


def linoss_scan_fn(M: Tensor, F: Tensor) -> Tensor:
    """
    Associative 2x2 scan: ``x_t = M x_{t-1} + F_t`` with ``x_{-1} = 0``.

    Args:
        M (torch.Tensor): Real 2x2 transitions of shape ``(P, 4)``, row-major.
        F (torch.Tensor): Forcing terms of shape ``(B, P, L, 2)``, real or
            complex.

    Returns:
        torch.Tensor: Scanned states of shape ``(B, P, L, 2)``.
    """
    return LinOSSScanFn.apply(M, F)
