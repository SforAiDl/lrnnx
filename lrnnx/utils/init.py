"""
Utility functions for SSM initialization.
Reference: https://github.com/lindermanlab/S5
"""

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def make_HiPPO(N: int) -> np.ndarray:
    """Create a HiPPO-LegS matrix."""
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make NPLR representation of HiPPO-LegS."""
    hippo = make_HiPPO(N)
    P = np.sqrt(np.arange(N) + 0.5)
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make DPLR representation of HiPPO-LegS."""
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, np.newaxis] * P[np.newaxis, :]
    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    P_transformed = V.conj().T @ P
    B_orig = B.copy()
    B_transformed = V.conj().T @ B
    Lambda = Lambda_real + 1j * Lambda_imag
    return Lambda, P_transformed, B_transformed, V, B_orig


def init_log_steps(
    P: int,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
) -> Tensor:
    """Initialize learnable timescale parameters.

    Args:
        P: State dimension (number of timescales).
        dt_min: Minimum timescale.
        dt_max: Maximum timescale.

    Returns:
        Log-timescales of shape (P,).
    """
    return torch.empty(P).uniform_(math.log(dt_min), math.log(dt_max))


def init_VinvB(
    Vinv: np.ndarray,
    local_P: int,
    H: int,
) -> Tensor:
    """Initialize B_tilde = V^{-1} @ B with lecun-style scaling.

    Args:
        Vinv: Inverse eigenvectors of shape (P, local_P).
        local_P: Local state dimension (2*P if conj_sym else P).
        H: Hidden dimension.

    Returns:
        B_tilde of shape (P, H, 2) for real/imag parameterization.
    """
    B = np.random.randn(local_P, H).astype(np.float32) * np.sqrt(1.0 / local_P)
    VinvB = Vinv @ B
    return torch.tensor(
        np.stack([VinvB.real, VinvB.imag], axis=-1).astype(np.float32)
    )


def init_CV(
    V: np.ndarray,
    local_P: int,
    H: int,
) -> Tensor:
    """Initialize C_tilde = C @ V with truncated normal.

    Args:
        V: Eigenvectors of shape (local_P, P).
        local_P: Local state dimension (2*P if conj_sym else P).
        H: Hidden dimension.

    Returns:
        C_tilde of shape (H, P, 2) for real/imag parameterization.
    """
    C = (
        np.random.randn(H, local_P).astype(np.float32)
        + 1j * np.random.randn(H, local_P).astype(np.float32)
    ) * np.sqrt(1.0 / local_P)
    CV = C @ V
    return torch.tensor(
        np.stack([CV.real, CV.imag], axis=-1).astype(np.float32)
    )
