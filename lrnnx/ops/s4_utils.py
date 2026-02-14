import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Function aliases
contract = torch.einsum
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

"""Structured matrix kernels"""

# Try CUDA extension
try:
    from csrc.s4.cauchy import cauchy_mult as cauchy_cuda
    from csrc.s4.vandermonde import log_vandermonde_cuda

    has_cuda_extension = True
except:
    has_cuda_extension = False

# Try pykeops
try:
    import pykeops
    from pykeops.torch import Genred

    has_pykeops = True

    def _broadcast_dims(*tensors):
        max_dim = max([len(tensor.shape) for tensor in tensors])
        tensors = [
            tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape)
            for tensor in tensors
        ]
        return tensors

    def cauchy_keops(v, z, w):
        expr_num = "z * ComplexReal(v) - Real2Complex(Sum(v * w))"
        expr_denom = "ComplexMult(z-w, z-Conj(w))"

        cauchy_mult = Genred(
            f"ComplexDivide({expr_num}, {expr_denom})",
            ["v = Vj(2)", "z = Vi(2)", "w = Vj(2)"],
            reduction_op="Sum",
            axis=1,
        )

        v, z, w = _broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2 * cauchy_mult(v, z, w, backend="GPU")
        return _r2c(r)

    def log_vandermonde_keops(v, x, L):
        expr = "ComplexMult(v, ComplexExp(ComplexMult(x, l)))"
        vandermonde_mult = Genred(
            expr,
            ["v = Vj(2)", "x = Vj(2)", "l = Vi(2)"],
            reduction_op="Sum",
            axis=1,
        )

        l = torch.arange(L).to(x)
        v, x, l = _broadcast_dims(v, x, l)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(v, x, l, backend="GPU")
        return 2 * _r2c(r).real

    def log_vandermonde_transpose_keops(u, v, x, L):
        """
        KeOps implementation of transposed log Vandermonde multiplication.

        Args:
            u (torch.Tensor): Input tensor of shape ``(..., H, L)``.
            v (torch.Tensor): Input tensor of shape ``(..., H, N)``.
            x (torch.Tensor): Input tensor of shape ``(..., H, N)``.
            L (int): Sequence length.

        Returns:
            torch.Tensor: Output tensor of shape ``(..., H, N)``.
        """
        expr = "ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))"
        vandermonde_mult = Genred(
            expr,
            ["u = Vj(2)", "v = Vi(2)", "x = Vi(2)", "l = Vj(2)"],
            reduction_op="Sum",
            axis=1,
        )

        l = torch.arange(L).to(x)
        u, v, x, l = _broadcast_dims(u, v, x, l)
        u = _c2r(u)
        v = _c2r(v)
        x = _c2r(x)
        l = _c2r(l)

        r = vandermonde_mult(u, v, x, l, backend="GPU")
        return _r2c(r)

except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        print(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of "
            "pykeops or the CUDA extension for better speed and memory efficiency."
        )


# Fallback versions
def cauchy_naive(v, z, w):
    """
    Naive PyTorch fallback for Cauchy matrix multiplication.

    Args:
        v (torch.Tensor): Input tensor of shape ``(..., N)``.
        z (torch.Tensor): Input tensor of shape ``(..., L)``.
        w (torch.Tensor): Input tensor of shape ``(..., N)``.

    Returns:
        torch.Tensor: The sum v/(z-w), shape ``(..., L)``.
    """
    v = _conj(v)
    w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    return torch.sum(cauchy_matrix, dim=-2)


def log_vandermonde_naive(v, x, L, conj=True):
    """
    Naive PyTorch fallback for log Vandermonde multiplication.

    Args:
        v (torch.Tensor): Input tensor of shape ``(..., N)``.
        x (torch.Tensor): Input tensor of shape ``(..., N)``.
        L (int): Sequence length.
        conj (bool, optional): Whether to use conjugate symmetry. Defaults to True.

    Returns:
        torch.Tensor: The sum v * x^l, shape ``(..., L)``.
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))
    vandermonde_prod = contract(
        "... n, ... n l -> ... l", v, vandermonde_matrix
    )
    return 2 * vandermonde_prod.real


def log_vandermonde_transpose_naive(u, v, x, L):
    """
    Naive PyTorch fallback for transposed log Vandermonde multiplication.

    Args:
        u (torch.Tensor): Input tensor of shape ``(..., L)``.
        v (torch.Tensor): Input tensor of shape ``(..., N)``.
        x (torch.Tensor): Input tensor of shape ``(..., N)``.
        L (int): Sequence length.

    Returns:
        torch.Tensor: Output tensor of shape ``(..., N)``.
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x))
    vandermonde_prod = contract(
        "... l, ... n, ... n l -> ... n", u.to(x), v.to(x), vandermonde_matrix
    )
    return vandermonde_prod


def get_cauchy_kernel():
    """
    Returns the best available Cauchy multiplication function.

    Returns:
        callable: The Cauchy kernel function (CUDA, KeOps, or naive fallback).
    """
    if has_cuda_extension:
        return cauchy_cuda
    if has_pykeops:
        return cauchy_keops
    return cauchy_naive


def get_vandermonde_kernel():
    """
    Returns the best available Vandermonde multiplication function.

    Returns:
        callable: The Vandermonde kernel function (CUDA, KeOps, or naive fallback).
    """
    if has_cuda_extension:
        return log_vandermonde_cuda
    if has_pykeops:
        return log_vandermonde_keops
    return log_vandermonde_naive


def get_vandermonde_transpose_kernel():
    """
    Returns the best available transpose Vandermonde multiplication function.

    Returns:
        callable: The transposed Vandermonde kernel function (KeOps or naive fallback).
    """
    if has_pykeops:
        return log_vandermonde_transpose_keops
    return log_vandermonde_transpose_naive


"""Helper modules"""


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    transposed=False,
    activate=False,
    **kwargs,
):
    """
    Returns a linear nn.Module with control over axes order, initialization, and activation.

    Args:
        d_input (int): Input dimension.
        d_output (int): Output dimension.
        bias (bool, optional): Whether to use bias. Defaults to True.
        transposed (bool, optional): If True, uses Conv1d instead of Linear. Defaults to False.
        activate (bool, optional): Whether to append a GELU activation. Defaults to False.
        **kwargs: Additional arguments passed to the linear layer.

    Returns:
        torch.nn.Module: The configured linear/activation module.
    """
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate:
        activation = nn.GELU()
        linear = nn.Sequential(linear, activation)
    return linear


class DropoutNd(nn.Module):
    """
    N-dimensional dropout module.

    Args:
        p (float, optional): Dropout probability. Defaults to 0.5.
        tie (bool, optional): Tie dropout mask across sequence lengths (Dropout1d/2d/3d). Defaults to True.
        transposed (bool, optional): Whether the sequence dimension is transposed. Defaults to True.
    """

    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                f"dropout probability has to be in [0, 1), but got {p}"
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """
        Forward pass for DropoutNd.

        Args:
            X (torch.Tensor): Input tensor of shape ``(batch, dim, lengths...)``.

        Returns:
            torch.Tensor: Tensor with dropout applied.
        """
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            mask_shape = (
                X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            )
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            return X
        return X


"""Functional utilities"""


def power(L, A, v=None):
    """
    Compute A^L and the scan sum_i A^i v_i.

    Args:
        L (int): Power to raise A to.
        A (torch.Tensor): Input matrix of shape ``(..., N, N)``.
        v (torch.Tensor, optional): Vector for scan sum, shape ``(..., N, L)``. Defaults to None.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: A^L, or a tuple of (A^L, scan sum).
    """
    I = torch.eye(A.shape[-1]).to(A)

    powers = [A]
    l = 1
    while True:
        if L % 2 == 1:
            I = powers[-1] @ I
        L //= 2
        if L == 0:
            break
        l *= 2
        if v is None:
            powers = [powers[-1] @ powers[-1]]
        else:
            powers.append(powers[-1] @ powers[-1])

    if v is None:
        return I

    # Handle non-power-of-2 case
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_

    # Handle reduction for power of 2
    while v.size(-1) > 1:
        v = rearrange(v, "... (z l) -> ... z l", z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    return I, v.squeeze(-1)


"""HiPPO utilities"""


def transition(measure, N, **measure_args):
    """
    Constructs A, B transition matrices for different measures.

    Args:
        measure (str): Measure type (e.g., "legt", "legs", "fourier").
        N (int): State dimension.
        **measure_args: Additional arguments for the measure.

    Returns:
        tuple[np.ndarray, np.ndarray]: Transition matrices A and B.
    """
    if measure == "legt":
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
        A *= 0.5
        B *= 0.5
    elif measure == "legs":
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
    elif measure in ["fourier", "fout"]:
        freqs = np.arange(N // 2)
        d = np.stack([np.zeros(N // 2), freqs], axis=-1).reshape(-1)[1:]
        A = np.pi * (-np.diag(d, 1) + np.diag(d, -1))
        B = np.zeros(N)
        B[0::2] = 2**0.5
        B[0] = 1
        A = A - B[:, None] * B[None, :]
        B = B[:, None]
    else:
        raise NotImplementedError

    return A, B


def rank_correction(measure, N, rank=1, dtype=torch.float):
    """
    Return low-rank matrix P such that A + PP^T is normal.

    Args:
        measure (str): Measure type.
        N (int): State dimension.
        rank (int, optional): Rank of the correction. Defaults to 1.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float.

    Returns:
        torch.Tensor: Rank correction matrix P.
    """
    if measure == "legs":
        assert rank >= 1
        P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)
    elif measure == "legt":
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)
        P *= 2 ** (-0.5)
    elif measure in ["fourier", "fout"]:
        P = torch.zeros(N)
        P[0::2] = 2**0.5
        P[0] = 1
        P = P.unsqueeze(0)
    else:
        raise NotImplementedError

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)
    return P


def nplr(
    measure,
    N,
    rank=1,
    dtype=torch.float,
    diagonalize_precision=True,
    B_clip=2.0,
):
    """
    Constructs NPLR form of HiPPO matrices.

    Returns w, p, q, V, B such that (w - p q^*, B) is unitarily equivalent to
    the original HiPPO A, B by the matrix V, i.e. A = V[w - p q^*]V^*, B = V B.

    Args:
        measure (str): Measure type.
        N (int): State dimension.
        rank (int, optional): Rank of the correction. Defaults to 1.
        dtype (torch.dtype, optional): Target data type. Defaults to torch.float.
        diagonalize_precision (bool, optional): Whether to diagonalize in double precision. Defaults to True.
        B_clip (float, optional): Clipping value for B. Defaults to 2.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The W, P, B, and V matrices.
    """
    assert dtype == torch.float or dtype == torch.double
    cdtype = torch.cfloat if dtype == torch.float else torch.cdouble

    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]

    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + torch.sum(P.unsqueeze(-2) * P.unsqueeze(-1), dim=-3)

    # Check AP is nearly skew-symmetric
    _A = AP + AP.transpose(-1, -2)
    if (err := torch.sum((_A - _A[0, 0] * torch.eye(N)) ** 2) / N) > 1e-5:
        print("WARNING: HiPPO matrix not skew symmetric", err)

    # Calculate real and imaginary parts separately
    W_re = torch.mean(torch.diagonal(AP), -1, keepdim=True)

    # Diagonalize in double precision
    if diagonalize_precision:
        AP = AP.to(torch.double)
    W_im, V = torch.linalg.eigh(AP * -1j)
    if diagonalize_precision:
        W_im, V = W_im.to(cdtype), V.to(cdtype)
    W = W_re + 1j * W_im

    # Only keep half of each conjugate pair
    _, idx = torch.sort(W.imag)
    W_sorted = W[idx]
    V_sorted = V[:, idx]

    # Handle edge case when eigenvalues can be 0
    V = V_sorted[:, : N // 2]
    W = W_sorted[: N // 2]
    assert (
        W[-2].abs() > 1e-4
    ), "Only 1 zero eigenvalue allowed in diagonal part of A"
    if W[-1].abs() < 1e-4:
        V[:, -1] = 0.0
        V[0, -1] = 2**-0.5
        V[1, -1] = 2**-0.5 * 1j

    _AP = V @ torch.diag_embed(W) @ V.conj().transpose(-1, -2)
    if (err := torch.sum((2 * _AP.real - AP) ** 2) / N) > 1e-5:
        print(
            "Warning: Diagonalization of A matrix not numerically precise - error",
            err,
        )

    V_inv = V.conj().transpose(-1, -2)

    B = contract("ij, j -> i", V_inv, B.to(V))
    P = contract("ij, ...j -> ...i", V_inv, P.to(V))

    if B_clip is not None:
        B = B.real + 1j * torch.clamp(B.imag, min=-B_clip, max=B_clip)

    return W, P, B, V


def dplr(
    init="hippo",
    N=64,
    rank=1,
    H=1,
    dtype=torch.float,
    real_random=False,
    real_scale=1.0,
    imag_random=False,
    imag_scale=1.0,
    B_random=False,
    B_init="constant",
    B_scale=1.0,
    P_scale=1.0,
    normalize=False,
):
    """
    Directly construct a DPLR matrix.

    Args:
        init (str, optional): Initialization method. Defaults to "hippo".
        N (int, optional): State size. Defaults to 64.
        rank (int, optional): Rank for DPLR parameterization. Defaults to 1.
        H (int, optional): Number of independent SSM copies. Defaults to 1.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float.
        real_random (bool, optional): Whether to randomize real part. Defaults to False.
        real_scale (float, optional): Scaling factor for real part. Defaults to 1.0.
        imag_random (bool, optional): Whether to randomize imaginary part. Defaults to False.
        imag_scale (float, optional): Scaling factor for imaginary part. Defaults to 1.0.
        B_random (bool, optional): Deprecated. Defaults to False.
        B_init (str, optional): Initialization method for B. Defaults to "constant".
        B_scale (float, optional): Scaling factor for B. Defaults to 1.0.
        P_scale (float, optional): Scaling factor for P. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize B. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Matrices A, P, B, V.
    """
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)

    # Construct real part of diagonal A
    if real_random:
        real_part = torch.rand(H, N // 2)
    else:
        real_part = 0.5 * torch.ones(H, N // 2)
    real_part = real_scale * real_part

    # Construct imaginary part of diagonal A
    if imag_random:
        imag_part = N // 2 * torch.rand(H, N // 2)
    else:
        imag_part = repeat(torch.arange(N // 2), "n -> h n", h=H)

    if init in ["random", "rand"]:
        imag_part = torch.exp(torch.randn(H, N // 2))
    elif init == "real":
        imag_part = 0 * imag_part
        if real_random:
            real_part = torch.rand(H, N // 2) * N // 2
        else:
            real_part = 1 + repeat(torch.arange(N // 2), "n -> h n", h=H)
    elif init in ["linear", "lin"]:
        imag_part = pi * imag_part
    elif init in ["inverse", "inv"]:
        imag_part = 1 / pi * N * (N / (1 + 2 * imag_part) - 1)
    elif init in ["inverse2", "inv2"]:
        imag_part = 1 / pi * N * (N / (1 + imag_part) - 1)
    elif init in ["quadratic", "quad"]:
        imag_part = 1 / pi * (1 + 2 * imag_part) ** 2
    elif init in ["legs", "hippo"]:
        A, _, _, _ = nplr("legs", N)
        imag_part = -A.imag
    else:
        raise NotImplementedError
    imag_part = imag_scale * imag_part

    # Construct diagonal A
    A = -real_part - 1j * imag_part
    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)

    # Initialize B
    if B_random:
        print("'B_random' is deprecated in favor of B_init='random'")
    if init in ["legs", "hippo"]:
        print(f"Initializing with S4D-LegS and ignoring argument {B_init=}")
        _, P, B, _ = nplr("legs", N, B_clip=2.0)
        B = repeat(B, "n -> h n", h=H).clone().contiguous()
    elif B_init == "constant":
        B = torch.ones(H, N // 2, dtype=dtype)
    elif B_init == "random":
        B = torch.randn(H, N // 2, dtype=dtype)
    elif B_init == "alternating":
        B = torch.ones(H, N // 4, 2, dtype=dtype)
        B[:, :, 1] *= -1
        B = B.view(H, N // 2)
    elif B_init == "unit-cw":
        z = torch.tensor(torch.exp(-2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, "n -> h n", h=H).clone().contiguous()
    elif B_init == "unit-ccw":
        z = torch.tensor(torch.exp(2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, "n -> h n", h=H).clone().contiguous()
    else:
        raise NotImplementedError
    B *= B_scale

    if normalize:
        norm = -B / A
        zeta = 2 * torch.sum(torch.abs(norm) ** 2, dim=-1, keepdim=True)
        B = B / zeta**0.5

    # Initialize P
    if B_init in ["legs", "hippo"]:
        P = repeat(P, "r n -> r h n", h=H).clone().contiguous()
    else:
        P = torch.randn(rank, H, N // 2, dtype=dtype)
        P = P * P_scale

    # Initialize V (only used in testing)
    V = torch.eye(N, dtype=dtype)[:, : N // 2]
    V = repeat(V, "n m -> h n m", h=H)

    return A, P, B, V


def ssm(init, N, R, H, **ssm_args):
    """
    Dispatcher to create single SSM initialization.

    Args:
        init (str): Initialization method.
        N (int): State size.
        R (int): Rank (for DPLR parameterization).
        H (int): Number of independent SSM copies.
        **ssm_args: Additional arguments.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Matrices A, P, B, V.
    """
    if init.startswith("diag") or init.startswith("dplr"):
        if init.startswith("diag"):
            ssm_args["P_scale"] = 0.0
        args = init[4:].split("-")
        assert args[0] == ""
        if len(args) > 1:
            ssm_args["init"] = args[1]
        A, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    else:
        A, P, B, V = nplr(init, N, R, **ssm_args)
        A = repeat(A, "n -> s n", s=H)
        P = repeat(P, "r n -> r s n", s=H)
        B = repeat(B, "n -> s n", s=H)
        V = repeat(V, "n m -> s n m", s=H)
    return A, P, B, V


combinations = {
    "hippo": ["legs", "fourier"],
    "diag": ["diag-inv", "diag-lin"],
    "all": ["legs", "fourier", "diag-inv", "diag-lin"],
}


def combination(inits, N, R, S, **ssm_args):
    """
    Create combination of SSM initializations.

    Args:
        inits (str | list[str]): Initialization methods.
        N (int): State size.
        R (int): Rank.
        S (int): Number of SSM copies.
        **ssm_args: Additional arguments.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Combined matrices A, P, B, V.
    """
    if isinstance(inits, str):
        inits = combinations[inits] if inits in combinations else [inits]

    assert (
        S % len(inits) == 0
    ), f"{S} SSM copies must be multiple of {len(inits)} inits"
    A, P, B, V = zip(
        *[ssm(init, N, R, S // len(inits), **ssm_args) for init in inits]
    )
    A = torch.cat(A, dim=0)
    P = torch.cat(P, dim=1)
    B = torch.cat(B, dim=0)
    V = torch.cat(V, dim=0)
    return A, P, B, V


"""Transform utilities"""


def inv_transform(param, transform="none"):
    """
    Initialize a (positive) parameter under a transform.

    Args:
        param (torch.Tensor): Parameter tensor.
        transform (str, optional): Transform type ("none", "exp", "relu", "sigmoid", "softplus"). Defaults to "none".

    Returns:
        torch.Tensor: Transformed parameter.
    """
    param = torch.clamp(param, min=1e-4)
    if transform == "none":
        return param
    elif transform == "exp":
        return torch.log(param)
    elif transform == "relu":
        return param
    elif transform == "sigmoid":
        return torch.logit(param)
    elif transform == "softplus":
        return torch.log(torch.exp(param) - 1)
    else:
        raise NotImplementedError


def param_transform(param, transform="none"):
    """
    Get a (positive) parameter under a transform.

    Args:
        param (torch.Tensor): Parameter tensor.
        transform (str, optional): Transform type. Defaults to "none".

    Returns:
        torch.Tensor: Transformed parameter.
    """
    if transform == "none":
        p = param
    elif transform == "exp":
        p = torch.exp(param)
    elif transform == "relu":
        p = F.relu(param) + 1e-4
    elif transform == "sigmoid":
        p = F.sigmoid(param)
    elif transform == "softplus":
        p = F.softplus(param)
    else:
        raise NotImplementedError
    return p


"""SSM parameter processing utilities"""


def init_dt(
    H,
    N,
    dt_min=0.001,
    dt_max=0.1,
    dt_tie=True,
    dt_transform="exp",
    deterministic=False,
    dtype=torch.float,
):
    """
    Initialize dt parameter.

    Args:
        H (int): Model dimension (number of independent SSMs).
        N (int): State size.
        dt_min (float, optional): Minimum dt value. Defaults to 0.001.
        dt_max (float, optional): Maximum dt value. Defaults to 0.1.
        dt_tie (bool, optional): Whether to tie dt across dimensions. Defaults to True.
        dt_transform (str, optional): Transform type for dt. Defaults to "exp".
        deterministic (bool, optional): Whether to use deterministic initialization. Defaults to False.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float.

    Returns:
        torch.Tensor: Initialized (inverse transformed) dt parameter.
    """
    if deterministic:
        assert dt_tie, "Deterministic dt initialization is tied"
        assert (
            dt_transform == "exp"
        ), "Deterministic dt transform should be 'exp'"
        inv_dt = torch.exp(
            torch.linspace(math.log(dt_min), math.log(dt_max), H)
        ).unsqueeze(-1)
    else:
        shape = (H, 1) if dt_tie else (H, N // 2)
        inv_dt = torch.rand(*shape, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        if dt_transform != "exp":
            inv_dt = inv_transform(torch.exp(inv_dt), dt_transform)
    return inv_dt


def init_ssm_dplr(
    N,
    H,
    n_ssm,
    channels,
    rank,
    init,
    deterministic=False,
    cdtype=torch.cfloat,
    **init_args,
):
    """
    Initialize DPLR (A, P, B, C) parameters and return broadcast repeat factor.

    Args:
        N (int): State size.
        H (int): Model dimension.
        n_ssm (int): Number of independent SSM copies.
        channels (int): Number of channels.
        rank (int): Rank.
        init (str): Initialization method.
        deterministic (bool, optional): Whether to use deterministic initialization. Defaults to False.
        cdtype (torch.dtype, optional): Complex data type. Defaults to torch.cfloat.
        **init_args: Additional initialization arguments.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tensors A, P, B, C.
    """
    A, P, B, V = combination(init, N, rank, n_ssm, **init_args)

    # Broadcast C to have H channels
    if deterministic:
        C = torch.zeros(channels, n_ssm, N, dtype=cdtype)
        C[:, :, :1] = 1.0
        C = contract("hmn, chn -> chm", V.conj().transpose(-1, -2), C)
        C = (
            repeat(C, "c t n -> c (v t) n", v=H // C.size(-2))
            .clone()
            .contiguous()
        )
    else:
        C = torch.randn(channels, H, N // 2, dtype=cdtype)

    # Broadcast other parameters to have n_ssm copies
    assert (
        n_ssm % B.size(-2) == 0
        and n_ssm % P.size(-2) == 0
        and n_ssm % A.size(-2) == 0
    )

    B = repeat(B, "t n -> (v t) n", v=n_ssm // B.size(-2)).clone().contiguous()
    P = (
        repeat(P, "r t n -> r (v t) n", v=n_ssm // P.size(-2))
        .clone()
        .contiguous()
    )
    A = repeat(A, "t n -> (v t) n", v=n_ssm // A.size(-2)).clone().contiguous()

    # N will be halved by caller for conjugate symmetry
    return A, P, B, C


def register_ssm_params(
    module,
    A,
    B,
    C,
    inv_dt,
    P,
    H,
    n_ssm,
    N,
    channels,
    rank,
    dt_fast,
    real_transform,
    imag_transform,
    is_real,
    verbose,
    l_max,
    diag=False,
):
    """
    Register SSM parameters on a module.

    Args:
        module (torch.nn.Module): The module to register parameters on.
        A (torch.Tensor): Tensor A.
        B (torch.Tensor): Tensor B.
        C (torch.Tensor): Tensor C.
        inv_dt (torch.Tensor): Inverse dt tensor.
        P (torch.Tensor): Tensor P.
        H (int): Model dimension.
        n_ssm (int): Number of independent SSMs.
        N (int): State size.
        channels (int): Number of channels.
        rank (int): Rank.
        dt_fast (bool): Whether dt is fast.
        real_transform (str): Transform for the real part.
        imag_transform (str): Transform for the imaginary part.
        is_real (bool): Whether parameters are real.
        verbose (bool): Whether to print construction details.
        l_max (int): Maximum sequence length.
        diag (bool, optional): If True, skip P registration and rank checks (for diagonal S4D). Defaults to False.

    Returns:
        int: The repeat broadcast factor.
    """
    if dt_fast:
        inv_dt = torch.asinh(inv_dt)

    assert H == inv_dt.size(0)
    assert n_ssm == A.size(-2) == B.size(-2)
    repeat = H // A.size(0)

    # Check shapes
    if not diag:
        # DPLR-specific checks
        assert rank == P.shape[-3]
        assert N == P.size(-1) == A.size(-1) == B.size(-1) == C.size(-1)
        assert n_ssm == P.size(-2)
    else:
        # Diagonal checks
        assert N == A.size(-1) == B.size(-1) == C.size(-1)

    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)

    # Broadcast C and B
    C = C.expand(torch.broadcast_shapes(C.shape, (1, H, N)))
    B = B.unsqueeze(0)
    assert channels == C.shape[0]

    if verbose:
        print(f"Constructing S4 (H, N, L) = ({H}, {N}, {l_max})")

    # Register helper
    def register(name, tensor, lr=None, wd=0.0):
        if lr == 0.0:
            module.register_buffer(name, tensor)
        else:
            module.register_parameter(name, nn.Parameter(tensor))
            optim = {}
            if lr is not None:
                optim["lr"] = lr
            if wd is not None:
                optim["weight_decay"] = wd
            setattr(getattr(module, name), "_optim", optim)

    # Register parameters
    register("inv_dt", inv_dt)

    if is_real:
        register("C", C.real)
        register("B", B.real)
        register("A_real", inv_transform(-A.real, real_transform))
    else:
        register("C", _c2r(_resolve_conj(C)))
        register("B", _c2r(B))
        register("A_real", inv_transform(-A.real, real_transform))
        register("A_imag", inv_transform(-A.imag, imag_transform))

    # Register P only for DPLR (not diagonal)
    if not diag:
        register("P", _c2r(P))

    return repeat


def process_ssm_params(
    A_real,
    A_imag,
    B,
    C,
    inv_dt,
    real_transform="exp",
    imag_transform="none",
    dt_transform="exp",
    dt_fast=False,
    is_real=False,
    repeat_factor=1,
    rate=1.0,
):
    """
    Process SSM parameters from stored form to usable form.

    Args:
        A_real (torch.Tensor): Real part of A.
        A_imag (torch.Tensor): Imaginary part of A.
        B (torch.Tensor): Tensor B.
        C (torch.Tensor): Tensor C.
        inv_dt (torch.Tensor): Inverse dt tensor.
        real_transform (str, optional): Transform for A_real. Defaults to "exp".
        imag_transform (str, optional): Transform for A_imag. Defaults to "none".
        dt_transform (str, optional): Transform for dt. Defaults to "exp".
        dt_fast (bool, optional): Whether dt is fast. Defaults to False.
        is_real (bool, optional): Whether parameters are real. Defaults to False.
        repeat_factor (int, optional): Broadcast repeat factor. Defaults to 1.
        rate (float, optional): Sampling rate. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Processed dt, A, B, C, dtA.
    """
    # Process A
    if is_real:
        A = -param_transform(A_real, real_transform)
        B = B
        C = C
    else:
        A = -param_transform(A_real, real_transform) - 1j * param_transform(
            A_imag, imag_transform
        )
        B = _r2c(B)
        C = _r2c(C)

    # Process dt
    if dt_fast:
        inv_dt = torch.sinh(inv_dt)
    dt = param_transform(inv_dt, dt_transform) * rate

    # Broadcast A and B
    A = repeat(A, "t n -> (v t) n", v=repeat_factor)
    B = repeat(B, "b t n -> b (v t) n", v=repeat_factor)
    dtA = dt * A

    return dt, A, B, C, dtA


def process_dplr_params(
    A_real,
    A_imag,
    B,
    C,
    P,
    inv_dt,
    real_transform="exp",
    imag_transform="none",
    dt_transform="exp",
    dt_fast=False,
    is_real=False,
    repeat_factor=1,
    rate=1.0,
):
    """
    Process DPLR SSM parameters including P and Q matrices.

    Args:
        A_real (torch.Tensor): Real part of A.
        A_imag (torch.Tensor): Imaginary part of A.
        B (torch.Tensor): Tensor B.
        C (torch.Tensor): Tensor C.
        P (torch.Tensor): Tensor P.
        inv_dt (torch.Tensor): Inverse dt tensor.
        real_transform (str, optional): Transform for A_real. Defaults to "exp".
        imag_transform (str, optional): Transform for A_imag. Defaults to "none".
        dt_transform (str, optional): Transform for dt. Defaults to "exp".
        dt_fast (bool, optional): Whether dt is fast. Defaults to False.
        is_real (bool, optional): Whether parameters are real. Defaults to False.
        repeat_factor (int, optional): Broadcast repeat factor. Defaults to 1.
        rate (float, optional): Sampling rate. Defaults to 1.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Processed dt, A, B, C, P, Q.
    """
    dt, A, B, C, dtA = process_ssm_params(
        A_real,
        A_imag,
        B,
        C,
        inv_dt,
        real_transform,
        imag_transform,
        dt_transform,
        dt_fast,
        is_real,
        repeat_factor,
        rate,
    )

    # Get P and Q for DPLR form
    P = _r2c(P)
    P = repeat(P, "r t n -> r (v t) n", v=repeat_factor)
    Q = P.conj()

    return dt, A, B, C, P, Q


def setup_default_state(C, N, H, batch_shape, step_mode="dense"):
    """
    Create default SSM state.

    Args:
        C (torch.Tensor): Tensor C to derive dtype and device from.
        N (int): State size.
        H (int): Model dimension.
        batch_shape (tuple): Batch shape dimensions.
        step_mode (str, optional): Step mode ("dense", "linear", etc.). Defaults to "dense".

    Returns:
        torch.Tensor: Zero-initialized state tensor.
    """
    if step_mode != "linear":
        N *= 2
    state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
    return state