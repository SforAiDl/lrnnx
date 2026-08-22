"""
Unit tests for the LinOSS 2x2 time-invariant scan.
"""

import pytest
import torch

from lrnnx.ops.linoss_scan import (
    _linoss_scan_bwd_ref,
    linoss_scan_fn,
    linoss_scan_ref,
)

BATCH_SIZE = 2
STATE_DIM = 4

# Covers sub-chunk, exact-chunk and multi-chunk sequences for the Triton
# kernel's BLOCK_L, plus non-powers of two.
SEQ_LENGTHS = [1, 2, 7, 16, 255, 256, 257, 1024, 4096, 8192]

RTOL = 3e-3
ATOL = 5e-3
RTOLW = 1e-3
ATOLW = 1e-3


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


def sequential_scan(M: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Naive ``x_t = M x_{t-1} + F_t`` recurrence used as ground truth."""
    L = F.shape[2]
    m11, m12, m21, m22 = M.unbind(-1)
    state = torch.zeros(
        F.shape[0], F.shape[1], 2, dtype=F.dtype, device=F.device
    )
    outs = []
    for i in range(L):
        z1 = m11 * state[..., 0] + m12 * state[..., 1] + F[:, :, i, 0]
        z2 = m21 * state[..., 0] + m22 * state[..., 1] + F[:, :, i, 1]
        state = torch.stack([z1, z2], dim=-1)
        outs.append(state)
    return torch.stack(outs, dim=2)


def make_inputs(seq_len: int, complex_f: bool, device: str, dtype):
    """Build a contractive ``M`` and a matching forcing tensor ``F``."""
    M = 0.5 * torch.rand(STATE_DIM, 4, device=device, dtype=dtype)
    F = torch.rand(
        BATCH_SIZE, STATE_DIM, seq_len, 2, device=device, dtype=dtype
    )
    if complex_f:
        F = torch.complex(F, torch.rand_like(F))
    return M, F


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("complex_f", [False, True])
def test_linoss_scan_ref_vs_loop(seq_len, complex_f):
    """The Kogge-Stone reference matches the sequential recurrence."""
    M, F = make_inputs(seq_len, complex_f, "cpu", torch.float64)

    torch.testing.assert_close(
        linoss_scan_ref(M, F), sequential_scan(M, F), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("complex_f", [False, True])
def test_linoss_scan_ref_does_not_alias_input(complex_f):
    """A length-1 scan must copy rather than hand back the input tensor."""
    M, F = make_inputs(1, complex_f, "cpu", torch.float32)

    assert linoss_scan_ref(M, F).data_ptr() != F.data_ptr()


@pytest.mark.parametrize("seq_len", [16, 257, 1024])
@pytest.mark.parametrize("complex_f", [False, True])
def test_linoss_scan_ref_gradients(seq_len, complex_f):
    """The analytic reference backward matches autograd through the loop."""
    M, F = make_inputs(seq_len, complex_f, "cpu", torch.float64)
    M_loop = M.clone().requires_grad_(True)
    F_loop = F.clone().requires_grad_(True)

    out = sequential_scan(M_loop, F_loop)
    d_out = torch.randn_like(out)
    if out.is_complex():
        (out.conj() * d_out).real.sum().backward()
    else:
        (out * d_out).sum().backward()

    dM, dF = _linoss_scan_bwd_ref(M, linoss_scan_ref(M, F), d_out)

    torch.testing.assert_close(dM, M_loop.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(dF, F_loop.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("complex_f", [False, True])
def test_linoss_scan_gradcheck(complex_f):
    """Double precision gradcheck of the public autograd function."""
    M, F = make_inputs(9, complex_f, "cpu", torch.float64)

    assert torch.autograd.gradcheck(
        linoss_scan_fn, (M.requires_grad_(True), F.requires_grad_(True))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("complex_f", [False, True])
def test_linoss_cuda_vs_ref_forward(seq_len, complex_f):
    """Triton forward matches the PyTorch reference."""
    M, F = make_inputs(seq_len, complex_f, "cuda", torch.float32)

    torch.testing.assert_close(
        linoss_scan_fn(M, F), linoss_scan_ref(M, F), rtol=RTOL, atol=ATOL
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("complex_f", [False, True])
def test_linoss_cuda_vs_ref_backward(seq_len, complex_f):
    """Triton reverse-scan gradients match the reference gradients."""
    M, F = make_inputs(seq_len, complex_f, "cuda", torch.float32)
    M_cuda = M.clone().requires_grad_(True)
    F_cuda = F.clone().requires_grad_(True)
    M_ref = M.cpu().clone().requires_grad_(True)
    F_ref = F.cpu().clone().requires_grad_(True)

    for out in (linoss_scan_fn(M_cuda, F_cuda), linoss_scan_fn(M_ref, F_ref)):
        if out.is_complex():
            (out.real.sum() + out.imag.sum()).backward()
        else:
            out.sum().backward()

    torch.testing.assert_close(
        M_cuda.grad.cpu(), M_ref.grad, rtol=RTOLW, atol=ATOLW
    )
    torch.testing.assert_close(
        F_cuda.grad.cpu(), F_ref.grad, rtol=RTOLW, atol=ATOLW
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float16, torch.bfloat16]
)
def test_linoss_cuda_dtypes(dtype):
    """The kernel runs in every supported precision and stays accurate."""
    M, F = make_inputs(512, False, "cuda", torch.float32)

    out = linoss_scan_fn(M.to(dtype), F.to(dtype))

    assert out.dtype == dtype
    torch.testing.assert_close(
        out.float(), linoss_scan_ref(M, F), rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_linoss_cuda_requires_grad_subsets():
    """Gradients are only produced for the inputs that ask for them."""
    M, F = make_inputs(128, False, "cuda", torch.float32)
    M_only = M.clone().requires_grad_(True)
    F_only = F.clone().requires_grad_(True)

    linoss_scan_fn(M_only, F).sum().backward()
    linoss_scan_fn(M, F_only).sum().backward()

    assert M_only.grad is not None and M_only.grad.shape == M.shape
    assert F_only.grad is not None and F_only.grad.shape == F.shape


@pytest.mark.parametrize(
    "M_shape, F_shape",
    [
        ((4, 3), (2, 4, 8, 2)),  # M is not a packed 2x2
        ((4, 4), (2, 4, 8, 3)),  # F does not hold 2-vectors
        ((4, 4), (2, 5, 8, 2)),  # mismatched state dims
    ],
)
def test_linoss_rejects_bad_shapes(device, M_shape, F_shape):
    """Malformed inputs raise instead of reading out of bounds."""
    M = torch.rand(*M_shape, device=device)
    F = torch.rand(*F_shape, device=device)

    with pytest.raises(ValueError):
        linoss_scan_fn(M, F)
