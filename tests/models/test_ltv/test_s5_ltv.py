"""
Tests for LTV S5 model with simplified_scan CUDA kernel.
"""

import numpy as np
import pytest
import torch

from lrnnx.models.ltv import S5
from lrnnx.ops.simplified_scan import s5_inner_fn, s5_inner_ref
from lrnnx.utils.init import (
    init_CV,
    init_log_steps,
    init_VinvB,
    make_DPLR_HiPPO,
)

BATCH_SIZE = 4
SEQ_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096]
DISCRETIZATIONS = ["bilinear", "zoh", "dirac"]

RTOL = 3e-3
ATOL = 5e-3
RTOLW = 1e-3
ATOLW = 1e-3


def init_s5_params(H: int, P: int, conj_sym: bool = True, device="cuda"):
    """
    Initialize S5 parameters (A, B, C, D, dt) using HiPPO initialization.

    Args:
        H: Hidden dimension
        P: State dimension
        conj_sym: Whether to use conjugate symmetry
        device: Device to place tensors on

    Returns:
        A: Complex eigenvalues (P,)
        B: Complex projection (P, H)
        C: Complex projection (H, P)
        D: Real skip connection (H,)
        dt: Real timesteps (P,)
    """
    np.random.seed(42)

    if conj_sym:
        Lambda, _, _, V, _ = make_DPLR_HiPPO(2 * P)
        Lambda = Lambda[:P]
        V = V[:, :P]
        Vinv = V.conj().T
        local_P = 2 * P
    else:
        Lambda, _, _, V, _ = make_DPLR_HiPPO(P)
        Vinv = np.linalg.inv(V)
        local_P = P

    A = torch.tensor(Lambda, dtype=torch.complex64, device=device)

    B_tensor = init_VinvB(Vinv, local_P, H)
    B = torch.complex(B_tensor[..., 0], B_tensor[..., 1]).to(device)

    C_tensor = init_CV(V, local_P, H)
    C = torch.complex(C_tensor[..., 0], C_tensor[..., 1]).to(device)

    D = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    dt = init_log_steps(P).exp().to(device)

    return A, B, C, D, dt


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("conj_sym", [True, False])
def test_inner_fn_vs_ref_forward(seq_len, discretization, conj_sym):
    """Test s5_inner_fn matches s5_inner_ref on forward pass."""
    H, P = 32, 16
    device = torch.device("cuda")
    torch.manual_seed(42)

    A, B, C, D, dt = init_s5_params(H, P, conj_sym=conj_sym, device=device)

    u = torch.randn(
        BATCH_SIZE, H, seq_len, dtype=torch.complex64, device=device
    )
    delta = (
        dt.unsqueeze(0)
        .unsqueeze(-1)
        .expand(BATCH_SIZE, P, seq_len)
        .contiguous()
    )

    y_fn = s5_inner_fn(
        u, delta, A, B, C, D, discretization=discretization, conj_sym=conj_sym
    )
    y_ref = s5_inner_ref(
        u, delta, A, B, C, D, discretization=discretization, conj_sym=conj_sym
    )

    assert torch.allclose(y_fn, y_ref, rtol=RTOL, atol=ATOL), (
        f"Forward mismatch (L={seq_len}, {discretization}, conj_sym={conj_sym}): "
        f"max_diff={(y_fn - y_ref).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("conj_sym", [True, False])
def test_inner_fn_vs_ref_backward(seq_len, discretization, conj_sym):
    """Test s5_inner_fn gradients match s5_inner_ref gradients."""
    H, P = 32, 16
    device = torch.device("cuda")
    torch.manual_seed(42)

    A_init, B_init, C_init, D_init, dt = init_s5_params(
        H, P, conj_sym=conj_sym, device=device
    )
    delta_base = (
        dt.unsqueeze(0)
        .unsqueeze(-1)
        .expand(BATCH_SIZE, P, seq_len)
        .contiguous()
    )

    u_fn = torch.randn(
        BATCH_SIZE,
        H,
        seq_len,
        dtype=torch.complex64,
        device=device,
        requires_grad=True,
    )

    A_fn = A_init.clone().requires_grad_(True)
    B_fn = B_init.clone().requires_grad_(True)
    C_fn = C_init.clone().requires_grad_(True)
    D_fn = D_init.clone().requires_grad_(True)

    u_ref = u_fn.clone().detach().requires_grad_(True)
    A_ref = A_init.clone().requires_grad_(True)
    B_ref = B_init.clone().requires_grad_(True)
    C_ref = C_init.clone().requires_grad_(True)
    D_ref = D_init.clone().requires_grad_(True)

    y_fn = s5_inner_fn(
        u_fn,
        delta_base,
        A_fn,
        B_fn,
        C_fn,
        D_fn,
        discretization=discretization,
        conj_sym=conj_sym,
    )
    y_ref = s5_inner_ref(
        u_ref,
        delta_base,
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        discretization=discretization,
        conj_sym=conj_sym,
    )

    g = torch.randn_like(y_fn)
    y_fn.backward(g)
    y_ref.backward(g)

    grad_pairs = [
        ("u", u_fn, u_ref, RTOL * 2, ATOL * 5),
        ("A", A_fn, A_ref, RTOLW, ATOLW * 5),
        ("B", B_fn, B_ref, RTOLW, ATOLW),
        ("C", C_fn, C_ref, RTOLW, ATOLW),
        ("D", D_fn, D_ref, RTOLW, ATOLW),
    ]

    for name, fn_param, ref_param, rtol, atol in grad_pairs:
        assert fn_param.grad is not None, f"No gradient for {name} (fn)"
        assert ref_param.grad is not None, f"No gradient for {name} (ref)"
        assert torch.allclose(
            fn_param.grad, ref_param.grad, rtol=rtol, atol=atol
        ), (
            f"Gradient mismatch for {name} (L={seq_len}, {discretization}, conj_sym={conj_sym}): "
            f"max_diff={(fn_param.grad - ref_param.grad).abs().max().item():.6e}"
        )


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("use_fast_path", [True, False])
def test_step_matches_forward(device, seq_len, discretization, use_fast_path):
    """Test step-by-step inference matches parallel forward."""
    if use_fast_path and not torch.cuda.is_available():
        pytest.skip("CUDA not available for fast path")

    H, P = 32, 16
    torch.manual_seed(42)

    model = S5(
        d_model=H,
        d_state=P,
        discretization=discretization,
        use_fast_path=use_fast_path,
    ).to(device)
    model.eval()

    x = torch.randn(BATCH_SIZE, seq_len, H, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_parallel = model(x)

    cache = model.allocate_inference_cache(BATCH_SIZE, max_seqlen=seq_len)
    y_steps = []

    with torch.no_grad():
        for t in range(seq_len):
            y_t, cache = model.step(x[:, t, :], cache)
            y_steps.append(y_t.squeeze(1))

    y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch (L={seq_len}, {discretization}, fast={use_fast_path}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_cuda_vs_ref_model_forward(seq_len, discretization):
    """Test CUDA model matches reference model."""
    H, P = 32, 16
    device = torch.device("cuda")
    torch.manual_seed(42)

    model_cuda = S5(
        d_model=H,
        d_state=P,
        discretization=discretization,
        use_fast_path=True,
    ).to(device)
    model_ref = S5(
        d_model=H,
        d_state=P,
        discretization=discretization,
        use_fast_path=False,
    ).to(device)
    model_ref.load_state_dict(model_cuda.state_dict())
    model_cuda.eval()
    model_ref.eval()

    x = torch.randn(BATCH_SIZE, seq_len, H, dtype=torch.float32, device=device)

    with torch.no_grad():
        y_cuda = model_cuda(x)
        y_ref = model_ref(x)

    assert torch.allclose(y_cuda, y_ref, rtol=RTOL, atol=ATOL), (
        f"CUDA/Ref mismatch (L={seq_len}, {discretization}): "
        f"max_diff={(y_cuda - y_ref).abs().max().item():.6e}"
    )
