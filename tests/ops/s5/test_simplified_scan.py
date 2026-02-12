import numpy as np
import pytest
import torch

from lrnnx.ops.simplified_scan import (
    simplified_scan_fn,
    simplified_scan_ref,
)
from lrnnx.utils.init import (
    init_CV,
    init_log_steps,
    init_VinvB,
    make_DPLR_HiPPO,
)

BATCH_SIZES = [2]
SEQ_LENGTHS = [8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
HID_DIMS = [8]
STATE_DIMS = [16]
DISCRETIZATIONS = ["bilinear", "zoh", "dirac"]


RTOL = 3e-3
ATOL = 5e-3
RTOLW = 1e-3
ATOLW = 1e-3


def init_ssm_params(H: int, P: int, conj_sym: bool = True, device="cuda"):
    """
    Initialize SSM parameters (A, B, C, dt) using HiPPO initialization.

    Args:
        H: Hidden dimension
        P: State dimension
        conj_sym: Whether to use conjugate symmetry
        device: Device to place tensors on

    Returns:
        A: Complex eigenvalues (P,)
        B: Complex projection (P, H)
        C: Complex projection (H, P)
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

    dt = init_log_steps(P).exp().to(device)

    return A, B, C, dt


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_forward_consistency(
    batch_size, seq_len, hid_dim, state_dim, discretization
):
    """Test CUDA and reference implementations produce consistent forward results."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    A, B, C, dt = init_ssm_params(
        hid_dim, state_dim, conj_sym=True, device=device
    )

    u = torch.randn(
        batch_size, hid_dim, seq_len, dtype=torch.complex64, device=device
    )
    delta = (
        dt.unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, state_dim, seq_len)
        .contiguous()
    )

    # Forward pass
    with torch.no_grad():
        y_cuda = simplified_scan_fn(
            u, delta, A, B, C, discretization=discretization
        )
        y_ref = simplified_scan_ref(
            u, delta, A, B, C, discretization=discretization
        )

    assert torch.allclose(y_cuda, y_ref, rtol=RTOL, atol=ATOL), (
        f"Forward pass mismatch (B={batch_size}, L={seq_len}, H={hid_dim}, P={state_dim}, {discretization}): "
        f"max_diff={(y_cuda - y_ref).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_gradient_consistency(
    batch_size, seq_len, hid_dim, state_dim, discretization
):
    """Test gradient consistency for all parameters (u, delta, A, B, C)."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Initialize SSM params using HiPPO
    A_init, B_init, C_init, dt = init_ssm_params(
        hid_dim, state_dim, conj_sym=True, device=device
    )
    delta_base = (
        dt.unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, state_dim, seq_len)
        .contiguous()
    )

    u_cuda = torch.randn(
        batch_size,
        hid_dim,
        seq_len,
        dtype=torch.complex64,
        device=device,
        requires_grad=True,
    )
    delta_cuda = delta_base.clone().requires_grad_(True)
    A_cuda = A_init.clone().requires_grad_(True)
    B_cuda = B_init.clone().requires_grad_(True)
    C_cuda = C_init.clone().requires_grad_(True)

    u_ref = u_cuda.clone().detach().requires_grad_(True)
    delta_ref = delta_base.clone().requires_grad_(True)
    A_ref = A_init.clone().requires_grad_(True)
    B_ref = B_init.clone().requires_grad_(True)
    C_ref = C_init.clone().requires_grad_(True)

    y_cuda = simplified_scan_fn(
        u_cuda,
        delta_cuda,
        A_cuda,
        B_cuda,
        C_cuda,
        discretization=discretization,
    )
    y_ref = simplified_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, discretization=discretization
    )

    g = torch.randn_like(y_cuda)
    y_cuda.backward(g)
    y_ref.backward(g)

    params = [
        ("u", u_cuda, u_ref),
        ("delta", delta_cuda, delta_ref),
        ("A", A_cuda, A_ref),
        ("B", B_cuda, B_ref),
        ("C", C_cuda, C_ref),
    ]

    for name, cuda_param, ref_param in params:
        assert cuda_param.grad is not None, f"CUDA gradient for {name} is None"
        assert (
            ref_param.grad is not None
        ), f"Reference gradient for {name} is None"
        if "delta" in name:
            # delta can have larger numerical errors due to its role in exponentials
            rtol, atol = RTOL * 5, ATOL * 10
        elif "u" in name:
            rtol, atol = RTOL * 2, ATOL * 5
        elif "A" in name:
            rtol, atol = RTOLW, ATOLW * 5
        else:
            rtol, atol = RTOLW, ATOLW

        assert torch.allclose(
            cuda_param.grad, ref_param.grad, rtol=rtol, atol=atol
        ), (
            f"Gradient mismatch for {name} (B={batch_size}, L={seq_len}, H={hid_dim}, P={state_dim}, {discretization}): "
            f"max_diff={(cuda_param.grad - ref_param.grad).abs().max().item():.6e}"
        )
