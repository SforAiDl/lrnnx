import pytest
import torch

from lrnnx.ops.s7_scan import s7_scan_fn, s7_scan_ref
from lrnnx.utils.init import make_DPLR_HiPPO

BATCH_SIZES = [2, 64]
SEQ_LENGTHS = [8, 16, 64, 128, 256, 512, 1024, 2048, 4096]
HID_DIMS = [8, 64]
STATE_DIMS = [16, 32]

RTOL = 3e-3
ATOL = 5e-3
RTOLW = 1e-3
ATOLW = 1e-3


def init_s7_params(
    batch_size: int, hid_dim: int, state_dim: int, seq_len: int, device="cuda"
):
    torch.manual_seed(42)

    u = torch.randn(
        batch_size, hid_dim, seq_len, dtype=torch.float32, device=device
    )

    A = 0.02 * torch.randn(
        batch_size, state_dim, seq_len, dtype=torch.float32, device=device
    ) + torch.tensor(
        make_DPLR_HiPPO(state_dim)[0].reshape(1, state_dim, 1),
        device=device,
        dtype=torch.float32,
    )

    B = (
        torch.randn(
            batch_size,
            state_dim,
            hid_dim,
            seq_len,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )

    C = (
        torch.randn(
            batch_size,
            hid_dim,
            state_dim,
            seq_len,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )

    bias = torch.randn(
        batch_size, state_dim, seq_len, dtype=torch.float32, device=device
    )

    return u, A, B, C, bias


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_forward_consistency(batch_size, seq_len, hid_dim, state_dim):
    torch.manual_seed(42)
    device = torch.device("cuda")

    u, A, B, C, bias = init_s7_params(
        batch_size, hid_dim, state_dim, seq_len, device=device
    )

    with torch.no_grad():
        y_cuda = s7_scan_fn(u, A, B, C, bias)
        y_ref = s7_scan_ref(u, A, B, C, bias)

    assert torch.allclose(y_cuda, y_ref, rtol=RTOL, atol=ATOL), (
        f"Forward pass mismatch (B={batch_size}, L={seq_len}, H={hid_dim}, N={state_dim}): "
        f"max_diff={(y_cuda - y_ref).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_gradient_consistency(batch_size, seq_len, hid_dim, state_dim):
    torch.manual_seed(42)
    device = torch.device("cuda")

    u_init, A_init, B_init, C_init, bias_init = init_s7_params(
        batch_size, hid_dim, state_dim, seq_len, device=device
    )

    u_cuda = u_init.clone().requires_grad_(True)
    A_cuda = A_init.clone().requires_grad_(True)
    B_cuda = B_init.clone().requires_grad_(True)
    C_cuda = C_init.clone().requires_grad_(True)
    bias_cuda = bias_init.clone().requires_grad_(True)

    u_ref = u_init.clone().requires_grad_(True)
    A_ref = A_init.clone().requires_grad_(True)
    B_ref = B_init.clone().requires_grad_(True)
    C_ref = C_init.clone().requires_grad_(True)
    bias_ref = bias_init.clone().requires_grad_(True)

    y_cuda = s7_scan_fn(u_cuda, A_cuda, B_cuda, C_cuda, bias_cuda)
    y_ref = s7_scan_ref(u_ref, A_ref, B_ref, C_ref, bias_ref)

    g = torch.randn_like(y_cuda)
    y_cuda.backward(g)
    y_ref.backward(g)

    params = [
        ("u", u_cuda, u_ref),
        ("A", A_cuda, A_ref),
        ("B", B_cuda, B_ref),
        ("C", C_cuda, C_ref),
        ("bias", bias_cuda, bias_ref),
    ]

    for name, cuda_param, ref_param in params:
        assert cuda_param.grad is not None, f"CUDA gradient for {name} is None"
        assert (
            ref_param.grad is not None
        ), f"Reference gradient for {name} is None"
        if name == "A":
            rtol, atol = RTOL * 5, ATOL * 10
        elif name == "u":
            rtol, atol = RTOL * 2, ATOL * 5
        else:
            rtol, atol = RTOLW, ATOLW

        assert torch.allclose(
            cuda_param.grad, ref_param.grad, rtol=rtol, atol=atol
        ), (
            f"Gradient mismatch for {name} (B={batch_size}, L={seq_len}, H={hid_dim}, N={state_dim}): "
            f"max_diff={(cuda_param.grad - ref_param.grad).abs().max().item():.6e}"
        )
