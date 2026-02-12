import math

import pytest
import torch

from lrnnx.ops.rglru_scan import (
    rglru_inner_fn,
    rglru_inner_ref,
    rglru_scan_fn,
    rglru_scan_ref,
)

BATCH_SIZES = [2, 64]
SEQ_LENGTHS = [8, 16, 64, 128, 256, 512, 1024, 2048, 4096]
HID_DIMS = [8, 64]
STATE_DIMS = [1]

RTOL = 3e-3
ATOL = 5e-3
RTOLW = 1e-3
ATOLW = 1e-3


def init_scan_params(
    batch_size: int,
    hid_dim: int,
    state_dim: int,
    seq_len: int,
    device="cuda",
):
    torch.manual_seed(42)

    u = torch.randn(
        batch_size, hid_dim, seq_len, dtype=torch.float32, device=device
    )
    # Mirror real usage: delta = c * sigmoid(randn), so it lives in (0, c)
    # and stays away from 0 where the backward has legitimate numerical
    # sensitivity. c = 8 is the default constant in RGLRUInnerFn.
    c = 8.0
    delta = c * torch.sigmoid(
        torch.randn(
            batch_size, hid_dim, seq_len, dtype=torch.float32, device=device
        )
    )
    # a in (0, 1) - initialise so a^c lands in [0.9, 0.999] range
    A = 0.9 + 0.099 * torch.rand(
        hid_dim, state_dim, dtype=torch.float32, device=device
    )

    return u, delta, A


def init_inner_params(
    batch_size: int,
    hid_dim: int,
    state_dim: int,
    seq_len: int,
    d_model: int = None,
    d_conv: int = 4,
    device="cuda",
):
    torch.manual_seed(42)

    if d_model is None:
        d_model = hid_dim

    x = torch.randn(
        batch_size, hid_dim, seq_len, dtype=torch.float32, device=device
    )
    conv1d_weight = (
        torch.randn(hid_dim, 1, d_conv, dtype=torch.float32, device=device)
        * 0.1
    )
    conv1d_bias = torch.zeros(hid_dim, dtype=torch.float32, device=device)
    a = 0.9 + 0.099 * torch.rand(
        hid_dim, state_dim, dtype=torch.float32, device=device
    )
    recurrent_gate_weight = (
        torch.randn(hid_dim, hid_dim, dtype=torch.float32, device=device)
        * 0.02
    )
    recurrent_gate_bias = torch.zeros(
        hid_dim, dtype=torch.float32, device=device
    )
    input_gate_weight = (
        torch.randn(hid_dim, hid_dim, dtype=torch.float32, device=device)
        * 0.02
    )
    input_gate_bias = torch.zeros(hid_dim, dtype=torch.float32, device=device)
    out_proj_weight = (
        torch.randn(d_model, hid_dim, dtype=torch.float32, device=device)
        * 0.02
    )
    out_proj_bias = torch.zeros(d_model, dtype=torch.float32, device=device)
    gate = torch.randn(
        batch_size, seq_len, hid_dim, dtype=torch.float32, device=device
    )

    return (
        x,
        conv1d_weight,
        conv1d_bias,
        a,
        recurrent_gate_weight,
        recurrent_gate_bias,
        input_gate_weight,
        input_gate_bias,
        out_proj_weight,
        out_proj_bias,
        gate,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_scan_forward_consistency(batch_size, seq_len, hid_dim, state_dim):
    torch.manual_seed(42)
    device = torch.device("cuda")

    u, delta, A = init_scan_params(
        batch_size, hid_dim, state_dim, seq_len, device=device
    )

    with torch.no_grad():
        y_cuda = rglru_scan_fn(u, delta, A)
        y_ref = rglru_scan_ref(u, delta, A)

    assert torch.allclose(y_cuda, y_ref, rtol=RTOL, atol=ATOL), (
        f"Scan forward mismatch "
        f"(B={batch_size}, L={seq_len}, H={hid_dim}, N={state_dim}): "
        f"max_diff={(y_cuda - y_ref).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_scan_gradient_consistency(batch_size, seq_len, hid_dim, state_dim):
    torch.manual_seed(42)
    device = torch.device("cuda")

    u_init, delta_init, A_init = init_scan_params(
        batch_size, hid_dim, state_dim, seq_len, device=device
    )

    u_cuda = u_init.clone().requires_grad_(True)
    delta_cuda = delta_init.clone().requires_grad_(True)
    A_cuda = A_init.clone().requires_grad_(True)

    u_ref = u_init.clone().requires_grad_(True)
    delta_ref = delta_init.clone().requires_grad_(True)
    A_ref = A_init.clone().requires_grad_(True)

    y_cuda = rglru_scan_fn(u_cuda, delta_cuda, A_cuda)
    y_ref = rglru_scan_ref(u_ref, delta_ref, A_ref)

    g = torch.randn_like(y_cuda)
    y_cuda.backward(g)
    y_ref.backward(g)

    # dA accumulates across batchxseq via atomicAdd: expected float32
    # drift grows with sqrt(accumulation count), so scale atol accordingly.
    params = [
        ("u", u_cuda, u_ref, RTOL * 2, ATOL * 5),
        ("delta", delta_cuda, delta_ref, RTOL * 5, ATOL * 10),
        ("A", A_cuda, A_ref, RTOLW * 5, ATOLW * 5),
    ]
    for name, cuda_p, ref_p, rtol, atol in params:
        assert cuda_p.grad is not None, f"CUDA grad for {name} is None"
        assert ref_p.grad is not None, f"Ref grad for {name} is None"
        assert torch.allclose(cuda_p.grad, ref_p.grad, rtol=rtol, atol=atol), (
            f"Scan grad mismatch for {name} "
            f"(B={batch_size}, L={seq_len}, H={hid_dim}, N={state_dim}): "
            f"max_diff={(cuda_p.grad - ref_p.grad).abs().max().item():.6e}"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_inner_forward_consistency(batch_size, seq_len, hid_dim, state_dim):
    torch.manual_seed(42)
    device = torch.device("cuda")

    (
        x,
        conv_w,
        conv_b,
        a,
        rg_w,
        rg_b,
        ig_w,
        ig_b,
        out_w,
        out_b,
        gate,
    ) = init_inner_params(
        batch_size, hid_dim, state_dim, seq_len, device=device
    )

    with torch.no_grad():
        y_cuda = rglru_inner_fn(
            x,
            conv_w,
            conv_b,
            a,
            rg_w,
            rg_b,
            ig_w,
            ig_b,
            out_w,
            out_b,
            gate,
        )
        y_ref = rglru_inner_ref(
            x,
            conv_w,
            conv_b,
            a,
            rg_w,
            rg_b,
            ig_w,
            ig_b,
            out_w,
            out_b,
            gate,
        )

    assert torch.allclose(y_cuda, y_ref, rtol=RTOL, atol=ATOL), (
        f"Inner forward mismatch "
        f"(B={batch_size}, L={seq_len}, H={hid_dim}, N={state_dim}): "
        f"max_diff={(y_cuda - y_ref).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("hid_dim", HID_DIMS)
@pytest.mark.parametrize("state_dim", STATE_DIMS)
def test_inner_gradient_consistency(batch_size, seq_len, hid_dim, state_dim):
    torch.manual_seed(42)
    device = torch.device("cuda")

    (
        x_init,
        conv_w_init,
        conv_b_init,
        a_init,
        rg_w_init,
        rg_b_init,
        ig_w_init,
        ig_b_init,
        out_w_init,
        out_b_init,
        gate_init,
    ) = init_inner_params(
        batch_size, hid_dim, state_dim, seq_len, device=device
    )

    # CUDA path
    x_cuda = x_init.clone().requires_grad_(True)
    conv_w_cuda = conv_w_init.clone().requires_grad_(True)
    conv_b_cuda = conv_b_init.clone().requires_grad_(True)
    a_cuda = a_init.clone().requires_grad_(True)
    rg_w_cuda = rg_w_init.clone().requires_grad_(True)
    rg_b_cuda = rg_b_init.clone().requires_grad_(True)
    ig_w_cuda = ig_w_init.clone().requires_grad_(True)
    ig_b_cuda = ig_b_init.clone().requires_grad_(True)
    out_w_cuda = out_w_init.clone().requires_grad_(True)
    out_b_cuda = out_b_init.clone().requires_grad_(True)
    gate_cuda = gate_init.clone().requires_grad_(True)

    # Ref path
    x_ref = x_init.clone().requires_grad_(True)
    conv_w_ref = conv_w_init.clone().requires_grad_(True)
    conv_b_ref = conv_b_init.clone().requires_grad_(True)
    a_ref = a_init.clone().requires_grad_(True)
    rg_w_ref = rg_w_init.clone().requires_grad_(True)
    rg_b_ref = rg_b_init.clone().requires_grad_(True)
    ig_w_ref = ig_w_init.clone().requires_grad_(True)
    ig_b_ref = ig_b_init.clone().requires_grad_(True)
    out_w_ref = out_w_init.clone().requires_grad_(True)
    out_b_ref = out_b_init.clone().requires_grad_(True)
    gate_ref = gate_init.clone().requires_grad_(True)

    y_cuda = rglru_inner_fn(
        x_cuda,
        conv_w_cuda,
        conv_b_cuda,
        a_cuda,
        rg_w_cuda,
        rg_b_cuda,
        ig_w_cuda,
        ig_b_cuda,
        out_w_cuda,
        out_b_cuda,
        gate_cuda,
    )
    y_ref = rglru_inner_ref(
        x_ref,
        conv_w_ref,
        conv_b_ref,
        a_ref,
        rg_w_ref,
        rg_b_ref,
        ig_w_ref,
        ig_b_ref,
        out_w_ref,
        out_b_ref,
        gate_ref,
    )

    g = torch.randn_like(y_cuda)
    y_cuda.backward(g)
    y_ref.backward(g.clone())

    a_atol = max(ATOLW * 5, ATOLW * math.sqrt(batch_size * seq_len))
    params = [
        ("x", x_cuda, x_ref, RTOL * 2, ATOL * 5),
        ("conv1d_weight", conv_w_cuda, conv_w_ref, RTOLW * 5, ATOLW * 5),
        ("conv1d_bias", conv_b_cuda, conv_b_ref, RTOLW * 5, ATOLW * 5),
        ("a", a_cuda, a_ref, RTOLW * 5, a_atol),
        ("rg_weight", rg_w_cuda, rg_w_ref, RTOLW * 5, ATOLW * 5),
        ("rg_bias", rg_b_cuda, rg_b_ref, RTOLW * 5, ATOLW * 5),
        ("ig_weight", ig_w_cuda, ig_w_ref, RTOLW * 5, ATOLW * 5),
        ("ig_bias", ig_b_cuda, ig_b_ref, RTOLW * 5, ATOLW * 5),
        ("out_proj_weight", out_w_cuda, out_w_ref, RTOLW * 5, ATOLW * 5),
        ("out_proj_bias", out_b_cuda, out_b_ref, RTOLW * 5, ATOLW * 5),
        ("gate", gate_cuda, gate_ref, RTOL * 5, ATOL * 10),
    ]
    for name, cuda_p, ref_p, rtol, atol in params:
        assert cuda_p.grad is not None, f"CUDA grad for {name} is None"
        assert ref_p.grad is not None, f"Ref grad for {name} is None"
        assert torch.allclose(cuda_p.grad, ref_p.grad, rtol=rtol, atol=atol), (
            f"Inner grad mismatch for {name} "
            f"(B={batch_size}, L={seq_len}, H={hid_dim}, N={state_dim}): "
            f"max_diff={(cuda_p.grad - ref_p.grad).abs().max().item():.6e}"
        )
