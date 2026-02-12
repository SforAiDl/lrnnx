"""
Unit tests for S7 (Selective and Simplified State Space Layers).
"""

import pytest
import torch

from lrnnx.models.ltv import S7

RTOL = 3e-3
ATOL = 5e-3

SEQ_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("use_fast_path", [True, False])
def test_s7_step_consistency(device, seq_len, use_fast_path):
    """Test that step-by-step inference matches parallel forward pass."""
    if use_fast_path and not torch.cuda.is_available():
        pytest.skip("CUDA not available for fast path")

    batch_size = 32
    d_model = 32
    d_state = 16

    model = S7(
        d_model=d_model, d_state=d_state, use_fast_path=use_fast_path
    ).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # Parallel forward pass
    with torch.no_grad():
        y_parallel = model(x)

    # Step-by-step inference
    cache = model.allocate_inference_cache(batch_size, max_seqlen=seq_len)
    y_steps = []

    with torch.no_grad():
        for t in range(seq_len):
            y_t, cache = model.step(x[:, t : t + 1, :], cache)
            y_steps.append(y_t.squeeze(1))

    y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch (L={seq_len}, fast={use_fast_path}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_s7_cuda_vs_ref_forward(device, seq_len):
    """Test CUDA model matches reference model."""
    batch_size = 32
    d_model = 32
    d_state = 16

    model_cuda = S7(d_model=d_model, d_state=d_state, use_fast_path=True).to(
        device
    )
    model_ref = S7(d_model=d_model, d_state=d_state, use_fast_path=False).to(
        device
    )
    model_ref.load_state_dict(model_cuda.state_dict())
    model_cuda.eval()
    model_ref.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        y_cuda = model_cuda(x)
        y_ref = model_ref(x)

    assert torch.allclose(y_cuda, y_ref, rtol=RTOL, atol=ATOL), (
        f"CUDA/Ref mismatch (L={seq_len}): "
        f"max_diff={(y_cuda - y_ref).abs().max().item():.6e}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA kernel not available"
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_s7_gradient_consistency(device, seq_len):
    """Test CUDA and reference produce consistent gradients."""
    batch_size = 2
    d_model = 32
    d_state = 16

    model_cuda = S7(d_model=d_model, d_state=d_state, use_fast_path=True).to(
        device
    )
    model_ref = S7(d_model=d_model, d_state=d_state, use_fast_path=False).to(
        device
    )
    model_ref.load_state_dict(model_cuda.state_dict())

    model_cuda.requires_grad_(True)
    model_ref.requires_grad_(True)

    x_cuda = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_ref = x_cuda.clone().detach().requires_grad_(True)

    # Forward pass
    y_cuda = model_cuda(x_cuda)
    y_ref = model_ref(x_ref)

    target = torch.ones_like(y_cuda)

    # Compute MSE loss
    loss_cuda = torch.nn.functional.mse_loss(y_cuda, target)
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)

    # Backward pass
    loss_cuda.backward()
    loss_ref.backward()

    # Compare parameter gradients
    for (name_cuda, p_cuda), (name_ref, p_ref) in zip(
        model_cuda.named_parameters(), model_ref.named_parameters()
    ):
        assert name_cuda == name_ref
        if p_cuda.grad is None and p_ref.grad is None:
            continue
        assert p_cuda.grad is not None, f"CUDA grad for {name_cuda} is None"
        assert p_ref.grad is not None, f"Ref grad for {name_ref} is None"
        assert torch.allclose(p_cuda.grad, p_ref.grad, rtol=RTOL, atol=ATOL), (
            f"Grad mismatch for {name_cuda} (L={seq_len}): "
            f"max_diff={(p_cuda.grad - p_ref.grad).abs().max().item():.6e}"
        )

    # Compare input gradients
    assert x_cuda.grad is not None, "CUDA input gradient is None"
    assert x_ref.grad is not None, "Ref input gradient is None"

    assert torch.allclose(x_cuda.grad, x_ref.grad, rtol=RTOL, atol=ATOL), (
        f"Input grad mismatch (L={seq_len}): "
        f"max_diff={(x_cuda.grad - x_ref.grad).abs().max().item():.6e}"
    )
