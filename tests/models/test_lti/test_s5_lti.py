"""
Unit tests for S5 SSM.
"""

import pytest
import torch

from lrnnx.models.lti import S5

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
def test_s5_step_consistency(device, seq_len):
    """Test that step-by-step inference matches parallel forward pass."""
    batch_size = 32
    d_model, d_state = 32, 16

    model = S5(d_model=d_model, d_state=d_state, discretization="zoh").to(
        device
    )
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # FFT/parallel forward pass
    with torch.no_grad():
        y_parallel = model(x)

    # Step-by-step inference
    inference_cache = model.allocate_inference_cache(batch_size=batch_size)
    inference_cache = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inference_cache.items()
    }
    y_steps = []

    with torch.no_grad():
        for t in range(seq_len):
            y_t, inference_cache = model.step(x[:, t, :], inference_cache)
            y_steps.append(y_t)

    y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch (L={seq_len}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_s5_fft_vs_recurrence(device, seq_len):
    """Test that FFT implementation matches manual recurrence."""
    batch_size = 32
    d_model, d_state = 32, 16

    model = S5(d_model=d_model, d_state=d_state, discretization="zoh").to(
        device
    )
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # FFT based implementation
    with torch.no_grad():
        y_fft = model(x)

    # Discretize and get parameters
    A_bar, gamma_bar, C = model.discretize()
    A_bar = A_bar.to(device)
    C = C.to(device)

    if isinstance(gamma_bar, float):
        B_bar = gamma_bar * model.B.to(torch.complex64).to(device)
    else:
        B_bar = (gamma_bar.unsqueeze(-1) * model.B).to(device)

    # Recurrence based implementation
    y_rec_batch = []
    for b in range(batch_size):
        state = torch.zeros(
            model.d_state, dtype=torch.complex64, device=device
        )
        rec_out = []
        for u in x[b]:
            state = (A_bar * state) + (B_bar @ u.to(torch.complex64))
            rec_out.append((C @ state).real + u @ model.D.to(device))
        y_rec_batch.append(torch.stack(rec_out))
    y_rec = torch.stack(y_rec_batch)

    assert torch.allclose(y_fft, y_rec, rtol=RTOL, atol=ATOL), (
        f"FFT/Recurrence mismatch (L={seq_len}): "
        f"max_diff={(y_fft - y_rec).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_s5_gradient_consistency(device, seq_len):
    """Test that FFT and recurrence implementations produce consistent gradients."""
    batch_size = 2
    d_model, d_state = 32, 16

    model_fft = S5(d_model=d_model, d_state=d_state, discretization="zoh").to(
        device
    )
    model_rec = S5(d_model=d_model, d_state=d_state, discretization="zoh").to(
        device
    )

    # Copy params to ensure same init
    with torch.no_grad():
        model_rec.A.copy_(model_fft.A)
        model_rec.B.copy_(model_fft.B)
        model_rec.log_dt.copy_(model_fft.log_dt)
        model_rec.C.copy_(model_fft.C)
        model_rec.D.copy_(model_fft.D)

    model_fft.requires_grad_(True)
    model_rec.requires_grad_(True)

    x = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_rec = x.clone().detach().requires_grad_(True)

    # FFT forward pass
    y_fft = model_fft(x)

    # Recurrence forward pass
    A_bar, gamma_bar, C = model_rec.discretize()
    if isinstance(gamma_bar, float):
        B_bar = gamma_bar * model_rec.B.to(torch.complex64)
    else:
        B_bar = gamma_bar.unsqueeze(-1) * model_rec.B

    y_rec_batch = []
    for b in range(batch_size):
        state = torch.zeros(
            model_rec.d_state, dtype=torch.complex64, device=device
        )
        rec_out = []
        for u in x_rec[b]:
            state = (A_bar * state) + (B_bar @ u.to(torch.complex64))
            rec_out.append((C @ state).real + u @ model_rec.D)
        y_rec_batch.append(torch.stack(rec_out))
    y_rec = torch.stack(y_rec_batch)

    # Create dummy target
    target = torch.ones_like(y_fft)

    # Compute MSE loss
    loss_fft = torch.nn.functional.mse_loss(y_fft, target)
    loss_rec = torch.nn.functional.mse_loss(y_rec, target)

    # Backward pass
    loss_fft.backward()
    loss_rec.backward()

    params = ["A", "B", "log_dt", "C", "D"]

    # Compare grads for all params
    for p in params:
        grad_fft = getattr(model_fft, p).grad
        grad_rec = getattr(model_rec, p).grad

        assert grad_fft is not None, f"FFT gradient for {p} is None"
        assert grad_rec is not None, f"Recurrence gradient for {p} is None"

        assert torch.allclose(grad_fft, grad_rec, rtol=RTOL, atol=ATOL), (
            f"Gradients for {p} don't match (L={seq_len}): "
            f"max_diff={(grad_fft - grad_rec).abs().max().item():.6e}"
        )

    # Compare input gradients
    assert x.grad is not None, "FFT input gradient is None"
    assert x_rec.grad is not None, "Recurrence input gradient is None"

    assert torch.allclose(x.grad, x_rec.grad, rtol=RTOL, atol=ATOL), (
        f"Input gradients don't match (L={seq_len}): "
        f"max_diff={(x.grad - x_rec.grad).abs().max().item():.6e}"
    )
