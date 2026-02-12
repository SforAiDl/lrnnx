"""
Unit tests for Linear Recurrent Unit (LRU) layer.
"""

import pytest
import torch

from lrnnx.models.lti import LRU

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
def test_lru_step_consistency(device, seq_len):
    """Test that step-by-step inference matches parallel forward pass."""
    batch_size = 32
    d_model, d_state = 32, 16

    model = LRU(d_model, d_state).to(device)
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
def test_lru_fft_vs_recurrence(device, seq_len):
    """Test that FFT implementation matches manual recurrence."""
    batch_size = 32
    d_model, d_state = 32, 16

    model = LRU(d_model, d_state).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # FFT based implementation
    with torch.no_grad():
        y_fft = model(x)

    # Get LRU parameters
    Lambda = torch.exp(
        -torch.exp(model.nu_log) + 1j * torch.exp(model.theta_log)
    ).to(device)
    B_norm = (
        (model.B_re + 1j * model.B_im)
        * torch.exp(model.gamma_log).unsqueeze(-1)
    ).to(device)
    C = (model.C_re + 1j * model.C_im).to(device)
    D = model.D.to(device)

    # Recurrence based implementation
    y_rec_batch = []
    for b in range(batch_size):
        state = torch.zeros(
            model.d_state, dtype=torch.complex64, device=device
        )
        rec_out = []
        for u in x[b]:
            state = Lambda * state + (B_norm @ u.to(torch.complex64))
            rec_out.append((C @ state).real + D * u)
        y_rec_batch.append(torch.stack(rec_out))
    y_rec = torch.stack(y_rec_batch)

    assert torch.allclose(y_fft, y_rec, rtol=RTOL, atol=ATOL), (
        f"FFT/Recurrence mismatch (L={seq_len}): "
        f"max_diff={(y_fft - y_rec).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_lru_gradient_consistency(device, seq_len):
    """Test that FFT and recurrence implementations produce consistent gradients."""
    batch_size = 2
    d_model, d_state = 32, 16

    model_fft = LRU(d_model, d_state).to(device)
    model_rec = LRU(d_model, d_state).to(device)

    # Copy params to ensure same init
    with torch.no_grad():
        model_rec.nu_log.copy_(model_fft.nu_log)
        model_rec.theta_log.copy_(model_fft.theta_log)
        model_rec.B_re.copy_(model_fft.B_re)
        model_rec.B_im.copy_(model_fft.B_im)
        model_rec.C_re.copy_(model_fft.C_re)
        model_rec.C_im.copy_(model_fft.C_im)
        model_rec.D.copy_(model_fft.D)
        model_rec.gamma_log.copy_(model_fft.gamma_log)

    model_fft.requires_grad_(True)
    model_rec.requires_grad_(True)

    x = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_rec = x.clone().detach().requires_grad_(True)

    # FFT forward pass
    y_fft = model_fft(x)

    # Recurrence forward pass
    Lambda = torch.exp(
        -torch.exp(model_rec.nu_log) + 1j * torch.exp(model_rec.theta_log)
    )
    B_norm = (model_rec.B_re + 1j * model_rec.B_im) * torch.exp(
        model_rec.gamma_log
    ).unsqueeze(-1)
    C = model_rec.C_re + 1j * model_rec.C_im

    y_rec_batch = []
    for b in range(batch_size):
        state = torch.zeros(
            model_rec.d_state, dtype=torch.complex64, device=device
        )
        rec_out = []
        for u in x_rec[b]:
            state = Lambda * state + (B_norm @ u.to(torch.complex64))
            rec_out.append((C @ state).real + model_rec.D * u)
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

    params = [
        "nu_log",
        "theta_log",
        "B_re",
        "B_im",
        "C_re",
        "C_im",
        "D",
        "gamma_log",
    ]

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
