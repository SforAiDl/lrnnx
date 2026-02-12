"""
Unit tests for S4D (Diagonal State Space) layer.
"""

import pytest
import torch

from lrnnx.models.lti import S4D

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
def test_s4d_step_consistency(device, seq_len):
    """Test that step-by-step inference matches parallel forward pass."""
    batch_size = 32
    d_model, d_state = 32, 32

    model = S4D(
        d_model=d_model,
        d_state=d_state,
        l_max=seq_len,
        channels=1,
        kernel="s4d",
        transposed=False,
    ).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # FFT/parallel forward pass
    with torch.no_grad():
        y_parallel, _ = model(x)

    # Step-by-step inference
    cache = model.allocate_inference_cache(batch_size, device=device)
    y_steps = []

    with torch.no_grad():
        for t in range(seq_len):
            y_t, cache = model.step(x[:, t, :], cache)
            y_steps.append(y_t)

    y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch (L={seq_len}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_s4d_fft_vs_recurrence(device, seq_len):
    """Test that FFT implementation matches manual recurrence."""
    batch_size = 32
    d_model, d_state = 32, 32

    model = S4D(
        d_model=d_model,
        d_state=d_state,
        l_max=seq_len,
        channels=1,
        kernel="s4d",
        transposed=False,
    ).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # FFT based implementation
    with torch.no_grad():
        y_fft, _ = model(x)

    # Manual recurrence implementation
    model.layer.kernel._setup_step()
    dA = model.layer.kernel.dA
    dB = model.layer.kernel.dB
    dC = model.layer.kernel.dC

    if dA.dim() == 3:
        H, N, _ = dA.shape
        use_matmul = True
    elif dA.dim() == 2:
        H, N = dA.shape
        use_matmul = False
    else:
        raise ValueError(f"Unexpected dA shape: {dA.shape}")

    x_internal = x.transpose(1, 2)

    y_kernel_batch = []
    for b in range(batch_size):
        state = torch.zeros(H, N, dtype=dA.dtype, device=device)
        rec_out = []
        for t in range(seq_len):
            u = x_internal[b, :, t]

            if use_matmul:
                state = torch.einsum(
                    "hmn,hn->hm", dA, state
                ) + dB * u.unsqueeze(-1)
            else:
                state = dA * state + dB * u.unsqueeze(-1)

            # S4D uses factor of 2 for real output
            y_t = torch.einsum("chn,hn->ch", dC, state)
            y_t = 2 * y_t.real

            rec_out.append(y_t.squeeze(0))
        y_kernel_batch.append(torch.stack(rec_out))

    y_kernel = torch.stack(y_kernel_batch)
    D = model.D

    y_kernel_expanded = y_kernel.unsqueeze(1).transpose(2, 3)
    x_expanded = x_internal.unsqueeze(1)

    y_with_D = y_kernel_expanded + x_expanded * D.unsqueeze(0).unsqueeze(-1)

    B_dim, C_dim, H_dim, L_dim = y_with_D.shape
    y_reshaped = y_with_D.reshape(B_dim, C_dim * H_dim, L_dim)
    y_before_act = y_reshaped.transpose(1, 2)

    y_after_layer_act = model.conv_activation(y_before_act)

    y_rec = model.mult_activation(y_after_layer_act)
    y_rec = model.drop(y_rec)
    y_rec = model.output_linear(y_rec)

    assert torch.allclose(y_fft, y_rec, rtol=RTOL, atol=ATOL), (
        f"FFT/Recurrence mismatch (L={seq_len}): "
        f"max_diff={(y_fft - y_rec).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_s4d_gradient_consistency(device, seq_len):
    """Test that FFT and recurrence implementations produce consistent gradients."""
    batch_size = 2
    d_model, d_state = 32, 32

    model_fft = S4D(
        d_model=d_model,
        d_state=d_state,
        l_max=seq_len,
        channels=1,
        kernel="s4d",
        transposed=False,
    ).to(device)
    model_rec = S4D(
        d_model=d_model,
        d_state=d_state,
        l_max=seq_len,
        channels=1,
        kernel="s4d",
        transposed=False,
    ).to(device)

    # Copy params to ensure same init
    with torch.no_grad():
        for (name_fft, param_fft), (name_rec, param_rec) in zip(
            model_fft.named_parameters(), model_rec.named_parameters()
        ):
            param_rec.copy_(param_fft)

    model_fft.requires_grad_(True)
    model_rec.requires_grad_(True)

    x = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_rec = x.clone().detach().requires_grad_(True)

    # FFT forward pass
    y_fft, _ = model_fft(x)

    # Manual recurrence forward pass
    model_rec.layer.kernel._setup_step()
    dA = model_rec.layer.kernel.dA
    dB = model_rec.layer.kernel.dB
    dC = model_rec.layer.kernel.dC

    if dA.dim() == 3:
        H, N, _ = dA.shape
        use_matmul = True
    elif dA.dim() == 2:
        H, N = dA.shape
        use_matmul = False
    else:
        raise ValueError(f"Unexpected dA shape: {dA.shape}")

    x_internal = x_rec.transpose(1, 2)

    y_kernel_batch = []
    for b in range(batch_size):
        state = torch.zeros(H, N, dtype=dA.dtype, device=x_rec.device)
        rec_out = []
        for t in range(seq_len):
            u = x_internal[b, :, t]

            if use_matmul:
                state = torch.einsum(
                    "hmn,hn->hm", dA, state
                ) + dB * u.unsqueeze(-1)
            else:
                state = dA * state + dB * u.unsqueeze(-1)

            y_t = torch.einsum("chn,hn->ch", dC, state)
            y_t = 2 * y_t.real

            rec_out.append(y_t.squeeze(0))
        y_kernel_batch.append(torch.stack(rec_out))

    y_kernel = torch.stack(y_kernel_batch)
    D = model_rec.D

    y_kernel_expanded = y_kernel.unsqueeze(1).transpose(2, 3)
    x_expanded = x_internal.unsqueeze(1)

    y_with_D = y_kernel_expanded + x_expanded * D.unsqueeze(0).unsqueeze(-1)

    B_dim, C_dim, H_dim, L_dim = y_with_D.shape
    y_reshaped = y_with_D.reshape(B_dim, C_dim * H_dim, L_dim)
    y_before_act = y_reshaped.transpose(1, 2)

    y_after_layer_act = model_rec.conv_activation(y_before_act)

    y_rec = model_rec.mult_activation(y_after_layer_act)
    y_rec = model_rec.drop(y_rec)
    y_rec = model_rec.output_linear(y_rec)

    # Create dummy target
    target = torch.ones_like(y_fft)

    # Compute MSE loss
    loss_fft = torch.nn.functional.mse_loss(y_fft, target)
    loss_rec = torch.nn.functional.mse_loss(y_rec, target)

    # Backward pass
    loss_fft.backward()
    loss_rec.backward()

    # Compare gradients for params (skip SSM kernel params that don't have grads from recurrence)
    for (name_fft, param_fft), (name_rec, param_rec) in zip(
        model_fft.named_parameters(), model_rec.named_parameters()
    ):
        grad_fft = param_fft.grad
        grad_rec = param_rec.grad

        # Skip if gradient is None in recurrence path
        if grad_rec is None:
            continue

        assert grad_fft is not None, f"FFT gradient for {name_fft} is None"

        assert torch.allclose(grad_fft, grad_rec, rtol=RTOL, atol=ATOL), (
            f"Gradients for {name_fft} don't match (L={seq_len}): "
            f"max_diff={(grad_fft - grad_rec).abs().max().item():.6e}"
        )

    # Compare input gradients
    assert x.grad is not None, "FFT input gradient is None"
    assert x_rec.grad is not None, "Recurrence input gradient is None"

    assert torch.allclose(x.grad, x_rec.grad, rtol=RTOL, atol=ATOL), (
        f"Input gradients don't match (L={seq_len}): "
        f"max_diff={(x.grad - x_rec.grad).abs().max().item():.6e}"
    )
