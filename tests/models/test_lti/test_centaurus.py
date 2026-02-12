"""
Unit tests for Centaurus SSM variants (neck, pointwise, dws, full).
"""

import pytest
import torch
from torch import einsum

from lrnnx.models.lti import (
    CentaurusDWS,
    CentaurusFull,
    CentaurusNeck,
    CentaurusPWNeck,
)

RTOL = 3e-3
ATOL = 5e-3

SEQ_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096]

# (constructor, d_model, d_state, sub_state_dim)
MODEL_CONFIGS = [
    pytest.param(CentaurusNeck, 16, 16, 2, id="neck"),
    pytest.param(CentaurusPWNeck, 8, 4, 2, id="pointwise"),
    pytest.param(CentaurusDWS, 8, 8, 2, id="dws"),
    pytest.param(CentaurusFull, 4, 16, 2, id="full"),
]


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


def _recurrence_forward(model, x):
    """Manual recurrence using model parameters, matching forward() semantics."""
    batch_size, seq_len, _ = x.shape
    device = x.device
    mode = getattr(model, "mode", "neck")

    delta = torch.exp(model.log_delta).to(device)
    A = model.A.to(device)

    if mode == "pointwise":
        N = model.d_state
        M = model.sub_state_dim
        delta_rep = delta.repeat_interleave(M)
        A_flat = A.reshape(-1)
        alpha = torch.exp(delta_rep.to(A_flat.dtype) * A_flat)

        B_bar = (delta_rep.unsqueeze(1) * model.B.to(device)).to(x.dtype)
        C_eff = model.C.to(x.dtype).to(device)

        y_rec_batch = []
        for b in range(batch_size):
            state = torch.zeros(N * M, dtype=A.dtype, device=device)
            rec_out = []
            for t in range(seq_len):
                s = einsum("sh,h->s", B_bar, x[b, t].to(B_bar.dtype))
                state = alpha * state + s.to(A.dtype)
                y_t = einsum("hs,s->h", C_eff, state.real)
                rec_out.append(y_t)
            y_rec_batch.append(torch.stack(rec_out))
        return torch.stack(y_rec_batch)

    # Non-pointwise modes: neck, dws, full
    N, M_dim = A.shape
    alpha = torch.exp(delta.to(A.dtype).unsqueeze(-1) * A)

    if mode == "neck":
        B_bar = (delta.unsqueeze(-1) * model.B.to(device)).to(x.dtype)
        C_eff = model.C.to(x.dtype).to(device)
    elif mode == "dws":
        B_bar = (delta.unsqueeze(-1) * torch.diag(model.B.to(device))).to(
            x.dtype
        )
        C_eff = torch.diag(model.C.to(device)).to(x.dtype)
    elif mode == "full":
        in_idx = model._full_in_index.to(device)
        out_idx = model._full_out_index.to(device)
        B_mat = model.B.new_zeros(N, model.d_model).to(device)
        B_mat.scatter_(1, in_idx.unsqueeze(1), model.B.to(device).unsqueeze(1))
        B_bar = (delta.unsqueeze(-1) * B_mat).to(x.dtype)
        C_mat = model.C.new_zeros(model.d_model, N).to(device)
        C_mat.scatter_(
            0, out_idx.unsqueeze(0), model.C.to(device).unsqueeze(0)
        )
        C_eff = C_mat.to(x.dtype)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    E = model.E.to(x.dtype).to(device)

    y_rec_batch = []
    for b in range(batch_size):
        state = torch.zeros(N, M_dim, dtype=A.dtype, device=device)
        rec_out = []
        for t in range(seq_len):
            s_n = einsum("nh,h->n", B_bar, x[b, t].to(B_bar.dtype))
            state = alpha * state + s_n.unsqueeze(-1).to(A.dtype)
            x_t = einsum("nm,nm->n", E, state.real)
            y_t = einsum("hn,n->h", C_eff, x_t)
            rec_out.append(y_t)
        y_rec_batch.append(torch.stack(rec_out))
    return torch.stack(y_rec_batch)


@pytest.mark.parametrize(
    "model_cls,d_model,d_state,sub_state_dim", MODEL_CONFIGS
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_centaurus_step_consistency(
    device, seq_len, model_cls, d_model, d_state, sub_state_dim
):
    """Test that step-by-step inference matches parallel forward pass."""
    batch_size = 2

    model = model_cls(
        d_model=d_model, d_state=d_state, sub_state_dim=sub_state_dim
    ).to(device)
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


@pytest.mark.parametrize(
    "model_cls,d_model,d_state,sub_state_dim", MODEL_CONFIGS
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_centaurus_fft_vs_recurrence(
    device, seq_len, model_cls, d_model, d_state, sub_state_dim
):
    """Test that FFT implementation matches manual recurrence."""
    batch_size = 2

    model = model_cls(
        d_model=d_model, d_state=d_state, sub_state_dim=sub_state_dim
    ).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    # FFT based implementation
    with torch.no_grad():
        y_fft = model(x)

    # Recurrence based implementation
    with torch.no_grad():
        y_rec = _recurrence_forward(model, x)

    assert torch.allclose(y_fft, y_rec, rtol=RTOL, atol=ATOL), (
        f"FFT/Recurrence mismatch (L={seq_len}): "
        f"max_diff={(y_fft - y_rec).abs().max().item():.6e}"
    )


@pytest.mark.parametrize(
    "model_cls,d_model,d_state,sub_state_dim", MODEL_CONFIGS
)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_centaurus_gradient_consistency(
    device, seq_len, model_cls, d_model, d_state, sub_state_dim
):
    """Test that FFT and recurrence implementations produce consistent gradients."""
    batch_size = 2

    model_fft = model_cls(
        d_model=d_model, d_state=d_state, sub_state_dim=sub_state_dim
    ).to(device)
    model_rec = model_cls(
        d_model=d_model, d_state=d_state, sub_state_dim=sub_state_dim
    ).to(device)

    # Copy params to ensure same init
    with torch.no_grad():
        for (n_fft, p_fft), (n_rec, p_rec) in zip(
            model_fft.named_parameters(), model_rec.named_parameters()
        ):
            p_rec.copy_(p_fft)

    model_fft.requires_grad_(True)
    model_rec.requires_grad_(True)

    x = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_rec = x.clone().detach().requires_grad_(True)

    # FFT forward pass
    y_fft = model_fft(x)

    # Recurrence forward pass
    y_rec = _recurrence_forward(model_rec, x_rec)

    # Create dummy target
    target = torch.ones_like(y_fft)

    # Compute MSE loss
    loss_fft = torch.nn.functional.mse_loss(y_fft, target)
    loss_rec = torch.nn.functional.mse_loss(y_rec, target)

    # Backward pass
    loss_fft.backward()
    loss_rec.backward()

    # Compare grads for all params
    for (name, p_fft), (_, p_rec) in zip(
        model_fft.named_parameters(), model_rec.named_parameters()
    ):
        grad_fft = p_fft.grad
        grad_rec = p_rec.grad

        assert grad_fft is not None, f"FFT gradient for {name} is None"
        assert grad_rec is not None, f"Recurrence gradient for {name} is None"

        assert torch.allclose(grad_fft, grad_rec, rtol=RTOL, atol=ATOL), (
            f"Gradients for {name} don't match (L={seq_len}): "
            f"max_diff={(grad_fft - grad_rec).abs().max().item():.6e}"
        )

    # Compare input gradients
    assert x.grad is not None, "FFT input gradient is None"
    assert x_rec.grad is not None, "Recurrence input gradient is None"

    assert torch.allclose(x.grad, x_rec.grad, rtol=RTOL, atol=ATOL), (
        f"Input gradients don't match (L={seq_len}): "
        f"max_diff={(x.grad - x_rec.grad).abs().max().item():.6e}"
    )
