"""
Unit tests for the LinOSS layer.
"""

import pytest
import torch

from lrnnx.models.lti import LinOSS

BATCH_SIZE = 2
D_MODEL = 8
D_STATE = 4

SEQ_LENGTHS = [16, 64, 128, 257, 1024]
DISCRETIZATIONS = ["im", "imex"]

RTOL = 3e-3
ATOL = 5e-3


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


def make_model(device, discretization):
    model = LinOSS(
        d_model=D_MODEL, d_state=D_STATE, discretization=discretization
    ).to(device)
    model.eval()
    return model


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_linoss_output_shape(device, discretization, seq_len):
    """The layer is shape preserving."""
    model = make_model(device, discretization)
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device)

    assert model(x).shape == x.shape


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_linoss_discretize_shapes(device, discretization):
    """``discretize`` returns a packed 2x2 block and a 2-vector multiplier."""
    model = make_model(device, discretization)

    A_bar, gamma_bar = model.discretize()

    assert A_bar.shape == (D_STATE, 4)
    assert gamma_bar.shape == (D_STATE, 2)


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
def test_linoss_step_consistency(device, discretization, seq_len):
    """Step-by-step inference matches the parallel forward pass."""
    model = make_model(device, discretization)
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device)

    with torch.no_grad():
        y_parallel = model(x)

        inference_cache = model.allocate_inference_cache(BATCH_SIZE, seq_len)
        y_steps = []
        for t in range(seq_len):
            y_t, inference_cache = model.step(x[:, t], inference_cache)
            y_steps.append(y_t)
        y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch ({discretization}, L={seq_len}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_linoss_batch_independence(device, discretization):
    """Batch elements are scanned independently."""
    model = make_model(device, discretization)
    x = torch.randn(BATCH_SIZE, 64, D_MODEL, device=device)

    with torch.no_grad():
        y_batched = model(x)
        y_single = torch.cat([model(x[i : i + 1]) for i in range(BATCH_SIZE)])

    torch.testing.assert_close(y_batched, y_single, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_linoss_is_causal(device, discretization):
    """Perturbing a timestep leaves every earlier output untouched."""
    model = make_model(device, discretization)
    x = torch.randn(BATCH_SIZE, 32, D_MODEL, device=device)
    x_perturbed = x.clone()
    x_perturbed[:, 16:] += 1.0

    with torch.no_grad():
        y = model(x)
        y_perturbed = model(x_perturbed)

    torch.testing.assert_close(
        y[:, :16], y_perturbed[:, :16], rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("seq_len", [16, 257])
def test_linoss_gradient_flow(device, discretization, seq_len):
    """Every parameter receives a finite gradient."""
    model = make_model(device, discretization)
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device)

    model(x).pow(2).mean().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(
            param.grad
        ).all(), f"{name} has a non-finite grad"
        assert param.grad.abs().sum() > 0, f"{name} has a zero gradient"


@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_linoss_gradcheck(discretization):
    """Double precision gradcheck of the layer against a numeric Jacobian."""
    model = make_model("cpu", discretization).double()
    x = torch.randn(1, 12, D_MODEL, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(model, (x,))


def test_linoss_rejects_bad_input_rank(device):
    """``forward`` needs ``(B, L, H)`` and ``step`` needs ``(B, H)``."""
    model = make_model(device, "im")

    with pytest.raises(ValueError):
        model(torch.randn(64, D_MODEL, device=device))

    with pytest.raises(ValueError):
        model.step(
            torch.randn(BATCH_SIZE, 1, D_MODEL, device=device),
            model.allocate_inference_cache(BATCH_SIZE),
        )


def test_linoss_has_no_convolution_kernel(device):
    """LinOSS is scan-only, so ``compute_kernel`` must refuse."""
    model = make_model(device, "im")

    with pytest.raises(NotImplementedError):
        model.compute_kernel()


def test_linoss_inference_cache_layout(device):
    """The cache uses the shared "lrnn_state" key and is stepped in place."""
    model = make_model(device, "im")
    inference_cache = model.allocate_inference_cache(BATCH_SIZE)
    state = inference_cache["lrnn_state"]

    assert state.shape == (BATCH_SIZE, D_STATE, 2)
    assert state.is_complex()
    assert not state.any()

    with torch.no_grad():
        model.step(
            torch.randn(BATCH_SIZE, D_MODEL, device=device), inference_cache
        )

    assert inference_cache["lrnn_state"] is state
    assert state.any()
