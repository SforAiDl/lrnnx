import pytest
import torch

from lrnnx.models.ltv import Mamba

RTOL = 3e-3
ATOL = 5e-3

SEQ_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096]
DISCRETIZATIONS = ["mamba", "zoh", "bilinear", "dirac"]


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [64])
@pytest.mark.parametrize("num_steps", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_mamba_multiple_steps(
    device, batch_size, d_model, num_steps, discretization
):
    """Test multiple autoregressive steps maintain correct state."""
    model = Mamba(d_model=d_model, discretization=discretization).to(device)
    cache = model.allocate_inference_cache(batch_size, max_seqlen=num_steps)

    for step in range(num_steps):
        x = torch.randn(batch_size, 1, d_model, device=device)
        y, cache = model.step(x, cache)

        assert y.shape == (
            batch_size,
            1,
            d_model,
        ), f"Step {step} output shape mismatch"
        assert torch.isfinite(
            y
        ).all(), f"Step {step} output contains non-finite values"
        assert (
            cache["seqlen_offset"] == step + 1
        ), f"Cache offset incorrect at step {step}"


@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_mamba_step_consistency(device, seq_len, discretization):
    """Test that step-by-step inference matches parallel forward pass."""
    batch_size = 16
    torch.manual_seed(42)

    d_model, d_state = 128, 64
    model = Mamba(
        d_model=d_model,
        d_state=d_state,
        discretization=discretization,
    ).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )

    with torch.no_grad():
        y_parallel = model(x)

    cache = model.allocate_inference_cache(batch_size, max_seqlen=seq_len)
    y_steps = []

    with torch.no_grad():
        for t in range(seq_len):
            y_t, cache = model.step(x[:, t : t + 1, :], cache)
            y_steps.append(y_t.squeeze(1))

    y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch (L={seq_len}, discretization={discretization}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_mamba_gradient_consistency(device, seq_len, discretization):
    """Test that CUDA fast path and reference implementation produce consistent gradients."""
    batch_size = 2
    torch.manual_seed(42)

    # create two models with same initialization
    d_model, d_state = 64, 16
    model_cuda = Mamba(
        d_model=d_model,
        d_state=d_state,
        discretization=discretization,
        use_fast_path=True,
    ).to(device)
    model_ref = Mamba(
        d_model=d_model,
        d_state=d_state,
        discretization=discretization,
        use_fast_path=False,
    ).to(device)

    # copy parameters to ensure same initialization
    model_ref.load_state_dict(model_cuda.state_dict())

    model_cuda.requires_grad_(True)
    model_ref.requires_grad_(True)

    x_cuda = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_ref = x_cuda.clone().detach().requires_grad_(True)

    # forward pass
    y_cuda = model_cuda(x_cuda)
    y_ref = model_ref(x_ref)

    target = torch.ones_like(y_cuda)

    # compute MSE loss
    loss_cuda = torch.nn.functional.mse_loss(y_cuda, target)
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)

    # backward pass
    loss_cuda.backward()
    loss_ref.backward()

    # compare parameter gradients (only for parameters that have gradients)
    for (name_cuda, param_cuda), (name_ref, param_ref) in zip(
        model_cuda.named_parameters(), model_ref.named_parameters()
    ):
        assert (
            name_cuda == name_ref
        ), f"Parameter name mismatch: {name_cuda} vs {name_ref}"

        # skip parameters that don't have gradients in either implementation
        if param_cuda.grad is None and param_ref.grad is None:
            continue

        assert (
            param_cuda.grad is not None
        ), f"CUDA gradient for {name_cuda} is None"
        assert (
            param_ref.grad is not None
        ), f"Reference gradient for {name_ref} is None"

        assert torch.allclose(
            param_cuda.grad, param_ref.grad, rtol=RTOL, atol=ATOL
        ), (
            f"Gradients for {name_cuda} don't match between CUDA and reference: "
            f"max_diff={(param_cuda.grad - param_ref.grad).abs().max().item():.6e}"
        )

    assert x_cuda.grad is not None, "CUDA input gradient is None"
    assert x_ref.grad is not None, "Reference input gradient is None"

    assert torch.allclose(x_cuda.grad, x_ref.grad, rtol=RTOL, atol=ATOL), (
        f"Input gradients don't match between CUDA and reference: "
        f"max_diff={(x_cuda.grad - x_ref.grad).abs().max().item():.6e}"
    )
