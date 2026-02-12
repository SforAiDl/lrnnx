"""
Unit tests for S5 with event-driven (integration_timesteps) mode.
"""

import pytest
import torch

from lrnnx.models.ltv import S5

RTOL = 3e-3
ATOL = 5e-3

SEQ_LENGTHS = [16, 64, 128, 256, 512, 1024, 2048, 4096]
DISCRETIZATIONS = ["zoh", "bilinear", "dirac"]


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
@pytest.mark.parametrize("use_fast_path", [True, False])
def test_s5_event_step_consistency(
    device, seq_len, discretization, use_fast_path
):
    """Test that step-by-step inference matches parallel forward pass with integration_timesteps."""
    batch_size = 32
    d_model, d_state = 32, 16
    timestep_scale = 0.1

    model = S5(
        d_model=d_model,
        d_state=d_state,
        discretization=discretization,
        use_fast_path=use_fast_path,
    ).to(device)
    model.eval()

    x = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )
    timesteps = (
        torch.rand(batch_size, seq_len, dtype=torch.float32, device=device)
        * timestep_scale
    )

    with torch.no_grad():
        y_parallel = model(x, integration_timesteps=timesteps)

    cache = model.allocate_inference_cache(batch_size, max_seqlen=seq_len)
    y_steps = []

    with torch.no_grad():
        for t in range(seq_len):
            y_t, cache = model.step(
                x[:, t : t + 1, :],
                cache,
                integration_timesteps=timesteps[:, t : t + 1],
            )
            y_steps.append(y_t.squeeze(1))

    y_sequential = torch.stack(y_steps, dim=1)

    assert torch.allclose(y_parallel, y_sequential, rtol=RTOL, atol=ATOL), (
        f"Step/Forward mismatch in event mode (L={seq_len}, discretization={discretization}, fast={use_fast_path}): "
        f"max_diff={(y_parallel - y_sequential).abs().max().item():.6e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@pytest.mark.parametrize("discretization", DISCRETIZATIONS)
def test_s5_event_gradient_consistency(device, seq_len, discretization):
    """Test that CUDA fast path and reference implementation produce consistent gradients in event mode."""
    batch_size = 2
    d_model, d_state = 32, 16
    timestep_scale = 0.1

    model_cuda = S5(
        d_model=d_model,
        d_state=d_state,
        discretization=discretization,
        use_fast_path=True,
    ).to(device)
    model_ref = S5(
        d_model=d_model,
        d_state=d_state,
        discretization=discretization,
        use_fast_path=False,
    ).to(device)

    model_ref.load_state_dict(model_cuda.state_dict())

    model_cuda.requires_grad_(True)
    model_ref.requires_grad_(True)

    x_cuda = torch.randn(
        batch_size, seq_len, d_model, device=device, requires_grad=True
    )
    x_ref = x_cuda.clone().detach().requires_grad_(True)
    timesteps = (
        torch.rand(batch_size, seq_len, dtype=torch.float32, device=device)
        * timestep_scale
    )

    y_cuda = model_cuda(x_cuda, integration_timesteps=timesteps)
    y_ref = model_ref(x_ref, integration_timesteps=timesteps)

    target = torch.ones_like(y_cuda)

    loss_cuda = torch.nn.functional.mse_loss(y_cuda, target)
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)

    loss_cuda.backward()
    loss_ref.backward()

    for (name_cuda, param_cuda), (name_ref, param_ref) in zip(
        model_cuda.named_parameters(), model_ref.named_parameters()
    ):
        assert (
            name_cuda == name_ref
        ), f"Parameter name mismatch: {name_cuda} vs {name_ref}"

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
            f"Gradients for {name_cuda} don't match in event mode (L={seq_len}): "
            f"max_diff={(param_cuda.grad - param_ref.grad).abs().max().item():.6e}"
        )

    assert x_cuda.grad is not None, "CUDA input gradient is None"
    assert x_ref.grad is not None, "Reference input gradient is None"

    assert torch.allclose(x_cuda.grad, x_ref.grad, rtol=RTOL, atol=ATOL), (
        f"Input gradients don't match in event mode (L={seq_len}): "
        f"max_diff={(x_cuda.grad - x_ref.grad).abs().max().item():.6e}"
    )
