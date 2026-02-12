import pytest
import torch

from lrnnx.ops.triton.simplified_state_update import (
    simplified_state_update,
    simplified_state_update_ref,
)

RTOL = 3e-3
ATOL = 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("discretization", ["bilinear", "zoh", "dirac"])
@pytest.mark.parametrize("conj_sym", [False, True])
@pytest.mark.parametrize("has_D", [False, True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize("dim", [256, 256 + 16, 512])
def test_simplified_state_update_matches_ref(
    dim, dstate, discretization, conj_sym, has_D
):
    device = "cuda"
    torch.manual_seed(0)
    batch, H, P = 4, dim, dstate

    state = torch.randn(batch, P, device=device, dtype=torch.complex64)
    x = torch.randn(batch, H, device=device, dtype=torch.float32)
    dt = torch.rand(P, device=device, dtype=torch.float32) * 0.1

    A = torch.complex(
        -torch.rand(P, device=device, dtype=torch.float32),
        0.1 * torch.randn(P, device=device, dtype=torch.float32),
    )
    B = torch.complex(
        torch.randn(P, H, device=device, dtype=torch.float32),
        0.1 * torch.randn(P, H, device=device, dtype=torch.float32),
    )
    C = torch.complex(
        torch.randn(H, P, device=device, dtype=torch.float32),
        0.1 * torch.randn(H, P, device=device, dtype=torch.float32),
    )
    D = (
        torch.randn(H, H, device=device, dtype=torch.float32) * 0.1
        if has_D
        else None
    )
    deltaA = None

    state_ref = state.detach().clone()

    out = simplified_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        deltaA=deltaA,
        discretization=discretization,
        conj_sym=conj_sym,
    )
    out_ref = simplified_state_update_ref(
        state_ref,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        deltaA=deltaA,
        discretization=discretization,
        conj_sym=conj_sym,
    )

    assert torch.allclose(state, state_ref, rtol=RTOL, atol=ATOL)
    assert torch.allclose(out, out_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("discretization", ["bilinear", "zoh", "dirac"])
@pytest.mark.parametrize("conj_sym", [False, True])
@pytest.mark.parametrize("has_deltaA", [False, True])
@pytest.mark.parametrize("has_D", [False, True])
@pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize("dim", [256, 256 + 16, 512])
def test_simplified_state_update_event_deltaA(
    dim, dstate, discretization, conj_sym, has_deltaA, has_D
):
    device = "cuda"
    torch.manual_seed(0)

    batch, H, P = 4, dim, dstate
    state = torch.randn(batch, P, device=device, dtype=torch.complex64)
    x = torch.randn(batch, H, device=device, dtype=torch.float32)
    dt = torch.rand(batch, P, device=device, dtype=torch.float32) * 0.1

    A = torch.complex(
        -torch.rand(P, device=device, dtype=torch.float32),
        0.1 * torch.randn(P, device=device, dtype=torch.float32),
    )
    B = torch.complex(
        torch.randn(P, H, device=device, dtype=torch.float32),
        0.1 * torch.randn(P, H, device=device, dtype=torch.float32),
    )
    C = torch.complex(
        torch.randn(H, P, device=device, dtype=torch.float32),
        0.1 * torch.randn(H, P, device=device, dtype=torch.float32),
    )
    D = (
        torch.randn(H, H, device=device, dtype=torch.float32) * 0.1
        if has_D
        else None
    )
    deltaA = (
        torch.rand(batch, P, device=device, dtype=torch.float32) * 0.1
        if has_deltaA
        else None
    )

    state_ref = state.detach().clone()

    out = simplified_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        deltaA=deltaA,
        discretization=discretization,
        conj_sym=conj_sym,
    )
    out_ref = simplified_state_update_ref(
        state_ref,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        deltaA=deltaA,
        discretization=discretization,
        conj_sym=conj_sym,
    )

    assert torch.allclose(state, state_ref, rtol=RTOL, atol=ATOL)
    assert torch.allclose(out, out_ref, rtol=RTOL, atol=ATOL)
