import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from lrnnx.ops.selective_scan import selective_scan_fn, selective_scan_ref


@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize("seqlen", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize("has_D", [True])
@pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("is_variable_B", [False, True])
@pytest.mark.parametrize("is_variable_C", [False, True])
@pytest.mark.parametrize(
    "discretization", ["mamba", "bilinear", "zoh", "dirac"]
)
def test_selective_scan_with_deltaA(
    is_variable_B,
    is_variable_C,
    delta_softplus,
    has_z,
    has_D,
    return_last_state,
    seqlen,
    itype,
    discretization,
):
    device = "cuda"
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:
        rtol, atol = (rtol * 2, atol * 2)

    torch.random.manual_seed(0)

    batch_size = 2
    dim = 128
    dstate = 64
    wtype = torch.float32

    A = (
        -repeat(
            torch.arange(1, dstate + 1, dtype=wtype, device=device),
            "n -> d n",
            d=dim,
        )
    ).requires_grad_()

    if not is_variable_B:
        B_shape = (dim, dstate)
    else:
        B_shape = (batch_size, 1, dstate, seqlen)
    B = torch.randn(
        *B_shape,
        device=device,
        dtype=wtype if not is_variable_B else itype,
        requires_grad=True,
    )

    if not is_variable_C:
        C_shape = (dim, dstate)
    else:
        C_shape = (batch_size, 1, dstate, seqlen)
    C = torch.randn(
        *C_shape,
        device=device,
        dtype=wtype if not is_variable_C else itype,
        requires_grad=True,
    )

    if has_D:
        D = torch.randn(
            dim, device=device, dtype=torch.float32, requires_grad=True
        )
    else:
        D = None

    if has_z:
        z = torch.randn(
            batch_size,
            dim,
            seqlen,
            device=device,
            dtype=itype,
            requires_grad=True,
        )
    else:
        z = None

    u = torch.randn(
        batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True
    )
    delta = (
        0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
    ).requires_grad_()

    deltaA = (
        0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
    ).requires_grad_()

    delta_bias = torch.rand(dim, device=device) - 4.0
    delta_bias.requires_grad_()

    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    deltaA_ref = deltaA.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_()

    out, *rest = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        deltaA=deltaA,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
        discretization=discretization,
    )
    if return_last_state:
        last_state = rest[0]

    out_ref, *rest = selective_scan_ref(
        u_ref,
        delta_ref,
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        z=z_ref,
        delta_bias=delta_bias_ref,
        deltaA=deltaA_ref,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
        discretization=discretization,
    )
    if return_last_state:
        last_state_ref = rest[0]

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(
        out, out_ref, rtol=rtol, atol=atol
    ), f"Forward pass outputs don't match: max diff = {(out - out_ref).abs().max().item()}"
    if return_last_state:
        assert torch.allclose(
            last_state, last_state_ref, rtol=rtol, atol=atol
        ), f"Forward pass last states don't match: max diff = {(last_state - last_state_ref).abs().max().item()}"

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f"du max diff: {(u.grad - u_ref.grad).abs().max().item()}")

    # for DIRAC discretization with deltaA, delta may not have a gradient in the reference
    if delta_ref.grad is not None:
        print(
            f"ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}"
        )
    else:
        print(
            f"ddelta max (ref grad is None): {delta.grad.abs().max().item()}"
        )

    print(
        f"ddeltaA max diff: {(deltaA.grad - deltaA_ref.grad).abs().max().item()}"
    )
    print(f"dA max diff: {(A.grad - A_ref.grad).abs().max().item()}")
    print(f"dB max diff: {(B.grad - B_ref.grad).abs().max().item()}")
    print(f"dC max diff: {(C.grad - C_ref.grad).abs().max().item()}")
    if has_D:
        print(f"dD max diff: {(D.grad - D_ref.grad).abs().max().item()}")
    if has_z:
        print(f"dz max diff: {(z.grad - z_ref.grad).abs().max().item()}")

    assert torch.allclose(
        u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2
    )

    if delta_ref.grad is not None:
        assert torch.allclose(
            delta.grad,
            delta_ref.grad.to(dtype=itype),
            rtol=rtol * 5,
            atol=atol * 10,
        )
    else:
        assert (
            delta.grad.abs().max().item() < atol * 10
        ), f"Expected delta.grad to be ~0 when reference is None, got max {delta.grad.abs().max().item()}"

    assert torch.allclose(
        deltaA.grad,
        deltaA_ref.grad.to(dtype=itype),
        rtol=rtol * 5,
        atol=atol * 10,
    )
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(
        B.grad,
        B_ref.grad,
        rtol=rtolw if not is_variable_B else rtol,
        atol=atolw if not is_variable_B else atol,
    )
    assert torch.allclose(
        C.grad,
        C_ref.grad,
        rtol=rtolw if not is_variable_C else rtol,
        atol=atolw if not is_variable_C else atol,
    )
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_z:
        assert torch.allclose(
            z.grad, z_ref.grad.to(dtype=itype), rtol=rtol, atol=atol
        )

        if delta_bias_ref.grad is not None:
            assert torch.allclose(
                delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw
            )
        else:
            assert (
                delta_bias.grad.abs().max().item() < atolw * 10
            ), f"Expected delta_bias.grad to be ~0 when reference is None, got max {delta_bias.grad.abs().max().item()}"


@pytest.mark.parametrize(
    "discretization", ["mamba", "zoh", "bilinear", "dirac"]
)
@pytest.mark.parametrize("itype", [torch.float32])
@pytest.mark.parametrize(
    "seqlen", [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
)
@pytest.mark.parametrize("return_last_state", [True])
@pytest.mark.parametrize("has_D", [True])
def test_selective_scan_complex_with_deltaA(
    has_D,
    return_last_state,
    seqlen,
    itype,
    discretization,
):
    device = "cuda"
    rtol, atol = (6e-4, 2e-3)
    rtolw, atolw = (1e-3, 1e-3)

    torch.random.manual_seed(42)

    batch_size = 2
    dim = 4
    dstate = 8
    wtype = torch.complex64

    A_real = (
        -repeat(
            torch.arange(1, dstate + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=dim,
        )
    ).contiguous()
    A_imag = repeat(
        torch.randn(dstate, dtype=torch.float32, device=device),
        "n -> d n",
        d=dim,
    ).contiguous()
    A = torch.complex(A_real, A_imag).requires_grad_()

    B_real = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    B_imag = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    B = torch.complex(B_real, B_imag).requires_grad_()

    C_real = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    C_imag = torch.randn(dim, dstate, device=device, dtype=torch.float32)
    C = torch.complex(C_real, C_imag).requires_grad_()

    if has_D:
        D = torch.randn(
            dim, device=device, dtype=torch.float32, requires_grad=True
        )
    else:
        D = None

    u = torch.randn(
        batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True
    )

    delta = (
        0.01 * torch.ones(batch_size, dim, seqlen, device=device, dtype=itype)
    ).requires_grad_()

    deltaA = (
        0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)
    ).requires_grad_()

    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    deltaA_ref = deltaA.detach().clone().requires_grad_()

    out, *rest = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D,
        z=None,
        delta_bias=None,
        deltaA=deltaA,
        delta_softplus=True,
        return_last_state=return_last_state,
        discretization=discretization,
    )
    if return_last_state:
        state = rest[0]

    out_ref, *rest = selective_scan_ref(
        u_ref,
        delta_ref,
        A_ref,
        B_ref,
        C_ref,
        D_ref,
        z=None,
        delta_bias=None,
        deltaA=deltaA_ref,
        delta_softplus=True,
        return_last_state=return_last_state,
        discretization=discretization,
    )
    if return_last_state:
        state_ref = rest[0]

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(
        out, out_ref, rtol=rtol, atol=atol
    ), f"Forward pass outputs don't match: max diff = {(out - out_ref).abs().max().item()}"

    if return_last_state:
        print(f"State max diff: {(state - state_ref).abs().max().item()}")
        assert torch.allclose(
            state, state_ref, rtol=rtol, atol=atol
        ), f"Forward pass last states don't match: max diff = {(state - state_ref).abs().max().item()}"

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f"du max diff: {(u.grad - u_ref.grad).abs().max().item()}")

    if delta_ref.grad is not None:
        print(
            f"ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}"
        )
    else:
        print(
            f"ddelta max (ref grad is None): {delta.grad.abs().max().item()}"
        )

    print(
        f"ddeltaA max diff: {(deltaA.grad - deltaA_ref.grad).abs().max().item()}"
    )
    print(f"dA max diff: {(A.grad - A_ref.grad).abs().max().item()}")
    print(f"dB max diff: {(B.grad - B_ref.grad).abs().max().item()}")
    print(f"dC max diff: {(C.grad - C_ref.grad).abs().max().item()}")
    if has_D:
        print(f"dD max diff: {(D.grad - D_ref.grad).abs().max().item()}")

    assert torch.allclose(
        u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2
    ), f"Gradient w.r.t. u doesn't match: max diff = {(u.grad - u_ref.grad).abs().max().item()}"

    if delta_ref.grad is not None:
        assert torch.allclose(
            delta.grad,
            delta_ref.grad.to(dtype=itype),
            rtol=rtol * 5,
            atol=atol * 10,
        ), f"Gradient w.r.t. delta doesn't match: max diff = {(delta.grad - delta_ref.grad).abs().max().item()}"
    else:
        assert (
            delta.grad.abs().max().item() < atol * 10
        ), f"Expected delta.grad to be ~0 when reference is None, got max {delta.grad.abs().max().item()}"

    assert torch.allclose(
        deltaA.grad,
        deltaA_ref.grad.to(dtype=itype),
        rtol=rtol * 5,
        atol=atol * 10,
    ), f"Gradient w.r.t. deltaA doesn't match: max diff = {(deltaA.grad - deltaA_ref.grad).abs().max().item()}"

    assert torch.allclose(
        A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5
    ), f"Gradient w.r.t. A doesn't match: max diff = {(A.grad - A_ref.grad).abs().max().item()}"
    assert torch.allclose(
        B.grad, B_ref.grad, rtol=rtolw, atol=atolw
    ), f"Gradient w.r.t. B doesn't match: max diff = {(B.grad - B_ref.grad).abs().max().item()}"
    assert torch.allclose(
        C.grad, C_ref.grad, rtol=rtolw, atol=atolw
    ), f"Gradient w.r.t. C doesn't match: max diff = {(C.grad - C_ref.grad).abs().max().item()}"
    if has_D:
        assert torch.allclose(
            D.grad, D_ref.grad, rtol=rtolw, atol=atolw
        ), f"Gradient w.r.t. D doesn't match: max diff = {(D.grad - D_ref.grad).abs().max().item()}"
