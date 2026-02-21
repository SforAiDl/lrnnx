"""
Simplified SSM Scan for S5-style models.
This module implements connects model using a CUDA kernel
"""

from __future__ import annotations

import torch

import simplified_scan_cuda
from lrnnx.ops.torch import custom_bwd, custom_fwd


class SimplifiedScanFn(torch.autograd.Function):
    """
    Autograd function for simplified SSM scan with complex input.

    B projects input from H-dim to P-dim (state dimension).
    The kernel operates in state space with identity B/C (diagonal SSM).
    C projects output from P-dim back to H-dim.

    Forward:  u (B,H,L) -> Bu = B @ u -> kernel -> x (B,P,L) -> y = C @ x -> (B,H,L)
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        deltaA: torch.Tensor | None,
        discretization: str,
        return_last_state=False,
    ) -> torch.Tensor:
        """
        Forward pass for the simplified SSM scan.

        Args:
            ctx (Any): Autograd context.
            u (torch.Tensor): Complex input tensor of shape ``(batch, H, seqlen)``. H is hidden/input dimension.
            delta (torch.Tensor): Real timestep tensor of shape ``(batch, P, seqlen)``. P is state dimension.
            A (torch.Tensor): Complex eigenvalues tensor of shape ``(P,)`` or ``(P, 1)``.
            B (torch.Tensor): Complex projection matrix of shape ``(P, H)``. Projects input to state space.
            C (torch.Tensor): Complex projection matrix of shape ``(H, P)``. Projects state to output.
            deltaA (torch.Tensor | None): Optional separate timestep for A discretization of shape ``(batch, P, seqlen)``.
            discretization (str): Discretization method ('bilinear', 'zoh', 'dirac').
            return_last_state (bool, optional): Whether to return the last hidden state. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Output tensor, and optionally the last state.
        """
        # Ensure contiguous
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if deltaA is not None and deltaA.stride(-1) != 1:
            deltaA = deltaA.contiguous()

        P = A.shape[0]  # State dimension

        # Ensure A is (P, 1) for the kernel
        if A.dim() == 1:
            A_kernel = A.unsqueeze(-1)
        else:
            A_kernel = A
        dstate = A_kernel.shape[1]  # Should be 1 for diagonal SSM

        # Modify input w/ projection (B, P, L)
        # this is needed because the kernel works in a SISO manner
        Bu = torch.einsum("ph,bhl->bpl", B, u)

        # Create identity B and C for the kernel (diagonal SSM)
        B_identity = torch.ones(P, dstate, dtype=A.dtype, device=A.device)
        C_identity = torch.ones(P, dstate, dtype=A.dtype, device=A.device)

        # Kernel forward pass
        # The naming is kept for consistency with the paper, the scan is not really
        # simplified, and works with complex inputs
        out, xs = simplified_scan_cuda.fwd(
            Bu,
            delta,
            A_kernel,
            B_identity,
            C_identity,
            deltaA,
            discretization,
        )

        # Get the actual output by projecting back with C
        y = torch.einsum("hp,bpl->bhl", C, out)

        # Save for backward
        ctx.discretization = discretization
        ctx.has_deltaA = deltaA is not None
        ctx.save_for_backward(
            u,
            delta,
            A_kernel,
            B,
            C,
            Bu,
            out,
            xs,
            B_identity,
            C_identity,
            deltaA if deltaA is not None else torch.empty(0, device=u.device),
        )

        last_state = (
            out[:, -1, 1::2] + 1j * out[:, -1, 0::2]
        )  # (batch, P, dstate)
        if return_last_state:
            return y, last_state.squeeze(-1)  # (batch, P)

        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, dout: torch.Tensor):
        """
        Backward pass for the simplified SSM scan.

        Args:
            ctx (Any): Autograd context.
            dout (torch.Tensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients with respect to inputs (du, ddelta, dA, dB, dC, ddeltaA, None, None).
        """
        (
            u,
            delta,
            A,
            B,
            C,
            Bu,
            out,
            xs,
            B_identity,
            C_identity,
            deltaA_saved,
        ) = ctx.saved_tensors

        # Restore deltaA (None if not provided)
        deltaA = deltaA_saved if ctx.has_deltaA else None

        # We assume the gradient is complex
        # in practice, it will not be.
        dx_out = torch.einsum("hp,bhl->bpl", C.conj(), dout)

        # dC = dout @ out^H summed over batch and seqlen
        dC = torch.einsum("bhl,bpl->hp", dout, out.conj())

        if dx_out.stride(-1) != 1:
            dx_out = dx_out.contiguous()

        # Backward through kernel (with identity B/C)
        result = simplified_scan_cuda.bwd(
            Bu,
            delta,
            A,
            B_identity,
            C_identity,
            deltaA,
            dx_out,
            xs,
            ctx.discretization,
        )

        # The CUDA function returns: dBu, ddelta, dA, dB_identity, dC_identity, [ddeltaA]
        # ddeltaA is present if deltaA was not None
        dBu, ddelta, dA = result[0], result[1], result[2]
        ddeltaA = None
        if ctx.has_deltaA and len(result) > 5:
            ddeltaA = result[5]

        # du = B^H @ dBu: (batch, H, seqlen)
        du = torch.einsum("ph,bpl->bhl", B.conj(), dBu)

        # dB = dBu @ u^H summed over batch and seqlen
        dB = torch.einsum("bpl,bhl->ph", dBu, u.conj())

        # Squeeze dA if it was originally 1D
        if dA.dim() == 2 and dA.shape[1] == 1:
            dA = dA.squeeze(-1)

        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            ddeltaA,
            None,
            None,
        )


def simplified_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    deltaA: torch.Tensor | None = None,
    return_last_state: bool = False,
    discretization: str = "bilinear",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified SSM scan using CUDA kernel.

    S5-style scan where B projects input to state space and C projects back.
    The kernel internally operates with identity B/C (diagonal SSM).

    Forward: u (B,H,L) -> Bu = B @ u -> kernel -> x (B,P,L) -> y = C @ x -> (B,H,L)

    Args:
        u (torch.Tensor): Complex input tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
        delta (torch.Tensor): Real timestep tensor of shape ``(batch, P, seqlen)``, dtype=float32.
        A (torch.Tensor): Complex state matrix eigenvalues of shape ``(P,)`` or ``(P, 1)``, dtype=complex64.
        B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype=complex64. Projects input to state.
        C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype=complex64. Projects state to output.
        deltaA (torch.Tensor | None, optional): Optional separate timestep for A discretization of shape ``(batch, P, seqlen)``, dtype=float32.
                If provided, A is discretized using deltaA while B uses delta. Defaults to None.
        return_last_state (bool, optional): Whether to return the last hidden state. Defaults to False.
        discretization (str, optional): Discretization method ('bilinear', 'zoh', 'dirac'). Defaults to "bilinear".

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]: 
            - Complex output tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
            - last_state : If return_last_state=True, returns state of shape ``(batch, P)``, dtype=complex64.
    """
    # Validate discretization method
    valid_methods = ("bilinear", "zoh", "dirac")
    if discretization not in valid_methods:
        raise ValueError(
            f"discretization must be one of {valid_methods}, got '{discretization}'"
        )

    # Use autograd function for gradient support
    return SimplifiedScanFn.apply(
        u, delta, A, B, C, deltaA, discretization, return_last_state
    )


def simplified_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    deltaA: torch.Tensor | None = None,
    return_last_state: bool = False,
    discretization: str = "bilinear",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of simplified scan (pure PyTorch).

    S5-style scan where B projects input to state space and C projects back.

    Forward: u (B,H,L) -> Bu = B @ u -> kernel -> x (B,P,L) -> y = C @ x -> (B,H,L)

    Args:
        u (torch.Tensor): Complex input tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
        delta (torch.Tensor): Real timestep tensor of shape ``(batch, P, seqlen)``, dtype=float32.
        A (torch.Tensor): Complex state matrix eigenvalues of shape ``(P,)`` or ``(P, 1)``, dtype=complex64.
        B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype=complex64. Projects input to state.
        C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype=complex64. Projects state to output.
        deltaA (torch.Tensor | None, optional): Optional separate timestep for A discretization of shape ``(batch, P, seqlen)``.
                If provided, A is discretized using deltaA while B uses delta. Defaults to None.
        return_last_state (bool, optional): Whether to return the last hidden state. Defaults to False.
        discretization (str, optional): Discretization method ('bilinear', 'zoh', 'dirac'). Defaults to "bilinear".

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            - out : Complex output tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
            - last_state : If return_last_state=True, also returns state of shape ``(batch, P)``, dtype=complex64.
    """
    dtype_in = u.dtype
    assert u.is_complex(), "Input u must be complex (complex64)"
    assert A.is_complex(), "State matrix A must be complex (complex64)"
    assert not delta.is_complex(), "Delta must be real (float32)"

    batch, H, seqlen = u.shape
    P = A.shape[0]  # State dimension

    # Ensure A is 1D for this reference
    if A.dim() == 2:
        A = A.squeeze(-1)

    # Ensure delta is float
    delta = delta.float()

    # Modify input w/ projection
    Bu = torch.einsum("ph,bhl->bpl", B, u)  # (batch, P, L)

    # Discretize A and B for each timestep
    Bu_expanded = Bu.unsqueeze(-1)  # (b, P, L, 1)
    delta_expanded = delta.unsqueeze(-1)  # (b, P, L, 1)
    # Use deltaA for A discretization if provided, otherwise use delta
    delta_for_A = (
        deltaA.unsqueeze(-1) if deltaA is not None else delta_expanded
    )
    A_expanded = (
        A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )  # (P,) -> (1, P, 1, 1)

    one = torch.tensor(1.0, dtype=A.dtype, device=A.device)

    if discretization == "bilinear":
        # A_bar = (1 + 0.5*delta_for_A*A) / (1 - 0.5*delta_for_A*A)
        # B_bar = delta / (1 - 0.5*delta*A) (identity B inside kernel)
        a_half_A = 0.5 * delta_for_A * A_expanded  # For A discretization
        a_half_B = 0.5 * delta_expanded * A_expanded  # For B discretization
        denom_A = one - a_half_A
        denom_B = one - a_half_B
        A_bar = (one + a_half_A) / denom_A  # (b, P, L, 1)
        deltaB_u = (delta_expanded / denom_B) * Bu_expanded  # (b, P, L, 1)
    elif discretization == "zoh":
        # A_bar = exp(delta_for_A * A)
        # B_bar = (exp(delta * A) - 1) / A (identity B inside kernel)
        A_bar = torch.exp(delta_for_A * A_expanded)  # (b, P, L, 1)
        A_disc_B = torch.exp(delta_expanded * A_expanded)
        deltaB_u = (
            (A_disc_B - one) / A_expanded
        ) * Bu_expanded  # (b, P, L, 1)
    elif discretization == "dirac":
        # A_bar = exp(delta_for_A * A)
        # B_bar = 1 (identity, no delta scaling)
        A_bar = torch.exp(delta_for_A * A_expanded)  # (b, P, L, 1)
        deltaB_u = Bu_expanded  # (b, P, L, 1) - no delta scaling
    else:
        raise ValueError(f"Unknown discretization method: {discretization}")

    # Sequential scan in state space
    x = torch.zeros((batch, P, 1), dtype=A.dtype, device=A.device)
    xs = []

    for i in range(seqlen):
        x = A_bar[:, :, i] * x + deltaB_u[:, :, i]
        xs.append(x.squeeze(-1))  # (b, P)

    last_state = x.squeeze(-1)  # (batch, P)
    x_seq = torch.stack(xs, dim=2)  # (b, P, L)

    # Step 2: Project state to output: y = C @ x
    # x_seq: (batch, P, seqlen), C: (H, P) -> y: (batch, H, seqlen)
    y = torch.einsum("hp,bpl->bhl", C, x_seq)

    out = y.to(dtype_in)

    if return_last_state:
        return out, last_state
    return out


class S5InnerFn(torch.autograd.Function):
    """
    The complete S5 model inner function with custom forward and backward.

    This wraps the simplified scan kernel and adds:
    1. Conjugate symmetry handling (2 * real if conj_sym else real)
    2. Skip connection with D matrix

    The scan kernel computes: x[t] = A_bar * x[t-1] + B_bar * (B @ u)[t], y = C @ x
    This function then applies: out = (2 if conj_sym else 1) * Re(y) + D * u.real
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        deltaA: torch.Tensor | None,
        discretization: str,
        conj_sym: bool,
    ):
        """
        Forward pass for the complete S5 model inner function.

        Args:
            ctx (Any): Autograd context.
            u (torch.Tensor): Complex input tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
            delta (torch.Tensor): Real timestep tensor of shape ``(batch, P, seqlen)``, dtype=float32.
            A (torch.Tensor): Complex eigenvalues tensor of shape ``(P,)`` or ``(P, 1)``, dtype=complex64.
            B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype=complex64.
            C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype=complex64.
            D (torch.Tensor): Real skip connection tensor of shape ``(H,)``, dtype=float32.
            deltaA (torch.Tensor | None): Optional separate timestep for A discretization of shape ``(batch, P, seqlen)``.
            discretization (str): Discretization method ('bilinear', 'zoh', or 'dirac').
            conj_sym (bool): If True, output is 2 * Re(y), else Re(y).

        Returns:
            torch.Tensor: Real output tensor of shape ``(batch, H, seqlen)``, dtype=float32.
        """
        # Ensure contiguous
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if deltaA is not None and deltaA.stride(-1) != 1:
            deltaA = deltaA.contiguous()

        P = A.shape[0]

        # Ensure A is (P, 1) for the kernel
        if A.dim() == 1:
            A_kernel = A.unsqueeze(-1)
        else:
            A_kernel = A
        dstate = A_kernel.shape[1]

        # Project input to state space: Bu = B @ u -> (batch, P, L)
        Bu = torch.einsum("ph,bhl->bpl", B, u)

        # Create identity B and C for the kernel (diagonal SSM)
        B_identity = torch.ones(P, dstate, dtype=A.dtype, device=A.device)
        C_identity = torch.ones(P, dstate, dtype=A.dtype, device=A.device)

        # Kernel forward pass
        scan_out, xs = simplified_scan_cuda.fwd(
            Bu,
            delta,
            A_kernel,
            B_identity,
            C_identity,
            deltaA,
            discretization,
        )

        # Project state to output: y_complex = C @ scan_out -> (batch, H, L)
        y_complex = torch.einsum("hp,bpl->bhl", C, scan_out)

        # Apply conjugate symmetry
        if conj_sym:
            y_real = 2 * y_complex.real
        else:
            y_real = y_complex.real

        # Apply skip connection: y = y_real + D * u.real
        # Note: u is complex, D operates on real part
        u_real = u.real if u.is_complex() else u
        y = y_real + D.unsqueeze(0).unsqueeze(-1) * u_real

        # Save for backward
        ctx.discretization = discretization
        ctx.conj_sym = conj_sym
        ctx.has_deltaA = deltaA is not None
        ctx.save_for_backward(
            u,
            delta,
            A_kernel,
            B,
            C,
            D,
            Bu,
            scan_out,
            xs,
            B_identity,
            C_identity,
            deltaA if deltaA is not None else torch.empty(0, device=u.device),
        )

        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, dout: torch.Tensor):
        """
        Backward pass computing gradients for all inputs.

        Args:
            ctx (Any): Autograd context.
            dout (torch.Tensor): Gradient of loss w.r.t. output tensor of shape ``(batch, H, seqlen)``, real.

        Returns:
            tuple: Gradients for u, delta, A, B, C, D, deltaA, None, None.
        """
        (
            u,
            delta,
            A,
            B,
            C,
            D,
            Bu,
            scan_out,
            xs,
            B_identity,
            C_identity,
            deltaA_saved,
        ) = ctx.saved_tensors
        discretization = ctx.discretization
        conj_sym = ctx.conj_sym
        deltaA = deltaA_saved if ctx.has_deltaA else None

        # dD = sum(dout * u_real) over batch and seqlen
        u_real = u.real if u.is_complex() else u
        dD = torch.einsum("bhl,bhl->h", dout, u_real)
        du_skip = D.unsqueeze(0).unsqueeze(-1) * dout  # (batch, H, L)

        # Gradient through conjugate symmetry: y_real = (2 if conj_sym else 1) * Re(y_complex)
        # dy_complex = (2 if conj_sym else 1) * dout (as complex with zero imaginary)
        scale = 2.0 if conj_sym else 1.0
        dy_complex = torch.complex(scale * dout, torch.zeros_like(dout))

        # dscan_out = C^H @ dy_complex
        dscan_out = torch.einsum("hp,bhl->bpl", C.conj(), dy_complex)

        # dC = dy_complex @ scan_out^H summed over batch and seqlen
        dC = torch.einsum("bhl,bpl->hp", dy_complex, scan_out.conj())

        if dscan_out.stride(-1) != 1:
            dscan_out = dscan_out.contiguous()

        # Backward through kernel (with identity B/C)
        result = simplified_scan_cuda.bwd(
            Bu,
            delta,
            A,
            B_identity,
            C_identity,
            deltaA,
            dscan_out,
            xs,
            discretization,
        )
        dBu, ddelta, dA = result[0], result[1], result[2]
        ddeltaA = None
        if ctx.has_deltaA and len(result) > 5:
            ddeltaA = result[5]

        # Gradient through B projection: Bu = B @ u
        # du_scan = B^H @ dBu
        du_scan = torch.einsum("ph,bpl->bhl", B.conj(), dBu)

        # dB = dBu @ u^H summed over batch and seqlen
        dB = torch.einsum("bpl,bhl->ph", dBu, u.conj())

        # Total gradient for u (complex): du_scan + du_skip (as complex)
        if u.is_complex():
            du = du_scan + torch.complex(du_skip, torch.zeros_like(du_skip))
        else:
            du = du_scan.real + du_skip

        # Squeeze dA if it was originally 1D
        if dA.dim() == 2 and dA.shape[1] == 1:
            dA = dA.squeeze(-1)

        return (
            du,
            ddelta,
            dA,
            dB,
            dC,
            dD,
            ddeltaA,
            None,  # discretization
            None,  # conj_sym
        )


def s5_inner_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    deltaA: torch.Tensor | None = None,
    discretization: str = "bilinear",
    conj_sym: bool = True,
) -> torch.Tensor:
    """
    S5 inner function using CUDA kernel.

    Computes the complete S5 forward pass:
    1. SSM scan: x[t] = A_bar * x[t-1] + B_bar * (B @ u)[t], y = C @ x
    2. Conjugate symmetry: y_real = (2 if conj_sym else 1) * Re(y)
    3. Skip connection: out = y_real + D * u.real

    Args:
        u (torch.Tensor): Complex input tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
        delta (torch.Tensor): Real timestep tensor of shape ``(batch, P, seqlen)``, dtype=float32.
        A (torch.Tensor): Complex eigenvalues tensor of shape ``(P,)`` or ``(P, 1)``, dtype=complex64.
        B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype=complex64.
        C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype=complex64.
        D (torch.Tensor): Real skip connection tensor of shape ``(H,)``, dtype=float32.
        deltaA (torch.Tensor | None, optional): Optional separate timestep for A discretization of shape ``(batch, P, seqlen)``. Defaults to None.
        discretization (str, optional): Discretization method ('bilinear', 'zoh', or 'dirac'). Defaults to "bilinear".
        conj_sym (bool, optional): If True, output is 2 * Re(y), else Re(y). Defaults to True.

    Returns:
        torch.Tensor: Real output tensor of shape ``(batch, H, seqlen)``, dtype=float32.
    """
    return S5InnerFn.apply(
        u, delta, A, B, C, D, deltaA, discretization, conj_sym
    )


def s5_inner_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    deltaA: torch.Tensor | None = None,
    discretization: str = "bilinear",
    conj_sym: bool = True,
) -> torch.Tensor:
    """
    Reference implementation of S5 inner function (pure PyTorch).

    Computes the complete S5 forward pass:
    1. SSM scan: x[t] = A_bar * x[t-1] + B_bar * (B @ u)[t], y = C @ x
    2. Conjugate symmetry: y_real = (2 if conj_sym else 1) * Re(y)
    3. Skip connection: out = y_real + D * u.real

    Args:
        u (torch.Tensor): Complex input tensor of shape ``(batch, H, seqlen)``, dtype=complex64.
        delta (torch.Tensor): Real timestep tensor of shape ``(batch, P, seqlen)``, dtype=float32.
        A (torch.Tensor): Complex eigenvalues tensor of shape ``(P,)`` or ``(P, 1)``, dtype=complex64.
        B (torch.Tensor): Complex projection matrix of shape ``(P, H)``, dtype=complex64.
        C (torch.Tensor): Complex projection matrix of shape ``(H, P)``, dtype=complex64.
        D (torch.Tensor): Real skip connection tensor of shape ``(H,)``, dtype=float32.
        deltaA (torch.Tensor | None, optional): Optional separate timestep for A discretization of shape ``(batch, P, seqlen)``. Defaults to None.
        discretization (str, optional): Discretization method ('bilinear', 'zoh', or 'dirac'). Defaults to "bilinear".
        conj_sym (bool, optional): If True, output is 2 * Re(y), else Re(y). Defaults to True.

    Returns:
        torch.Tensor: Real output tensor of shape ``(batch, H, seqlen)``, dtype=float32.
    """
    # Use simplified_scan_ref for the SSM computation
    y_complex = simplified_scan_ref(
        u,
        delta,
        A,
        B,
        C,
        deltaA=deltaA,
        return_last_state=False,
        discretization=discretization,
    )

    # Apply conjugate symmetry
    if conj_sym:
        y_real = 2 * y_complex.real
    else:
        y_real = y_complex.real

    # Apply skip connection
    u_real = u.real if u.is_complex() else u
    out = y_real + D.unsqueeze(0).unsqueeze(-1) * u_real

    return out