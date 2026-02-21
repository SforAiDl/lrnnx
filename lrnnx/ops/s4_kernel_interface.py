import math
from collections import defaultdict
from functools import partial
from typing import Mapping, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lrnnx.ops.s4_utils import (
    combination,
    get_cauchy_kernel,
    get_vandermonde_kernel,
    get_vandermonde_transpose_kernel,
    inv_transform,
    param_transform,
    power,
    process_dplr_params,
    process_ssm_params,
    setup_default_state,
)

# Function aliases
contract = torch.einsum
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

cauchy_k = get_cauchy_kernel()
vandermonde_k = get_vandermonde_kernel()
vandermonde_transpose_k = get_vandermonde_transpose_kernel()


class S4KernelBase(nn.Module):
    """
    Base class for S4 kernels - receives parameters from the parent model.

    Args:
        d_model (int): Model dimension.
        l_max (int | None): Maximum sequence length.
        channels (int): Number of channels/heads.
        param_config (dict): A dictionary containing:
            
            * Parameter references: A_real, A_imag, B, C, inv_dt, P (nn.Parameters owned by S4/S4D)
            * Computed scalars: N, H, channels, rank, repeat
            * Config flags: dt_fast, real_transform, imag_transform, dt_transform,
              is_real, deterministic, verbose
            * S4D-only: disc
    """

    def __init__(
        self,
        d_model: int,
        l_max: Optional[int],
        channels: int,
        param_config: dict,
    ):
        super().__init__()

        # ── parameter references (owned by the parent S4/S4D model) ──
        self.A_real = param_config["A_real"]
        self.A_imag = param_config.get("A_imag")  # None when is_real=True
        self.B = param_config["B"]
        self.C = param_config["C"]
        self.P = param_config.get("P")  # None for S4D (diagonal)
        self.inv_dt = param_config["inv_dt"]

        # ── derived dimensions (already computed by the parent) ──
        self.N = param_config["N"]  # halved for conjugate symmetry
        self.H = param_config["H"]
        self.channels = param_config["channels"]
        self.rank = param_config["rank"]
        self.repeat = param_config["repeat"]  # broadcast factor H // n_ssm

        # ── flags / transforms ──
        self.dt_fast = param_config["dt_fast"]
        self.real_transform = param_config["real_transform"]
        self.imag_transform = param_config["imag_transform"]
        self.dt_transform = param_config["dt_transform"]
        self.is_real = param_config["is_real"]
        self.deterministic = param_config["deterministic"]
        self.verbose = param_config["verbose"]

        # ── model geometry ──
        self.d_model = d_model
        self.L = self.l_max = l_max


class S4Kernel(S4KernelBase):
    """
    SSM kernel for diagonal + low rank (DPLR) state matrices - pure convolution operation.

    Args:
        d_model (int): Model dimension.
        l_max (int | None): Maximum sequence length.
        channels (int): Number of channels/heads.
        param_config (dict): Configuration dictionary containing parameter references and flags.
    """

    def __init__(self, d_model, l_max, channels, param_config):
        super().__init__(d_model, l_max, channels, param_config)
        self.register_buffer("l_kernel", torch.tensor(0))

    def forward(self, state=None, rate=1.0, L=None):
        """
        Compute SSM convolution kernel - the core operation.

        Args:
            state (torch.Tensor, optional): State tensor. Defaults to None.
            rate (float, optional): Sampling rate. Defaults to 1.0.
            L (int, optional): Sequence length. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - k_B : Convolution kernel.
                - k_state : Kernel state, if state is provided.
        """
        # Initialize C~ if necessary
        if (
            self.l_kernel.item() == 0
            and self.l_max is not None
            and self.l_max > 0
        ):
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        if L is None:
            L = round(self.l_kernel.item() / rate)

        continuous_L = round(rate * L)
        while continuous_L > self.l_kernel.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.l_kernel.item() / rate)

        # Process parameters
        dt, A, B, C, P, Q = process_dplr_params(
            self.A_real,
            self.A_imag if not self.is_real else None,
            self.B,
            self.C,
            self.P,
            self.inv_dt,
            self.real_transform,
            self.imag_transform,
            self.dt_transform,
            self.dt_fast,
            self.is_real,
            self.repeat,
            rate,
        )

        # Get FFT nodes
        omega, z = self._omega(
            discrete_L, dtype=A.dtype, device=A.device, cache=(rate == 1.0)
        )

        # Augment B with state
        if state is not None:
            s = _conj(state) if state.size(-1) == self.N else state
            sA = s * _conj(A) - contract(
                "bhm, rhm, rhn -> bhn", s, _conj(Q), _conj(P)
            )
            s = s / dt + sA / 2
            s = s[..., : self.N]
            B = torch.cat([s, B], dim=-3)

        # Incorporate dt into A
        A = A * dt

        # Stack B and P, C and Q
        B = torch.cat([B, P], dim=-3)
        C = torch.cat([C, Q], dim=-3)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)
        v = v * dt

        # Calculate resolvent at omega
        r = cauchy_k(v, z, A)

        # Low-rank Woodbury correction
        k_f = self._woodbury_correction(r)

        # Final correction for bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]
        else:
            k_state = None
        k_B = k[-1, :, :, :]

        return k_B, k_state

    @torch.no_grad()
    def _setup_C(self, L):
        """Construct C~ from C."""
        if self.l_kernel.item() == 0:
            if self.verbose:
                print(f"S4: Initializing kernel to length {L}")
            double_length = False
        elif L > self.l_kernel.item():
            if self.verbose:
                print(
                    f"S4: Doubling length from {self.l_kernel.item()} to {2*self.l_kernel.item()}"
                )
            double_length = True
            L = self.l_kernel.item()
        else:
            return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)

        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length:
            prod = -prod
        C_ = C_ - prod
        C_ = C_[..., : self.N]
        self.C.copy_(_c2r(C_))

        self.l_kernel = (
            2 * self.l_kernel if double_length else self.l_kernel + L
        )

    def _omega(self, L, dtype, device, cache=True):
        """Calculate (and cache) FFT nodes."""
        if (
            cache
            and hasattr(self, "omega")
            and self.omega.size(-1) == L // 2 + 1
        ):
            return self.omega, self.z

        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        if cache:
            self.omega = omega
            self.z = z
        return omega, z

    def _woodbury_correction(self, r):
        """Apply low-rank Woodbury correction."""
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (
                1 + r[-1:, -1:, :, :]
            )
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[
                :1, 1:, :, :
            ] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum(
                "i j h n, j k h n, k l h n -> i l h n", r01, r11, r10
            )

        return k_f

    @torch.no_grad()
    def double_length(self):
        """Double the sequence length representation."""
        self._setup_C(2 * self.l_kernel)

    @torch.no_grad()
    def _setup_linear(self):
        """Preprocessing for fast linear-time stepping."""
        dt, A, B, C, P, Q = process_dplr_params(
            self.A_real,
            self.A_imag if not self.is_real else None,
            self.B,
            self.C,
            self.P,
            self.inv_dt,
            self.real_transform,
            self.imag_transform,
            self.dt_transform,
            self.dt_fast,
            self.is_real,
            self.repeat,
            rate=1.0,
        )

        D = (2.0 / dt - A).reciprocal()
        R = (
            torch.eye(self.rank, dtype=A.dtype, device=A.device)
            + 2 * contract("r h n, h n, s h n -> h r s", Q, D, P).real
        )
        Q_D = rearrange(Q * D, "r h n -> h r n")
        try:
            R = torch.linalg.solve(R, Q_D)
        except:
            R = torch.tensor(
                np.linalg.solve(
                    R.to(Q_D).contiguous().detach().cpu(),
                    Q_D.contiguous().detach().cpu(),
                )
            ).to(Q_D)
        R = rearrange(R, "h r n -> r h n")

        self.step_params = {
            "D": D,
            "R": R,
            "P": P,
            "Q": Q,
            "B": B,
            "E": 2.0 / dt + A,
        }

    def _step_state_linear(self, u=None, state=None):
        """Linear-time step function."""
        C = _r2c(self.C)

        if u is None:
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N:
            contract_fn = lambda p, x, y: contract(
                "r h n, r h m, ... h m -> ... h n",
                _conj(p),
                _conj(x),
                _conj(y),
            )[..., : self.N]
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract(
                "r h n, r h m, ... h m -> ... h n", p, x, y
            )

        D, E, R, P, Q, B = (
            step_params["D"],
            step_params["E"],
            step_params["R"],
            step_params["P"],
            step_params["Q"],
            step_params["B"],
        )

        new_state = E * state - contract_fn(P, Q, state)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation."""
        self._setup_linear()
        C = _r2c(self.C)

        state = torch.eye(
            2 * self.N, dtype=C.dtype, device=C.device
        ).unsqueeze(-2)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, "1 h n -> h n")
        return dA, dB

    def _step_state(self, u, state):
        """Quadratic step function."""
        next_state = torch.einsum(
            self.state_contraction, self.dA, state
        ) + torch.einsum(self.input_contraction, self.dB, u)
        return next_state

    def _setup_step(self, mode="dense"):
        """Set up dA, dB, dC for stepping."""
        # Ensure C has been transformed to C~ before we read it.
        # forward() does this automatically, but _setup_step can be called
        # directly (e.g. for manual recurrence in tests/inference) without
        # a prior forward pass.
        if (
            self.l_kernel.item() == 0
            and self.l_max is not None
            and self.l_max > 0
        ):
            self._setup_C(self.l_max)

        self.dA, self.dB = self._setup_state()

        C = _conj(_r2c(self.C))
        if self.l_kernel.item() == 0:
            dC = C
        else:
            dA_L = power(self.l_kernel.item(), self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)
            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2), C.unsqueeze(-1)
            ).squeeze(-1)
        self.dC = dC

        self._step_mode = mode
        if mode == "linear":
            self.dC = 2 * self.dC[:, :, : self.N]
        elif mode == "diagonal":
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            if self.verbose:
                print(
                    "Diagonalization error:",
                    torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA),
                )
            self.dA = L
            self.dB = contract("h n m, h m -> h n", V_inv, self.dB)
            self.dC = contract("h n m, c h n -> c h m", V, self.dC)
        elif mode == "dense":
            pass
        else:
            raise NotImplementedError(
                "Step mode must be {'dense' | 'linear' | 'diagonal'}"
            )

    def default_state(self, *batch_shape):
        """
        Create default state.

        Args:
            *batch_shape: Variable length argument list for batch dimensions.

        Returns:
            torch.Tensor: A zero-initialized state tensor.
        """
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        step_mode = getattr(self, "_step_mode", "dense")
        if step_mode != "linear":
            N *= 2
            if step_mode == "diagonal":
                self.state_contraction = "h n, ... h n -> ... h n"
            else:
                self.state_contraction = "h m n, ... h n -> ... h m"
            self.input_contraction = "h n, ... h -> ... h n"

        self.output_contraction = "c h n, ... h n -> ... c h"
        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """
        Perform single step.

        Args:
            u (torch.Tensor): Input tensor.
            state (torch.Tensor): Current state tensor.

        Returns:
            A tuple containing:
                - y.real (torch.Tensor): Output tensor.
                - new_state (torch.Tensor): Updated state tensor.
        """
        if self._step_mode == "linear":
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = torch.einsum(self.output_contraction, self.dC, new_state)
        return y.real, new_state

    def forward_state(self, u, state):
        """
        Forward the state through a sequence.

        Args:
            u (torch.Tensor): Input sequence tensor of shape ``(B, H, L)``.
            state (torch.Tensor): State tensor of shape ``(B, H, N)``.

        Returns:
            torch.Tensor: The updated state tensor.
        """
        dA, dB = self._setup_state()

        conj = state.size(-1) != dA.size(-1)
        if conj:
            state = _conj(state)

        v = contract("h n, b h l -> b h n l", dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj:
            next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state


class S4DKernel(S4KernelBase):
    """
    SSM kernel using diagonal state matrix (S4D model) - pure convolution operation.

    Args:
        d_model (int): Model dimension.
        l_max (int | None): Maximum sequence length.
        channels (int): Number of channels/heads.
        param_config (dict): Configuration dictionary containing parameter references and flags,
            including the S4D-specific 'disc' key.
    """

    def __init__(self, d_model, l_max, channels, param_config):
        self.disc = param_config.get("disc", "zoh")
        super().__init__(d_model, l_max, channels, param_config)

    def forward(self, L, state=None, rate=1.0):
        """
        Compute SSM convolution kernel - the core operation.

        Args:
            L (int): Sequence length.
            state (torch.Tensor, optional): State tensor. Defaults to None.
            rate (float, optional): Sampling rate. Defaults to 1.0.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - K : Convolution kernel.
                - K_state : Kernel state, if state is provided.
        """
        # Process parameters
        dt, A, B, C, dtA = process_ssm_params(
            self.A_real,
            self.A_imag if not self.is_real else None,
            self.B,
            self.C,
            self.inv_dt,
            self.real_transform,
            self.imag_transform,
            self.dt_transform,
            self.dt_fast,
            self.is_real,
            self.repeat,
            rate,
        )

        # Augment B with state
        if state is not None:
            s = state / dt
            if self.disc == "bilinear":
                s = s * (1.0 + dtA / 2)
            elif self.disc == "zoh":
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.0)
            B = torch.cat([s, B], dim=-3)

        # Combine B and C
        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        # Main kernel computation
        if self.disc == "zoh":
            C = C * (torch.exp(dtA) - 1.0) / A
            K = vandermonde_k(C, dtA, L)
        elif self.disc == "bilinear":
            C = C * (1.0 - dtA / 2).reciprocal() * dt
            dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            K = vandermonde_k(C, dA.log(), L)
        else:
            raise ValueError(f"Discretization {self.disc} not supported")

        K = K.view(-1, self.channels, self.H, L)

        if state is not None:
            K_state = K[:-1, :, :, :]
        else:
            K_state = None
        K = K[-1, :, :, :]

        return K, K_state

    def _setup_step(self):
        """Set up dA, dB, dC for stepping."""
        dt, A, B, C, dtA = process_ssm_params(
            self.A_real,
            self.A_imag if not self.is_real else None,
            self.B,
            self.C,
            self.inv_dt,
            self.real_transform,
            self.imag_transform,
            self.dt_transform,
            self.dt_fast,
            self.is_real,
            self.repeat,
            rate=1.0,
        )

        if self.disc == "zoh":
            self.dA = torch.exp(dtA)
            self.dB = B * (torch.exp(dtA) - 1.0) / A
        elif self.disc == "bilinear":
            self.dA = (1.0 + dtA / 2) / (1.0 - dtA / 2)
            self.dB = B * (1.0 - dtA / 2).reciprocal() * dt

        self.dB = rearrange(self.dB, "1 h n -> h n")
        self.dC = C

    def default_state(self, *batch_shape):
        """
        Create default state.

        Args:
            *batch_shape: Variable length argument list for batch dimensions.

        Returns:
            torch.Tensor: A zero-initialized state tensor.
        """
        C = _r2c(self.C)
        # For diagonal S4D, we don't need to double N - state is just (H, N)
        state = torch.zeros(
            *batch_shape, self.H, self.N, dtype=C.dtype, device=C.device
        )
        return state

    def step(self, u, state):
        """
        Single step operation.

        Args:
            u (torch.Tensor): Input tensor of shape ``(B, H)``.
            state (torch.Tensor): Current state tensor of shape ``(B, H, N)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - y.real : Output tensor (scaled by 2).
                - next_state : Updated state tensor.
        """
        next_state = contract(
            "h n, b h n -> b h n", self.dA, state
        ) + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2 * y.real, next_state

    def forward_state(self, u, state):
        """
        Pass state forward through sequence.

        Args:
            u (torch.Tensor): Input sequence tensor of shape ``(B, H, L)``.
            state (torch.Tensor): Initial state tensor of shape ``(B, H, N)``.

        Returns:
            torch.Tensor: The updated state tensor.
        """
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous()
        v = vandermonde_transpose_k(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state


kernel_registry = {
    "s4": S4Kernel,
    "s4d": S4DKernel,
}