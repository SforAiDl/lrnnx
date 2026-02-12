"""
Centaurus: Let SSMs be Conv Nets implementation.
https://openreview.net/forum?id=PkpNRmBZ32
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

import torch
from torch import Tensor, einsum, nn

from lrnnx.core.convolution import opt_ssm_forward
from lrnnx.models.lti.base import LTI_LRNN


class CentaurusBase(LTI_LRNN, ABC):
    """Common base for Centaurus mode variants (neck, pointwise, dws, full).

    Example
    -------
    >>> # Use via subclasses (CentaurusNeck, CentaurusDWS, CentaurusFull, CentaurusPWNeck)
    >>> # or through the Centaurus wrapper
    >>> model = CentaurusNeck(d_model=64, d_state=64, sub_state_dim=8, discretization="zoh")
    >>> x = torch.randn(2, 128, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 128, 64])
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        sub_state_dim: int,
        discretization: Literal["zoh", "bilinear", "dirac", "async"] = "zoh",
    ) -> None:
        super().__init__(discretization=discretization)

        self.discretization = discretization
        self.d_model = d_model
        self.d_state = d_state
        self.sub_state_dim = sub_state_dim

        self._init_common_parameters()
        self._init_mode_parameters()

    def _positive_delta(self) -> Tensor:
        """
        Ensures the learned delta parameters remain strictly positive by
        exponentiating the free log-parameter (matches the public Centaurus
        implementation that stores log_dt and applies exp(log_dt)).
        """
        return torch.exp(self.log_delta)

    def _init_common_parameters(self) -> None:
        """Initialise A, E, and the log-parameter for delta shared across all modes."""
        m = torch.arange(self.sub_state_dim, dtype=torch.float32)
        a_base = torch.complex(
            real=torch.full((self.sub_state_dim,), -0.5),
            imag=m * math.pi / self.sub_state_dim,
        )
        A_init = a_base.unsqueeze(0).repeat(self.d_state, 1)
        self.A = nn.Parameter(A_init)

        self.E = nn.Parameter(
            torch.randn(self.d_state, self.sub_state_dim) * math.sqrt(2)
        )

        delta_init = torch.logspace(-3, -1, steps=self.d_state)
        self.log_delta = nn.Parameter(delta_init.log())

    @abstractmethod
    def _init_mode_parameters(self) -> None:
        """Implemented by subclasses to create B/C and any extra buffers."""

    @abstractmethod
    def _effective_B(self) -> Tensor:
        """Mode-specific: discretised input projection (N, H)."""

    @abstractmethod
    def _effective_C(self) -> Tensor:
        """Mode-specific: output projection (H, N)."""

    def compute_kernel(self) -> tuple[Tensor, Tensor]:
        """
        Computes the discrete-time latent convolution kernel with intra-state mode
        mixing using the shared Centaurus formulation.

        Returns
        -------
        tuple[Tensor, Tensor]
            - k: Latent kernel of shape (N, L), where N is the number of state channels.
            - Empty tensor: Placeholder for compatibility with standard LTI interface expectations.
        """
        arange = torch.arange(self.seq_len, device=self.A.device)  # (L,)
        dtA = einsum("n,nm->nm", self._positive_delta(), self.A)
        K_intermediate = einsum("nm,l->nml", dtA, arange).exp()
        K = einsum("nml,nm->nl", K_intermediate.real, self.E)
        return K, torch.empty(0, device=self.A.device)

    def discretize(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        This method is intentionally not implemented for Centaurus variants.

        Raises
        ------
        NotImplementedError
            Always raised, since Centaurus does not support explicit discretization
            via this method.
        """
        raise NotImplementedError(
            "Centaurus implicitly performs ZOH discretization in its kernel computation and forward methods. "
            "Currently, it does not support any other discretization strategy, and consequently does not use the standard LTI_LRNN discretize() method."
        )

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through a Centaurus LTI mode variant.

        Parameters
        ----------
        x : Tensor
            Input sequence of shape (B, L, H_in).

        integration_timesteps : Tensor, optional
            Placeholder for async models. Not used in the current implementation.

        lengths : Tensor, optional
            Placeholder for future bidirectional models. Not used in the current implementation.

        Returns
        -------
        Tensor
            Output sequence of shape (B, L, H_out), where H_out is the output channel dimension.
        """
        self.seq_len = x.shape[1]
        u = x  # (B, L, H_in)

        K, _ = self.compute_kernel()  # (N, L)
        B_bar = self._effective_B()  # (N, H_in)
        C_eff = self._effective_C().to(self.A.dtype)  # (H_out, N)

        out = opt_ssm_forward(u, K, B_bar, C_eff)  # (B, L, H_out)
        return out  # (B, L, H_out)

    def _precompute_discrete(self, *, device=None):
        """
        Precomputes discrete-time system matrices for Centaurus.
        """
        dev = device or self.A.device
        delta = self._positive_delta().to(dev)  # (N,)
        A = self.A.to(dev)  # (N, M) complex

        # precompute the discrete spectrum
        dtA = einsum("n,nm->nm", delta, A)  # (N, M) complex
        A_bar = dtA.exp()  # (N, M) complex
        # Effective projections for the current mode (see subclasses)
        B_bar = self._effective_B().to(dev)  # (N, H)
        C_eff = self._effective_C().to(dev)  # (H, N)

        # Register once; subsequent calls just overwrite the non-persistent buffers.
        if not hasattr(self, "_A_bar"):
            # First streaming call: allocate device-local buffers.
            self.register_buffer("_A_bar", A_bar, persistent=False)
            self.register_buffer("_B_bar", B_bar, persistent=False)
            self.register_buffer("_C_eff", C_eff, persistent=False)
        else:
            self._A_bar = A_bar
            self._B_bar = B_bar
            self._C_eff = C_eff

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocate initial streaming state and cache matrices.

        Returns
        -------
        Dict[str, Any]
            Cache dict with initial state and precomputed discrete parameters.
        """
        dev = self.A.device
        cdt = (
            self.A.dtype
        )  # Centaurus keeps the recurrent state in the complex plane

        self._precompute_discrete(device=dev)

        # Maintain the tensorised (N, M) state so the lifted input s[n] can be
        # broadcast to each sub‑state m before mixing back to channels.
        initial_state = torch.zeros(
            batch_size,
            self.d_state,
            self.sub_state_dim,
            device=dev,
            dtype=cdt,
        )

        return {
            "lrnn_state": initial_state,
            "A_bar": self._A_bar,
            "B_bar": self._B_bar,
            "E": self.E.to(dev),
            "C": self._C_eff,
        }

    def step(
        self,
        x: torch.Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Single-timestep streaming update for a Centaurus variant.

        This method performs one recurrent update of the Centaurus block using
        the cached discrete-time parameters in the (B, N, M) layout.

        Inputs
        ------
        x : torch.Tensor
            Input tensor of shape (B, H_in) - the current timestep input.
        inference_cache : Dict[str, Any]
            Cache from allocate_inference_cache().

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape (B, H_out) (real).
        inference_cache : Dict[str, Any]
            Updated cache dictionary.
        """

        dev = x.device
        state = inference_cache["lrnn_state"]

        A_bar = inference_cache["A_bar"].to(dev)
        B_bar = inference_cache["B_bar"].to(dev)
        E = inference_cache["E"].to(dev)
        C = inference_cache["C"].to(dev)

        # Project input channels into spectral modes (one scalar per state n)
        u_bn = torch.einsum("nh,bh->bn", B_bar, x.to(B_bar.dtype))  # (B,N)

        # Sub‑state evolution
        new_state = A_bar.unsqueeze(0) * state + u_bn.unsqueeze(-1)  # (B,N,M)

        # Collapse sub‑states with E and keep only the real part before readout
        mixed_bn = torch.einsum("bnm,nm->bn", new_state.real, E)  # (B,N)

        # Project back to channels
        y = torch.einsum("hn,bn->bh", C, mixed_bn)  # (B,H_out)

        inference_cache["lrnn_state"].copy_(new_state)
        return y, inference_cache


class CentaurusNeck(CentaurusBase):
    """Bottleneck block with dense in/out projections.

    Example
    -------
    >>> model = CentaurusNeck(d_model=64, d_state=64, sub_state_dim=8)
    >>> x = torch.randn(2, 128, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 128, 64])
    """

    def _init_mode_parameters(self) -> None:
        # Expose legacy mode attribute for reference tests
        self.mode = "neck"
        self.B = nn.Parameter(torch.empty(self.d_state, self.d_model))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

        self.C = nn.Parameter(torch.empty(self.d_model, self.d_state))
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))

    def _effective_B(self) -> Tensor:
        return self._positive_delta()[:, None] * self.B

    def _effective_C(self) -> Tensor:
        return self.C


class CentaurusDWS(CentaurusBase):
    """Depthwise-separable block with one state per channel.

    Example
    -------
    >>> model = CentaurusDWS(d_model=64, d_state=64, sub_state_dim=8)
    >>> x = torch.randn(2, 128, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 128, 64])
    """

    def _init_mode_parameters(self) -> None:
        self.mode = "dws"
        self.B = nn.Parameter(torch.ones(self.d_model))
        self.C = nn.Parameter(torch.ones(self.d_model))

    def _effective_B(self) -> Tensor:
        delta = self._positive_delta()
        return delta[:, None] * torch.diag(self.B)

    def _effective_C(self) -> Tensor:
        return torch.diag(self.C)


class CentaurusFull(CentaurusBase):
    """Fully connected block with a state per (in, out) pair.

    Example
    -------
    >>> model = CentaurusFull(d_model=64, d_state=64, sub_state_dim=8)
    >>> x = torch.randn(2, 128, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 128, 64])
    """

    def _init_mode_parameters(self) -> None:
        self.mode = "full"
        expected = self.d_model * self.d_model
        in_indices = []
        out_indices = []
        for out_ch in range(self.d_model):
            for in_ch in range(self.d_model):
                out_indices.append(out_ch)
                in_indices.append(in_ch)
        self.register_buffer(
            "_full_in_index",
            torch.tensor(in_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_full_out_index",
            torch.tensor(out_indices, dtype=torch.long),
            persistent=False,
        )

        self.B = nn.Parameter(
            torch.randn(self.d_state) * math.sqrt(2.0 / max(self.d_model, 1))
        )
        self.C = nn.Parameter(
            torch.randn(self.d_state) * math.sqrt(2.0 / max(self.d_state, 1))
        )

    def _effective_B(self) -> Tensor:
        delta = self._positive_delta()
        B_vals = self.B
        B_mat = torch.zeros(
            self.d_state,
            self.d_model,
            device=B_vals.device,
            dtype=B_vals.dtype,
        )
        # Use scatter_ to place each scalar B value at its (state, in_channel) slot
        # according to precomputed indices, avoiding Python loops when building the
        # sparse-to-dense projection matrix.
        B_mat.scatter_(
            1, self._full_in_index.unsqueeze(1), B_vals.unsqueeze(1)
        )
        return delta[:, None] * B_mat

    def _effective_C(self) -> Tensor:
        C_vals = self.C
        C_mat = torch.zeros(
            self.d_model,
            self.d_state,
            device=C_vals.device,
            dtype=C_vals.dtype,
        )
        # scatter_ maps each scalar C value to its (out_channel, state) slot using
        # the precomputed indices, avoiding explicit Python loops.
        C_mat.scatter_(
            0, self._full_out_index.unsqueeze(0), C_vals.unsqueeze(0)
        )
        return C_mat


class CentaurusPWNeck(CentaurusBase):
    """Pointwise bottleneck (s5 in public implementations) that
    flattens (N, M) -> (N*M).

    This variant removes E-mixing and repeats delta over M sub-states per state,
    yielding independent SISO lanes aggregated in a single flattened axis.

    Example
    -------
    >>> model = CentaurusPWNeck(d_model=64, d_state=64, sub_state_dim=8)
    >>> x = torch.randn(2, 128, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 128, 64])
    """

    def _init_mode_parameters(self) -> None:
        # Expose mode for reference helpers/tests
        self.mode = "pointwise"

        tot_states = self.d_state * self.sub_state_dim
        # B: (N*M, H)
        B_init = torch.ones(tot_states, self.d_model) / math.sqrt(
            max(self.d_model, 1)
        )
        self.B = nn.Parameter(B_init)
        # C: (H, N*M)
        C_init = torch.randn(self.d_model, tot_states) * math.sqrt(
            2.0 / max(tot_states, 1)
        )
        self.C = nn.Parameter(C_init)
        # Grouped config: no E; mixing is implicit via flattening
        self.E = None

        # Delta per state n, shared across M sub-states
        delta_init = torch.logspace(-3, -1, steps=self.d_state)
        self.log_delta = nn.Parameter(delta_init.log())

    def _effective_B(self) -> Tensor:
        # (N*M, H) via repeating |delta| across M and scaling rows of B
        delta_rep = (
            self._positive_delta()
            .repeat_interleave(self.sub_state_dim)
            .unsqueeze(1)
        )  # (N*M, 1)
        return delta_rep * self.B

    def _effective_C(self) -> Tensor:
        # (H, N*M)
        return self.C

    def compute_kernel(self) -> tuple[Tensor, Tensor]:
        arange = torch.arange(self.seq_len, device=self.A.device)  # (L,)
        dtA = einsum("n,nm->nm", self._positive_delta(), self.A)  # (N, M)
        dtA_flat = dtA.reshape(-1)  # (N*M,)
        K_intermediate = einsum("s,l->sl", dtA_flat, arange).exp()  # (N*M, L)
        return K_intermediate.real, torch.empty(0, device=self.A.device)

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        self.seq_len = x.shape[1]
        K, _ = self.compute_kernel()  # (N*M, L)
        B_bar = self._effective_B()  # (N*M, H)
        C_eff = self._effective_C().to(self.A.dtype)  # (H, N*M)
        return opt_ssm_forward(x, K, B_bar, C_eff)

    def _precompute_discrete(self, *, device=None):
        dev = device or self.A.device
        delta = self._positive_delta().to(dev)
        A = self.A.to(dev)
        dtA_flat = einsum("n,nm->nm", delta, A).reshape(-1)
        A_bar_flat = dtA_flat.exp()
        B_flat = self._effective_B().to(dev)
        C_eff = self._effective_C().to(dev)
        if not hasattr(self, "_A_bar_flat"):
            self.register_buffer("_A_bar_flat", A_bar_flat, persistent=False)
            self.register_buffer("_B_bar_flat", B_flat, persistent=False)
            self.register_buffer("_C_eff_flat", C_eff, persistent=False)
        else:
            self._A_bar_flat = A_bar_flat
            self._B_bar_flat = B_flat
            self._C_eff_flat = C_eff

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        dev = self.A.device
        cdt = self.A.dtype
        self._precompute_discrete(device=dev)
        state = torch.zeros(
            batch_size,
            self.d_state * self.sub_state_dim,
            device=dev,
            dtype=cdt,
        )
        return {
            "lrnn_state": state,
            "A_bar_flat": self._A_bar_flat,  # (N*M,)
            "B_bar_flat": self._B_bar_flat,  # (N*M, H)
            "C_flat": self._C_eff_flat,  # (H, N*M)
        }

    def step(
        self,
        x: torch.Tensor,
        inference_cache: Dict[str, Any],
        **kwargs,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        dev = x.device
        state = inference_cache["lrnn_state"]
        A_bar_flat = inference_cache["A_bar_flat"].to(dev)  # (N*M,)
        B_bar_flat = inference_cache["B_bar_flat"].to(dev)  # (N*M, H)
        C_flat = inference_cache["C_flat"].to(dev)  # (H, N*M)

        # Project input into each flattened SISO lane and update flattened state.
        input_proj = torch.einsum(
            "sh,bh->bs", B_bar_flat, x.to(B_bar_flat.dtype)
        )
        new_state = A_bar_flat.unsqueeze(0) * state + input_proj.to(
            A_bar_flat.dtype
        )
        # Readout in channel space from flattened SISO real states.
        y = torch.einsum("hs,bs->bh", C_flat, new_state.real)
        inference_cache["lrnn_state"].copy_(new_state)
        return y, inference_cache


class Centaurus:
    """Backwards-compatible wrapper that returns a mode-specific class instance.

    Example
    -------
    >>> model = Centaurus(d_model=64, d_state=64, sub_state_dim=8, mode="neck")
    >>> x = torch.randn(2, 128, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 128, 64])
    """

    def __new__(
        cls,
        d_model: int,
        d_state: int,
        sub_state_dim: int,
        discretization: Literal["zoh", "bilinear", "dirac", "async"] = "zoh",
        mode: Literal["neck", "pointwise", "pw", "s5", "dws", "full"] = "neck",
        **kwargs,
    ):
        mapping = {
            "neck": CentaurusNeck,
            "pointwise": CentaurusPWNeck,
            "pw": CentaurusPWNeck,
            "s5": CentaurusPWNeck,  # alias to match public implementation
            "dws": CentaurusDWS,
            "full": CentaurusFull,
        }

        # Pass through extra kwargs safely; CentaurusBase ignores unknowns.
        return mapping[mode](
            d_model=d_model,
            d_state=d_state,
            sub_state_dim=sub_state_dim,
            discretization=discretization,
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        # __new__ returns the real module; accept arbitrary kwargs for compatibility.
        pass
