# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified to incorporate different discretizations and event-based processing.
# Reference: https://github.com/Efficient-Scalable-Machine-Learning/event-based-mamba.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lrnnx.ops.selective_scan import mamba_inner_fn, selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from lrnnx.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from typing import Any, Dict, Optional, Tuple, Union

from lrnnx.models.ltv.base import LTV_LRNN


class Mamba(LTV_LRNN):
    """
    Mamba: Selective State Space Model with optional event-based processing.

    When integration_timesteps is provided in forward(), uses asymmetric
    discretization (separate dtA and dtB) for event-driven processing.
    Otherwise, uses standard Mamba discretization.

    Example:
        >>> model = Mamba(d_model=64, d_state=16, d_conv=4)
        >>> x = torch.randn(2, 128, 64)
        >>> y = model(x)
        >>> y.shape
        torch.Size([2, 128, 64])

    Args:
        d_model (int): Model dimension.
        d_state (int, optional): SSM state dimension (N). Defaults to 16.
        d_conv (int, optional): Convolution kernel size. Defaults to 4.
        expand (int, optional): Expansion factor for inner dimension. Defaults to 2.
        dt_rank (Union[int, str], optional): Rank for delta projection, "auto" = ceil(d_model / 16). Defaults to "auto".
        dt_min (float, optional): Minimum value for delta initialization. Defaults to 0.001.
        dt_max (float, optional): Maximum value for delta initialization. Defaults to 0.1.
        dt_init (str, optional): Initialization method ("random" or "constant"). Defaults to "random".
        dt_scale (float, optional): Scale factor for dt initialization. Defaults to 1.0.
        dt_init_floor (float, optional): Floor value for dt initialization. Defaults to 1e-4.
        conv_bias (bool, optional): Whether to use bias in convolution. Defaults to True.
        bias (bool, optional): Whether to use bias in linear projections. Defaults to False.
        use_fast_path (bool, optional): Whether to use fused CUDA kernels. Defaults to True.
        layer_idx (int, optional): Layer index for multi-layer caching. Defaults to None.
        device (torch.device, optional): Device for parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type for parameters. Defaults to None.
        discretization (str, optional): Discretization type. Defaults to "mamba".
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        discretization="mamba",
    ):
        # pass None to base class since Mamba handles discretization via CUDA kernel, not discretize_fn
        super().__init__(discretization=None)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        )
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.discretization = discretization

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )

        # for standard Mamba: single dt_proj (used as dtB in event mode)
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # for event mode: separate dtA_proj
        self.dtA_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(
                1, self.d_state + 1, dtype=torch.float32, device=device
            ),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_inner, device=device)
        )  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(
        self,
        hidden_states,
        integration_timesteps: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        inference_cache: Optional[Dict[str, Any]] = None,
    ):
        """
        Forward pass through Mamba.

        Args:
            hidden_states (torch.Tensor): Input tensor, shape ``(B, L, D)``.
            integration_timesteps (torch.Tensor, optional): Time intervals between events.
                Shape ``(B, L)``. When provided, uses asymmetric discretization with
                separate dtA and dtB for event-driven processing. Defaults to None.
            lengths (torch.Tensor, optional): Not used by Mamba currently. Defaults to None.
            inference_cache (dict, optional): Cache for autoregressive generation.
                If provided, contains "conv_state" and "lrnn_state" tensors. Defaults to None.

        Returns:
            torch.Tensor: Output tensor, shape ``(B, L, D)``.
        """
        batch, seqlen, dim = hidden_states.shape

        # event mode: use asymmetric discretization
        use_event_mode = integration_timesteps is not None

        conv_state, ssm_state = None, None
        if inference_cache is not None:
            conv_state = inference_cache.get("conv_state")
            ssm_state = inference_cache.get("lrnn_state")
            seqlen_offset = inference_cache.get("seqlen_offset", 0)
            if seqlen_offset > 0:
                # Use step() for autoregressive decoding
                # inference_cache is updated in-place
                out, inference_cache = self.step(
                    hidden_states,
                    inference_cache,
                    integration_timesteps=integration_timesteps,
                )
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(
                self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1"
            )

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if (
            self.use_fast_path
            and causal_conv1d_fn is not None
            and inference_cache is None
            and not use_event_mode  # can't use fast path for event mode
        ):  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(x, (self.d_conv - x.shape[-1], 0))
                )  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

            B = x_dbl[:, self.dt_rank : self.dt_rank + self.d_state]
            B = rearrange(
                B, "(b l) dstate -> b dstate l", l=seqlen
            ).contiguous()

            C = x_dbl[:, -self.d_state :]
            C = rearrange(
                C, "(b l) dstate -> b dstate l", l=seqlen
            ).contiguous()

            # compute dt from x_dbl
            dt, _, _ = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

            if use_event_mode:
                # event mode: asymmetric discretization with separate dtA and dtB
                # compute dtA from dtA_proj bias scaled by integration timesteps
                dtA = repeat(
                    self.dtA_proj.bias, "d -> b d l", b=batch, l=seqlen
                )
                dtA = integration_timesteps.unsqueeze(1) * F.softplus(dtA)

                # apply softplus to dt (with bias) for B discretization
                dt = F.softplus(dt + self.dt_proj.bias.float()[:, None])

                # run selective scan with separate deltaA for asymmetric discretization
                assert self.activation in ["silu", "swish"]

                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=None,
                    deltaA=dtA,
                    delta_softplus=False,
                    return_last_state=ssm_state is not None,
                    discretization=self.discretization,
                )
            else:
                # standard Mamba mode
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                    discretization=self.discretization,
                )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(
        self,
        x: torch.Tensor,
        inference_cache: Dict[str, Any],
        integration_timesteps: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Performs a single recurrent step of Mamba.

        Args:
            x (torch.Tensor): Input at current timestep, shape ``(B, 1, D)``.
            inference_cache (Dict[str, Any]): Cache dictionary containing:
                - "conv_state": Convolution state, shape ``(B, D_inner, d_conv)``
                - "lrnn_state": SSM state, shape ``(B, D_inner, N)``
                - "seqlen_offset": Current position in sequence
            integration_timesteps (torch.Tensor, optional): Integration timestep,
                shape ``(B, 1)`` or ``(B,)``. When provided, uses event-based
                asymmetric discretization. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[torch.Tensor, Dict[str, Any]]: A tuple containing:
                - out (torch.Tensor): Output at current timestep, shape ``(B, 1, D)``.
                - inference_cache (Dict[str, Any]): Updated cache dictionary.
        """
        conv_state = inference_cache["conv_state"]
        ssm_state = inference_cache["lrnn_state"]
        hidden_states = x

        use_event_mode = integration_timesteps is not None

        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"),
                dim=-1,
            )  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        B = x_db[:, self.dt_rank : self.dt_rank + self.d_state]
        C = x_db[:, -self.d_state :]
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # compute dt for B matrix discretization
        dt = x_db[:, : self.dt_rank]
        # don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)

        # compute deltaA for event mode
        deltaA = None
        if use_event_mode:
            # compute dtA from dtA_proj bias scaled by integration timesteps
            dtA = self.dtA_proj.bias.expand(x.shape[0], -1)
            timestep = (
                integration_timesteps.view(-1, 1)
                if integration_timesteps.dim() > 1
                else integration_timesteps.unsqueeze(-1)
            )
            deltaA = timestep * F.softplus(dtA)

        if selective_state_update is not None:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=self.dt_proj.bias,
                dt_softplus=True,
                deltaA=deltaA,
                discretization=self.discretization,
            )
        else:
            dt_with_bias = F.softplus(
                dt + self.dt_proj.bias.to(dtype=dt.dtype)
            )

            # discretize A: use deltaA if provided, otherwise use dt
            if deltaA is not None:
                dA = torch.exp(torch.einsum("bd,dn->bdn", deltaA, A))
            else:
                dA = torch.exp(torch.einsum("bd,dn->bdn", dt_with_bias, A))

            # discretize B based on discretization method
            if self.discretization == "zoh":
                A_dt = torch.einsum("bd,dn->bdn", dt_with_bias, A)
                expm1_A_dt = torch.exp(A_dt) - 1.0
                B_tilde = expm1_A_dt / A.unsqueeze(0)
                dB = B.unsqueeze(1) * B_tilde
            elif self.discretization == "bilinear":
                v = 0.5 * torch.einsum("bd,dn->bdn", dt_with_bias, A)
                den_inv = 1.0 / (1.0 - v)
                dB = B.unsqueeze(1) * den_inv * dt_with_bias.unsqueeze(-1)
            elif self.discretization == "dirac":
                dB = B.unsqueeze(1).expand_as(dA)
            else:  # "mamba"
                dB = torch.einsum("bd,bn->bdn", dt_with_bias, B)

            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)

        out = self.out_proj(y)

        # Update cache in-place and return
        inference_cache["conv_state"] = conv_state
        inference_cache["lrnn_state"] = ssm_state
        inference_cache["seqlen_offset"] = (
            inference_cache.get("seqlen_offset", 0) + 1
        )

        return out.unsqueeze(1), inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Allocates cache for Mamba autoregressive inference.

        Args:
            batch_size (int): The batch size for inference.
            max_seqlen (int): Maximum sequence length (not used by Mamba,
                but kept for interface consistency).
            dtype (torch.dtype, optional): Data type for allocated tensors. Defaults to None.
            **kwargs: Additional arguments (unused).

        Returns:
            Dict[str, Any]: Cache dictionary containing:
                - "conv_state": Convolution state, shape ``(B, D_inner, d_conv)``.
                - "lrnn_state": SSM state, shape ``(B, D_inner, N)``.
                - "seqlen_offset": Current position in the sequence (starts at 0).
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_conv,
            device=device,
            dtype=conv_dtype,
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.d_model * self.expand,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {
            "conv_state": conv_state,
            "lrnn_state": ssm_state,
            "seqlen_offset": 0,
        }