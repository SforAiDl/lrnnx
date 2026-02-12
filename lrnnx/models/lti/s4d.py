"""
Taken from the original S4 implementation and modified to fit into the LRNNX framework.
https://github.com/state-spaces/s4
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lrnnx.core.convolution import FFTConvS4
from lrnnx.models.lti.base import LTI_LRNN
from lrnnx.ops.s4_utils import (
    DropoutNd,
    LinearActivation,
    init_dt,
    init_ssm_dplr,
    register_ssm_params,
)

contract = torch.einsum


class S4D(LTI_LRNN):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.

    Example
    -------
    >>> model = S4D(d_model=64, d_state=64, l_max=1024)
    >>> x = torch.randn(2, 1024, 64)
    >>> y = model(x)
    >>> y.shape
    torch.Size([2, 1024, 64])
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        gate=None,
        final_act="glu",
        postact=None,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        # Kernel/SSM configuration args
        l_max=None,
        channels=1,
        d_state=64,
        dt_min=0.001,
        dt_max=0.1,
        dt_tie=True,
        dt_transform="exp",
        dt_fast=False,
        rank=1,
        n_ssm=None,
        init="legs",
        deterministic=False,
        real_transform="exp",
        imag_transform="none",
        is_real=False,
        lr=None,
        wd=0.0,
        verbose=True,
        disc="zoh",  # S4D-specific: discretization method
        **layer_args,  # Any remaining args for FFTConv
    ):
        super().__init__(
            discretization="no_discretization"
        )  # discretization is unused in S4

        self.d_model = d_model
        self.transposed = transposed
        self.gate = gate
        self.bottleneck = bottleneck

        # Store config needed for kernel
        self.l_max = l_max
        self.channels = channels
        self.N = d_state
        self.dtype, self.cdtype = torch.float, torch.cfloat
        self.dt_fast = dt_fast
        self.real_transform = real_transform
        self.imag_transform = imag_transform
        self.is_real = is_real
        self.deterministic = deterministic
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_tie = dt_tie
        self.dt_transform = dt_transform
        self.rank = rank
        self.H = d_model
        self.n_ssm = n_ssm if n_ssm is not None else self.H
        self.init = init
        self.verbose = verbose
        self.disc = disc  # S4D-specific

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                activate=False,
            )

        # Initialize dt
        inv_dt = init_dt(
            self.H,
            self.N,
            self.dt_min,
            self.dt_max,
            self.dt_tie,
            self.dt_transform,
            self.deterministic,
            self.dtype,
        )

        _fft_conv_keys = {
            "kernel",
            "swap_channels",
            "drop_kernel",  # FFTConvS4
            "d_model",
            "l_max",
            "channels",
            "transposed",  # also FFTConvS4
            "d_state",
            "dt_min",
            "dt_max",
            "dt_tie",
            "dt_transform",  # S4KernelBase (dt)
            "dt_fast",
            "rank",
            "n_ssm",
            "init",  # S4KernelBase (A,B,C)
            "real_transform",
            "imag_transform",
            "is_real",  # S4KernelBase (transforms)
            "deterministic",
            "verbose",
            "lr",
            "wd",  # S4KernelBase (misc)
            "disc",  # S4DKernel
        }
        init_args = {
            k: v for k, v in layer_args.items() if k not in _fft_conv_keys
        }

        # Initialize A, P, B, C
        A, P, B, C = init_ssm_dplr(
            self.N,
            self.H,
            self.n_ssm,
            self.channels,
            self.rank,
            self.init,
            self.deterministic,
            self.cdtype,
            **init_args,
        )

        # Halve N for conjugate symmetry
        self.N //= 2

        # Register parameters on THIS module (S4D), not on kernel
        # Note: For S4D we use diag=True (no P parameter)
        self.repeat = register_ssm_params(
            self,  # Register on S4D module
            A,
            B,
            C,
            inv_dt,
            P,  # P is ignored when diag=True
            self.H,
            self.n_ssm,
            self.N,
            self.channels,
            self.rank,
            self.dt_fast,
            self.real_transform,
            self.imag_transform,
            self.is_real,
            self.verbose,
            self.l_max,
            diag=True,  # S4D uses diagonal (not DPLR)
        )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                activate=True,
            )

        # D parameter and convolution activation
        self.D = nn.Parameter(torch.randn(channels, d_model))
        self.conv_activation = nn.GELU()

        # Create FFTConvS4 with parameter references
        self.layer = FFTConvS4(
            d_model,
            l_max=l_max,
            channels=channels,
            transposed=False,
            dropout=dropout,
            tie_dropout=tie_dropout,
            kernel_type="s4d",
            param_config={
                "A_real": self.A_real,
                "A_imag": self.A_imag if not self.is_real else None,
                "B": self.B,
                "C": self.C,
                "P": None,  # S4D doesn't use P
                "inv_dt": self.inv_dt,
                "N": self.N,
                "H": self.H,
                "channels": self.channels,
                "rank": self.rank,
                "repeat": self.repeat,
                "dt_fast": self.dt_fast,
                "real_transform": self.real_transform,
                "imag_transform": self.imag_transform,
                "dt_transform": self.dt_transform,
                "is_real": self.is_real,
                "deterministic": self.deterministic,
                "verbose": self.verbose,
                "disc": self.disc,  # S4D-specific
            },
            **layer_args,
        )

        # Check if we need output_gate
        if gate is not None:
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model * self.channels,
                    self.d_model * gate,
                    transposed=False,
                    activate=False,
                )

        # Pointwise operations
        self.mult_activation = nn.GELU()
        dropout_fn = (
            partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        )
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            print(
                "Warning: 'postact' option changed to 'final_act' and will be removed in a future version."
            )
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                (
                    self.d_model * gate
                    if gate is not None
                    else self.layer.d_output
                ),
                self.d_model,
                transposed=False,
                activate=True,
            )

    def forward(
        self, x, lengths=None, **kwargs
    ):  # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        if self.transposed:
            x = rearrange(x, "b d ... -> b ... d")
        L = x.size(1)

        # Mask out padding tokens
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(
                    lengths, dtype=torch.long, device=x.device
                )
            else:
                lengths = None
        if lengths is not None:
            assert (
                isinstance(lengths, torch.Tensor)
                and lengths.ndim == 1
                and lengths.size(0) in [1, x.size(0)]
            )
            mask = torch.where(
                torch.arange(L, device=lengths.device)[:, None]
                < lengths[:, None, None],
                1.0,
                0.0,
            )
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(
            x, **kwargs
        )  # (B C H L) in transposed=False mode

        # Post-convolution operations
        # Add D term
        x_for_D = x.transpose(-1, -2)  # (B L H) -> (B H L)
        y = y + contract("bhl,ch->bchl", x_for_D, self.D)

        # Reshape to flatten channels
        if self.layer.swap_channels:
            y = rearrange(y, "b c h l -> b (h c) l")
        else:
            y = rearrange(y, "b c h l -> b (c h) l")

        # Transpose back to (B L H) format
        y = y.transpose(-1, -2)  # (B C*H L) -> (B L C*H)

        # Apply convolution activation
        y = self.conv_activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        if self.transposed:
            y = rearrange(y, "b d ... -> b ... d")

        return y, state

    def step(
        self,
        x: torch.Tensor,
        inference_cache: dict,
        **kwargs,
    ) -> tuple:
        """Perform a single recurrent step of the S4D model.

        Args
        ----
            x (Tensor): Input at current timestep, shape (B, H).
            inference_cache (Dict[str, Any]): Cache from allocate_inference_cache().

        Returns
        -------
            Tuple[Tensor, Dict[str, Any]]: Output y_t of shape (B, H) and updated cache.
        """
        state = inference_cache["lrnn_state"]

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state)  # (B C H)

        # Post-convolution operations
        # Add D term
        y = y + x.unsqueeze(-2) * self.D
        # Reshape to flatten channels
        y = rearrange(y, "b c h -> b (c h)")
        # Apply convolution activation
        y = self.conv_activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        inference_cache["lrnn_state"].copy_(next_state)
        return y, inference_cache

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int = 1,
        dtype=None,
        **kwargs,
    ) -> dict:
        """Allocate cache for step-by-step inference.

        Calls setup_step() to prepare discrete-time matrices (dA, dB, dC),
        then creates a zero-initialised hidden state.

        Args
        ----
            batch_size (int): Batch size for inference.
            max_seqlen (int): Unused, kept for interface consistency.

        Returns
        -------
            Dict[str, Any]: Cache dict with "lrnn_state" key.
        """
        self.layer.setup_step()
        state = self.default_state(
            batch_size, device=next(self.parameters()).device
        )
        return {"lrnn_state": state}

    def default_state(self, *batch_shape, device=None):
        return self.layer.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model
