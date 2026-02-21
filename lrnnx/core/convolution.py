"""
FFT convolution with optimized einsum contractions.
Ref.: https://arxiv.org/abs/2409.03377
"""

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from lrnnx.ops.s4_kernel_interface import kernel_registry
from lrnnx.ops.s4_utils import DropoutNd

contract = torch.einsum


@torch.compiler.disable
def fft_conv(equation: str, input: Tensor, *args) -> Tensor:
    """
    FFT based convolution operation.

    Args:
        equation (str): Einsum equation for the convolution.
        input (torch.Tensor): Input tensor, shape ``(B, L, H)`` or ``(B, L, N)``.
        *args: Either single kernel ``(L, H, H)`` or ``(K, B_norm / B_bar, C)`` tensors.

    Returns:
        torch.Tensor: Convolved output tensor, shape ``(B, L, H)`` or ``(B, L, N)``.
    """
    L = input.shape[1]

    input_f = torch.fft.fft(input, 2 * L, dim=1)  # (B, 2L, H)
    args = tuple(arg.cfloat() for arg in args)

    if len(args) == 1:
        kernel = args[0]
        kernel_f = torch.fft.fft(kernel, 2 * L, dim=0)  # (2L, H, H)
        output_f = torch.einsum(equation, input_f, kernel_f)  # (B, 2L, H)

    else:
        K, B_norm, C = args
        K_f = torch.fft.fft(K, 2 * L, dim=1)  # (N, 2L)
        output_f = torch.einsum(
            equation, input_f, K_f, B_norm, C
        )  # (B, 2L, H)

    output = torch.fft.ifft(output_f, dim=1)  # (B, 2L, H)
    return output[:, :L, :]  # (B, L, H)


def opt_ssm_forward(x: Tensor, K: Tensor, B_: Tensor, C: Tensor) -> Tensor:
    """
    Optimized FFT convolution.

    Args:
        x (torch.Tensor): Input tensor, shape ``(B, L, H)``.
        K (torch.Tensor): Kernel tensor, shape ``(L, H, H)`` or ``(L, N)``.
        B_ (torch.Tensor): Normalized input projection matrix, shape ``(N, H)``.
        C (torch.Tensor): Output projection matrix, shape ``(H, N)``.

    Returns:
        torch.Tensor: Output tensor, shape ``(B, L, H)``.
    """
    B, _, H_in = x.shape
    H_out, N = C.shape

    if (1 / H_in + 1 / H_out) > (1 / B + 1 / N):
        if H_in * H_out <= N:
            # strategy 1
            kernel = torch.einsum("on,nl,ni->loi", C, K, B_).real  # (L, H, H)
            return fft_conv("bli,loi->blo", x, kernel).real  # (B, L, H)
    else:
        if N <= H_in:
            # strategy 2
            x_proj = torch.einsum(
                "blh,nh->bln", x.to(B_.dtype), B_
            )  # (B, L, N)
            x_conv = fft_conv("bln,ln->bln", x_proj, K.T)  # (B, L, N)
            return torch.einsum("bln,hn->blh", x_conv, C).real  # (B, L, H)

    # fallback
    return fft_conv("blh,nl,nh,on->blo", x, K, B_, C).real  # (B, L, H)


class FFTConvS4(nn.Module):
    """Implements an FFT Convolution around a convolution kernel."""

    def __init__(
        self,
        d_model,
        l_max=None,
        channels=1,
        swap_channels=False,
        transposed=True,
        dropout=0.0,
        tie_dropout=False,
        drop_kernel=0.0,
        kernel_type=None,
        param_config=None,
        kernel=None,
        **kernel_args,
    ):
        """
        Initialize FFTConvS4.

        Args:
            d_model (int): Model dimension (in CNN terminology, "channels").
            l_max (int, optional): Maximum kernel length. ``None`` for a global kernel. Defaults to None.
            channels (int, optional): Number of "heads"; SSM maps 1-dim to C-dim. Defaults to 1.
            swap_channels (bool, optional): Whether to swap channel ordering. Defaults to False.
            transposed (bool, optional): Backbone axis ordering. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            tie_dropout (bool, optional): Tie dropout mask across sequence length. Defaults to False.
            drop_kernel (float, optional): Kernel dropout probability. Defaults to 0.0.
            kernel_type (str, optional): Kernel algorithm (``'s4'`` for DPLR, ``'s4d'`` for diagonal). Defaults to None.
            param_config (dict, optional): References to SSM parameters (A, B, C, dt, P, etc.). Defaults to None.
            kernel (str, optional): Alternative kernel specification. Defaults to None.
            **kernel_args: Additional arguments forwarded to the kernel class.
        """
        super().__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.channels = channels
        self.transposed = transposed
        self.swap_channels = swap_channels

        if param_config is not None:
            if kernel_type is None:
                raise ValueError(
                    "kernel_type must be provided with param_config"
                )

            kernel_cls = kernel_registry[kernel_type]
            self.kernel = kernel_cls(
                d_model=d_model,
                l_max=l_max,
                channels=channels,
                param_config=param_config,
            )
        else:
            if kernel is None:
                raise ValueError(
                    "Either param_config or kernel must be provided"
                )

            kernel_cls = kernel_registry[kernel]
            self.kernel = kernel_cls(
                d_model=self.d_model,
                l_max=self.l_max,
                channels=channels,
                **kernel_args,
            )

        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.drop_kernel = (
            nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()
        )

    def forward(
        self, x, state=None, rate=1.0, **kwargs
    ):  # absorbs return_output and transformer src mask
        """
        Forward pass through FFTConvS4.

        Args:
            x (torch.Tensor): Input tensor, shape ``(B, D, L)`` if ``self.transposed``
                else ``(B, L, D)``.
            state (torch.Tensor, optional): Recurrent state. Defaults to None.
            rate (float, optional): Rate for kernel computation. Defaults to 1.0.
            **kwargs: Additional keyword arguments (absorbs return_output, src mask, etc.).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - y : Convolution output, shape ``(B, C, H, L)``.
                - next_state : State for recurrent mode, or ``None``.
        """

        # Always work with (B D L) dimension in this module
        if not self.transposed:
            x = x.transpose(-1, -2)
        L = x.size(-1)

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state = self.kernel(
            L=l_kernel, rate=rate, state=state
        )  # (C H L) (B C H L)

        # Kernel dropout
        k = self.drop_kernel(k)

        # FFT convolution (core operation)
        k_f = torch.fft.rfft(k, n=l_kernel + L)  # (C H L)
        x_f = torch.fft.rfft(x, n=l_kernel + L)  # (B H L)
        y_f = contract("bhl,chl->bchl", x_f, k_f)
        y = torch.fft.irfft(y_f, n=l_kernel + L)[..., :L]  # (B C H L)

        # Compute state update
        if state is not None:
            y = y + k_state
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None

        return y, next_state

    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, state):
        """
        Step one time step as a recurrent model.

        Intended to be used during validation.

        Args:
            x (torch.Tensor): Input tensor, shape ``(B, H)``.
            state (torch.Tensor): Recurrent state, shape ``(B, H, N)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - y : Output, shape ``(B, C, H)``.
                - next_state : Updated state, shape ``(B, H, N)``.
        """

        y, next_state = self.kernel.step(x, state)  # (B C H)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_output(self):
        return self.d_model * self.channels
