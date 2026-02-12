"""
Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py
Copyright (c) 2024, Tri Dao.
"""

from typing import Optional

import torch
from torch import Tensor, nn

if torch.cuda.is_available():
    from lrnnx.ops.triton.layer_norm import RMSNorm, layer_norm_fn
else:
    RMSNorm, layer_norm_fn = nn.RMSNorm, None


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        mlp_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=True,
        residual_in_fp32=False,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        This Block has a slightly different structure compared to a regular prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params=None,
        **mixer_kwargs,
    ):
        """Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual)
                if residual is not None
                else hidden_states
            )
            hidden_states = self.norm(
                residual.to(dtype=self.norm.weight.dtype)
            )
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )

        # TODO: inference params?
        try:
            hidden_states = self.mixer(
                hidden_states,
                inference_params=inference_params,
                **mixer_kwargs,
            )
        except TypeError:
            # fallback for now
            hidden_states = self.mixer(hidden_states, **mixer_kwargs)

        # some mixers (e.g. S4, S4D) return (output, state) tuples
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(
                    residual.to(dtype=self.norm2.weight.dtype)
                )
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(
        self, batch_size, max_seqlen, dtype=None, **kwargs
    ):
        """Allocate inference cache for the mixer."""
        try:
            return self.mixer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            )
        except TypeError:
            return self.mixer.allocate_inference_cache(batch_size, **kwargs)
