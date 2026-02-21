"""
Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mha.py
Copyright (c) 2024, Tri Dao.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


def _update_kv_cache(kv, inference_params, layer_idx):
    """
    Update key-value cache for a specific layer during inference.

    Args:
        kv (torch.Tensor): Key-value tensor of shape ``(batch_size, seqlen, 2, num_heads_kv, head_dim)``
            or ``(batch_size, 1, 2, num_heads_kv, head_dim)`` for single token updates.
        inference_params (Any): Inference parameters containing cache information and offsets.
        layer_idx (int): Index of the layer whose cache is being updated.

    Returns:
        torch.Tensor: Updated key-value cache of shape ``(batch_size, current_seqlen, 2, num_heads_kv, head_dim)``.
    """
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """
    Multi-head self-attention and cross-attention with optional convolution and rotary embeddings.

    This implementation supports various attention configurations including:
    
    * Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
    * Optional 1D convolution for local context
    * Rotary position embeddings
    * Integrated MLP for efficiency
    * Causal and non-causal attention

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        num_heads_kv (int, optional): Number of key-value heads for MQA/GQA.
            If None, uses num_heads (standard MHA). Defaults to None.
        head_dim (int, optional): Dimension per head. If None, uses ``embed_dim // num_heads``. Defaults to None.
        mlp_dim (int, optional): Dimension of integrated MLP. If 0, no MLP is used. Defaults to 0.
        qkv_proj_bias (bool, optional): Whether to use bias in QKV projection. Defaults to True.
        out_proj_bias (bool, optional): Whether to use bias in output projection. Defaults to True.
        softmax_scale (float, optional): Scale factor for attention scores.
            If None, uses ``1/sqrt(head_dim)``. Defaults to None.
        causal (bool, optional): Whether to use causal (masked) attention. Defaults to False.
        layer_idx (int, optional): Layer index for caching during inference. Defaults to None.
        d_conv (int, optional): Kernel size for 1D convolution. If 0, no convolution. Defaults to 0.
        rotary_emb_dim (int, optional): Dimension for rotary embeddings. If 0, no rotary embeddings. Defaults to 0.
        rotary_emb_base (float, optional): Base for rotary embeddings. Defaults to 10000.0.
        rotary_emb_interleaved (bool, optional): Whether to use interleaved rotary embeddings. Defaults to False.
        device (torch.device, optional): Device to place tensors on. Defaults to None.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to None.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,  # If None, use embed_dim // num_heads
        mlp_dim=0,
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        d_conv=0,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.causal = causal

        self.num_heads = num_heads
        self.num_heads_kv = (
            num_heads_kv if num_heads_kv is not None else num_heads
        )
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert (
                self.embed_dim % num_heads == 0
            ), "embed_dim must be divisible by num_heads"
        self.head_dim = (
            head_dim if head_dim is not None else self.embed_dim // num_heads
        )
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        if self.rotary_emb_dim > 0:
            assert (
                RotaryEmbedding is not None
            ), "rotary requires flash_attn to be installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.in_proj = nn.Linear(
            embed_dim,
            qkv_dim + self.mlp_dim,
            bias=qkv_proj_bias,
            **factory_kwargs,
        )
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                qkv_dim,
                qkv_dim,
                kernel_size=self.d_conv,
                padding=self.d_conv - 1,
                groups=qkv_dim,
                **factory_kwargs,
            )
        self.out_proj = nn.Linear(
            out_dim + self.mlp_dim // 2,
            embed_dim,
            bias=out_proj_bias,
            **factory_kwargs,
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate inference cache for key-value states and convolution states.

        Args:
            batch_size (int): Batch size for inference.
            max_seqlen (int): Maximum sequence length for inference.
            dtype (torch.dtype, optional): Data type for cache tensors.
                If None, uses output projection weight dtype. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: A tuple containing:
                - kv_cache : Tensor of shape ``(batch_size, max_seqlen, 2, num_heads_kv, head_dim)``.
                - conv_state : Tensor of shape ``(batch_size, qkv_dim, d_conv)`` or None if ``d_conv=0``.
        """
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        if self.d_conv > 0:
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=device,
                dtype=dtype,
            )
        else:
            conv_state = None
        kv_cache = torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )
        return kv_cache, conv_state

    def _update_kv_cache(self, kv, inference_params):
        """
        Update key-value cache during inference.

        Args:
            kv (torch.Tensor): Key-value tensor of shape ``(batch_size, seqlen, 2, num_heads_kv, head_dim)``.
            inference_params (Any): Inference parameters containing cache information.

        Returns:
            torch.Tensor: Updated key-value cache.
        """
        assert (
            self.layer_idx is not None
        ), "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combines rotary embedding, KV cache update, and attention.

        This is an optimized implementation that performs three operations in one step:
        
        1. Apply rotary position embeddings to queries and keys
        2. Update the key-value cache
        3. Compute attention with the updated cache

        Args:
            q (torch.Tensor): Query tensor of shape ``(batch_size, seqlen_q, num_heads, head_dim)``.
            kv (torch.Tensor): Key-value tensor of shape ``(batch_size, seqlen_k, 2, num_heads_kv, head_dim)``.
            inference_params (Any): Inference parameters containing cache and sequence information.

        Returns:
            torch.Tensor: Attention output of shape ``(batch_size, seqlen_q, num_heads, head_dim)``.
        """
        assert (
            inference_params is not None and inference_params.seqlen_offset > 0
        )
        if self.rotary_emb_dim > 0:
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
        kv_cache = kv_cache[:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        assert (
            flash_attn_with_kvcache is not None
        ), "flash_attn must be installed"
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            rotary_interleaved=(
                self.rotary_emb.interleaved
                if self.rotary_emb_dim > 0
                else False
            ),
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """
        Update key-value cache and compute attention.

        This method handles both initial caching (when ``seqlen_offset=0``) and
        incremental updates during autoregressive generation.

        Args:
            q (torch.Tensor): Query tensor of shape ``(batch_size, seqlen_q, num_heads, head_dim)``.
            kv (torch.Tensor): Key-value tensor of shape ``(batch_size, seqlen_k, 2, num_heads_kv, head_dim)``.
            inference_params (Any): Inference parameters containing cache and sequence information.

        Returns:
            torch.Tensor: Attention output of shape ``(batch_size, seqlen_q, num_heads, head_dim)``.
        """
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            k, v = kv.unbind(dim=-3)
            k = torch.repeat_interleave(
                k, dim=2, repeats=self.num_heads // self.num_heads_kv
            )
            v = torch.repeat_interleave(
                v, dim=2, repeats=self.num_heads // self.num_heads_kv
            )
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=self.causal,
                scale=self.softmax_scale,
            ).transpose(1, 2)
        else:
            batch = q.shape[0]
            kv_cache, _ = inference_params.key_value_memory_dict[
                self.layer_idx
            ]
            kv_cache = kv_cache[:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

    def forward(self, x, inference_params=None):
        """
        Forward pass of Multi-Head Attention.

        Performs the complete attention computation including:
        
        1. QKV projection with optional integrated MLP
        2. Optional 1D convolution for local context
        3. Rotary position embedding (if enabled)
        4. Multi-head attention computation
        5. Output projection

        Note:
            During inference, this method automatically allocates and manages
            key-value caches for efficient autoregressive generation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, seq_len, embed_dim)``.
            inference_params (Any, optional): Parameters for inference mode containing:
                - key_value_memory_dict: Cache for storing KV states.
                - seqlen_offset: Current sequence position offset.
                - max_seqlen: Maximum sequence length.
                - lengths_per_sample: Per-sample sequence lengths (optional). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, seq_len, embed_dim)``.
        """
        if (
            inference_params is not None
            and self.layer_idx not in inference_params.key_value_memory_dict
        ):
            inference_params.key_value_memory_dict[self.layer_idx] = (
                self.allocate_inference_cache(
                    x.shape[0], inference_params.max_seqlen, dtype=x.dtype
                )
            )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = (
            inference_params.max_seqlen
            if inference_params is not None
            else None
        )
        qkv = self.in_proj(x)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split(
                [qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1
            )
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        if self.d_conv > 0:
            # The inference code for conv1d is pretty messy, should clean it up
            if inference_params is None or inference_params.seqlen_offset == 0:
                if causal_conv1d_fn is None:
                    qkv = rearrange(
                        self.conv1d(rearrange(qkv, "b s d -> b d s"))[
                            ..., : -(self.d_conv - 1)
                        ],
                        "b d s -> b s d",
                    ).contiguous()
                else:
                    qkv = causal_conv1d_fn(
                        qkv.transpose(1, 2),
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias,
                    ).transpose(1, 2)
                if inference_params is not None:
                    _, conv_state = inference_params.key_value_memory_dict[
                        self.layer_idx
                    ]
                    # If we just take qkv[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    qkv_t = rearrange(qkv, "b l d -> b d l")
                    conv_state.copy_(
                        F.pad(qkv_t, (self.d_conv - qkv_t.shape[-1], 0))
                    )  # Update state (B D W)
            else:
                _, conv_state = inference_params.key_value_memory_dict[
                    self.layer_idx
                ]
                assert (
                    qkv.shape[1] == 1
                ), "Only support decoding with 1 token at a time for now"
                qkv = qkv.squeeze(1)
                # Conv step
                if causal_conv1d_update is None:
                    conv_state.copy_(
                        torch.roll(conv_state, shifts=-1, dims=-1)
                    )  # Update state (B D W)
                    conv_state[:, :, -1] = qkv
                    qkv = torch.sum(
                        conv_state
                        * rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        dim=-1,
                    )  # (B D)
                    if self.conv1d.bias is not None:
                        qkv = qkv + self.conv1d.bias
                else:
                    qkv = causal_conv1d_update(
                        qkv,
                        conv_state,
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias,
                    )
                qkv = qkv.unsqueeze(1)
        q, kv = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_heads_kv * 2 * self.head_dim,
            ],
            dim=-1,
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(
            kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim
        )
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q,
                    kv,
                    seqlen_offset=seqlen_offset,
                    max_seqlen=rotary_max_seqlen,
                )
            if inference_params is None:
                k, v = kv.unbind(dim=-3)
                k = torch.repeat_interleave(
                    k, dim=2, repeats=self.num_heads // self.num_heads_kv
                )
                v = torch.repeat_interleave(
                    v, dim=2, repeats=self.num_heads // self.num_heads_kv
                )
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=self.causal,
                    scale=self.softmax_scale,
                ).transpose(1, 2)
            else:
                context = self._update_kvcache_attention(
                    q, kv, inference_params
                )
        else:
            context = self._apply_rotary_update_kvcache_attention(
                q, kv, inference_params
            )
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out