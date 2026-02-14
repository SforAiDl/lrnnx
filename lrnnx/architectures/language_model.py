"""
Language Model architecture.
Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
"""

import inspect
import json
import math
import os
from collections import namedtuple
from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from lrnnx.layers.block import Block
from lrnnx.layers.mha import MHA
from lrnnx.layers.mlp import GatedMLP

if torch.cuda.is_available():
    from lrnnx.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
else:
    RMSNorm, layer_norm_fn, rms_norm_fn = nn.RMSNorm, None, None


def _get_mixer_class_from_string(mixer_type: str):
    """
    Get mixer class from string name.

    Args:
        mixer_type (str): Name of the mixer type (e.g., "LRU", "S5", "S6", "S7", 
            "Stream", "Centaurus", "Mamba", "attn").

    Returns:
        type: The corresponding PyTorch neural network class.
    """
    from lrnnx.models.lti.centaurus import Centaurus
    from lrnnx.models.lti.lru import LRU
    from lrnnx.models.lti.s4 import S4
    from lrnnx.models.lti.s4d import S4D
    from lrnnx.models.lti.s5 import S5
    from lrnnx.models.ltv.mamba import Mamba
    from lrnnx.models.ltv.rglru import RGLRU
    from lrnnx.models.ltv.s7 import S7

    mixer_registry = {
        "LRU": LRU,
        "S4": S4,
        "S4D": S4D,
        "S5": S5,
        "Centaurus": Centaurus,
        "Mamba": Mamba,
        "RGLRU": RGLRU,
        "S7": S7,
    }

    if mixer_type == "attn":
        return "attn"

    if mixer_type not in mixer_registry:
        raise ValueError(
            f"Unknown mixer type: {mixer_type}. "
            f"Available types: {list(mixer_registry.keys()) + ['attn']}"
        )

    return mixer_registry[mixer_type]


def create_block(
    d_model: int,
    d_state: int,
    d_intermediate: int,
    mixer_type: str,
    mixer_kwargs: Optional[Dict] = None,
    attn_cfg: Optional[Dict] = None,
    norm_epsilon: float = 1e-5,
    rms_norm: bool = False,
    residual_in_fp32: bool = False,
    fused_add_norm: bool = True,
    layer_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Block:
    """
        Create a block.

        :param d_model: Model dimension
        :type d_model: int
        :param d_state: State dimension
        :type d_state: int
        :param d_intermediate: Intermediate dimension for MLP layers (0 to disable MLP)
        :type d_intermediate: int
        :param mixer_type: Name of the mixer type (e.g., "LRU", "S5", "attn")
        :type mixer_type: str
        :param mixer_kwargs: Additional arguments for mixer
        :type mixer_kwargs: dict, optional
        :param attn_cfg: Configuration for attention layers
        :type attn_cfg: dict, optional
        :param norm_epsilon: Epsilon value for layer normalization
        :type norm_epsilon: float
        :param rms_norm: Whether to use RMSNorm instead of LayerNorm
        :type rms_norm: bool
        :param residual_in_fp32: Whether to compute residuals in float32
        :type residual_in_fp32: bool
        :param fused_add_norm: Whether to use fused add+norm operations
        :type fused_add_norm: bool
        :param layer_idx: Index of the current layer
        :type layer_idx: int, optional
        :param device: Device to place tensors on
        :type device: torch.device, optional
        :param dtype: Data type for tensors
        :type dtype: torch.dtype, optional
        :return: A configured block module
        :rtype: Block
    """
    if attn_cfg is None:
        attn_cfg = {}
    if mixer_kwargs is None:
        mixer_kwargs = {}

    factory_kwargs = {"device": device, "dtype": dtype}

    if mixer_type == "attn":
        mixer_cls = partial(
            MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs
        )
    else:
        mixer_cls_from_str = _get_mixer_class_from_string(mixer_type)
        # Only pass d_state if the mixer class accepts it
        sig = inspect.signature(mixer_cls_from_str)
        if "d_state" in sig.parameters:
            mixer_cls = partial(
                mixer_cls_from_str, d_state=d_state, **mixer_kwargs
            )
        else:
            mixer_cls = partial(mixer_cls_from_str, **mixer_kwargs)

    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs,
    )

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,  # change to 2 if we have MLP
) -> None:
    """
    Initialize weights following GPT-2 scheme.

    :param module: Module to initialize
    :type module: nn.Module
    :param n_layer: Number of layers in the model
    :type n_layer: int
    :param initializer_range: Standard deviation for weight initialization
    :type initializer_range: float
    :param rescale_prenorm_residual: Whether to rescale prenorm residual weights
    :type rescale_prenorm_residual: bool
    :param n_residuals_per_layer: Number of residual connections per layer
    :type n_residuals_per_layer: int
    """
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class LRNNModel(nn.Module):
    """
    Core LRNN backbone.
    
    :param d_model: Model dimension
    :type d_model: int
    :param d_state: State dimension
    :type d_state: int
    :param n_layer: Number of layers in the model
    :type n_layer: int
    :param vocab_size: Size of the vocabulary
    :type vocab_size: int
    :param mixer_types: List of mixer type names for each layer (e.g., ["S5", "S7", "attn", ...])
    :type mixer_types: list
    :param d_intermediate: Intermediate dimension for MLP layers (0 to disable MLP)
    :type d_intermediate: int
    :param mixer_kwargs: Additional arguments for mixer.
        Should be a dict mapping mixer type names to their kwargs,
        e.g., `{"S5": {"dt_min": 0.001}, "attn": {"num_heads": 8}}`.
        If a single dict is provided without mixer type keys, it will be applied to all mixers.
    :type mixer_kwargs: dict, optional
    :param mlp_cls: MLP class to use
    :type mlp_cls: class, optional
    :param norm_epsilon: Epsilon value for layer normalization
    :type norm_epsilon: float
    :param rms_norm: Whether to use RMSNorm instead of LayerNorm
    :type rms_norm: bool
    :param initializer_cfg: Configuration for weight initialization
    :type initializer_cfg: dict, optional
    :param fused_add_norm: Whether to use fused add+norm operations
    :type fused_add_norm: bool
    :param residual_in_fp32: Whether to compute residuals in float32
    :type residual_in_fp32: bool
    :param device: Device to place tensors on
    :type device: torch.device, optional
    :param dtype: Data type for tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_layer: int,
        vocab_size: int,
        mixer_types: list,
        d_intermediate: int = 0,
        mixer_kwargs: Optional[Dict] = None,
        mlp_cls=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg: Optional[Dict[str, Any]] = None,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError(
                    "Failed to import Triton LayerNorm / RMSNorm kernels"
                )

        # mixer_types should have n_layer entries
        if len(mixer_types) != n_layer:
            raise ValueError(
                f"mixer_types must have length n_layer ({n_layer}), "
                f"got {len(mixer_types)}"
            )

        if mixer_kwargs is None:
            mixer_kwargs = {}
        if mlp_cls is None:
            mlp_cls = GatedMLP

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # LRNN layers
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_state=d_state,
                    d_intermediate=d_intermediate,
                    mixer_type=mixer_types[i],
                    mixer_kwargs=(
                        mixer_kwargs.get(mixer_types[i], {})
                        if isinstance(mixer_kwargs, dict)
                        else {}
                    ),
                    attn_cfg=(
                        mixer_kwargs.get("attn", {})
                        if isinstance(mixer_kwargs, dict)
                        and mixer_types[i] == "attn"
                        else {}
                    ),
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        # normalization
        norm_cls = RMSNorm if rms_norm else nn.LayerNorm
        self.norm_f = norm_cls(d_model, eps=norm_epsilon, **factory_kwargs)

        # initialize weights
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=(
                    1 if d_intermediate == 0 else 2
                ),  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict:
        """
        Allocate inference cache for autoregressive generation.

        :param batch_size: Batch size for inference
        :type batch_size: int
        :param max_seqlen: Maximum sequence length for inference
        :type max_seqlen: int
        :param dtype: Data type for cache tensors
        :type dtype: torch.dtype, optional
        :param kwargs: Additional keyword arguments (e.g., for specific mixer types)
        :return: Dictionary mapping layer indices to their allocated caches
        :rtype: dict
        """
        cache = {}
        for i, layer in enumerate(self.layers):
            try:
                cache[i] = layer.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype, **kwargs
                )
            except TypeError:
                # fallback for now
                cache[i] = layer.allocate_inference_cache(batch_size, **kwargs)
        return cache

    def step(
        self,
        input_ids: Tensor,
        caches: Dict,
        integration_timesteps: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Single-step inference for autoregressive generation.

        :param input_ids: Input token IDs of shape (B, 1) - single token
        :type input_ids: Tensor
        :param caches: Dictionary mapping layer indices to their cached states
        :type caches: Dict
        :param integration_timesteps: Integration timesteps for LTV models (shape: B, 1 or B)
        :type integration_timesteps: Tensor, optional
        :return: Hidden states of shape (B, 1, d_model)
        :rtype: Tensor
        """
        hidden_states = self.embedding(input_ids)
        residual = None

        for i, layer in enumerate(self.layers):
            layer_cache = caches.get(i)

            # norm
            if not self.fused_add_norm:
                residual = (
                    (hidden_states + residual)
                    if residual is not None
                    else hidden_states
                )
                normed = layer.norm(residual.to(dtype=layer.norm.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                normed, residual = layer_norm_fn(
                    hidden_states,
                    layer.norm.weight,
                    layer.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=layer.norm.eps,
                    is_rms_norm=isinstance(layer.norm, RMSNorm),
                )

            # check LTV/LTI
            from lrnnx.models.ltv.base import LTV_LRNN

            if isinstance(layer.mixer, LTV_LRNN):
                mixer_out, updated_cache = layer.mixer.step(
                    normed,
                    layer_cache,
                    integration_timesteps=integration_timesteps,
                )
                caches[i] = updated_cache
                hidden_states = mixer_out
            elif isinstance(layer_cache, dict):
                # Dict-based cache (S4, S4D, Centaurus, S5, LRU)
                mixer_out, updated_cache = layer.mixer.step(
                    normed.squeeze(1), layer_cache
                )
                caches[i] = updated_cache
                hidden_states = mixer_out.unsqueeze(1)  # (B, H) -> (B, 1, H)

            if layer.mlp is not None:
                if not self.fused_add_norm:
                    residual = hidden_states + residual
                    normed2 = layer.norm2(
                        residual.to(dtype=layer.norm2.weight.dtype)
                    )
                    if self.residual_in_fp32:
                        residual = residual.to(torch.float32)
                else:
                    normed2, residual = layer_norm_fn(
                        hidden_states,
                        layer.norm2.weight,
                        layer.norm2.bias,
                        residual=residual,
                        prenorm=True,
                        residual_in_fp32=self.residual_in_fp32,
                        eps=layer.norm2.eps,
                        is_rms_norm=isinstance(layer.norm2, RMSNorm),
                    )
                hidden_states = layer.mlp(normed2)

        # norm
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual)
                if residual is not None
                else hidden_states
            )
            hidden_states = self.norm_f(
                residual.to(dtype=self.norm_f.weight.dtype)
            )
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        inference_params: Optional[Dict] = None,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        **mixer_kwargs,
    ) -> Tensor:
        """
        Forward pass of the LRNN backbone.

        :param input_ids: Input token IDs of shape (B, L)
        :type input_ids: Tensor
        :param inference_params: Parameters for inference mode
        :type inference_params: Dict, optional
        :param integration_timesteps: Timesteps for LTV models (shape: B, L)
        :type integration_timesteps: Tensor, optional
        :param lengths: Sequence lengths for variable-length sequences (shape: B)
        :type lengths: Tensor, optional
        :param mixer_kwargs: Additional keyword arguments passed to mixer layers
        :return: Hidden states of shape (B, L, d_model)
        :rtype: Tensor
        """
        hidden_states = self.embedding(input_ids)
        residual = None

        if integration_timesteps is not None:
            mixer_kwargs["integration_timesteps"] = integration_timesteps
        if lengths is not None:
            mixer_kwargs["lengths"] = lengths

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                **mixer_kwargs,
            )

        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual)
                if residual is not None
                else hidden_states
            )
            hidden_states = self.norm_f(
                residual.to(dtype=self.norm_f.weight.dtype)
            )
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        return hidden_states


class LRNNLMHeadModel(nn.Module):
    """
    LRNN Language Model with a language modeling head.
    
    :param d_model: Model dimension
    :type d_model: int
    :param d_state: State dimension
    :type d_state: int
    :param n_layer: Number of layers in the model
    :type n_layer: int
    :param vocab_size: Size of the vocabulary
    :type vocab_size: int
    :param mixer_types: List of mixer type names for each layer (e.g., ["S5", "S7", "attn", ...])
    :type mixer_types: list
    :param d_intermediate: Intermediate dimension for MLP layers (0 to disable MLP)
    :type d_intermediate: int
    :param mixer_kwargs: Additional arguments for mixer
    :type mixer_kwargs: dict, optional
    :param mlp_cls: MLP class to use
    :type mlp_cls: class, optional
    :param norm_epsilon: Epsilon value for layer normalization
    :type norm_epsilon: float
    :param rms_norm: Whether to use RMSNorm instead of LayerNorm
    :type rms_norm: bool
    :param initializer_cfg: Configuration for weight initialization
    :type initializer_cfg: dict, optional
    :param fused_add_norm: Whether to use fused add+norm operations
    :type fused_add_norm: bool
    :param residual_in_fp32: Whether to compute residuals in float32
    :type residual_in_fp32: bool
    :param tie_embeddings: Whether to tie input and output embeddings
    :type tie_embeddings: bool
    :param pad_vocab_size_multiple: Pad vocabulary size to multiple of this value
    :type pad_vocab_size_multiple: int
    :param device: Device to place tensors on
    :type device: torch.device, optional
    :param dtype: Data type for tensors
    :type dtype: torch.dtype, optional
    """
    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_layer: int,
        vocab_size: int,
        mixer_types: list,
        d_intermediate: int = 0,
        mixer_kwargs: Optional[Dict] = None,
        mlp_cls=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = False,
        tie_embeddings: bool = True,
        pad_vocab_size_multiple: int = 8,
        initializer_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.mixer_types = mixer_types
        self.d_intermediate = d_intermediate
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.tie_embeddings = tie_embeddings
        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        factory_kwargs = {"device": device, "dtype": dtype}

        # pad vocabulary size
        padded_vocab_size = vocab_size
        if vocab_size % pad_vocab_size_multiple != 0:
            padded_vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        # core LRNN model
        self.backbone = LRNNModel(
            d_model=d_model,
            d_state=d_state,
            n_layer=n_layer,
            vocab_size=padded_vocab_size,
            mixer_types=mixer_types,
            mixer_kwargs=mixer_kwargs,
            d_intermediate=d_intermediate,
            mlp_cls=mlp_cls,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )

        # language modeling head
        self.lm_head = nn.Linear(
            d_model, padded_vocab_size, bias=False, **factory_kwargs
        )

        # initialize weights
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # tie embeddings if specified
        if tie_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        """
        Tie input and output embeddings.

        This makes the embedding layer and language modeling head share the same weights,
        which is a common practice to reduce parameters and improve performance.
        """
        self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict:
        """
        Allocate inference cache.

        :param batch_size: Batch size for inference
        :type batch_size: int
        :param max_seqlen: Maximum sequence length for inference
        :type max_seqlen: int
        :param dtype: Data type for cache tensors
        :type dtype: torch.dtype, optional
        :param kwargs: Additional keyword arguments passed to backbone cache allocation
        :return: Dictionary mapping layer indices to their allocated caches
        :rtype: dict
        """
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def step(
        self,
        input_ids: Tensor,
        caches: Dict,
        integration_timesteps: Optional[Tensor] = None,
    ) -> namedtuple:
        """
        Single-step inference for autoregressive generation.

        :param input_ids: Input token IDs of shape (B, 1) - single token
        :type input_ids: Tensor
        :param caches: Dictionary mapping layer indices to their cached states
        :type caches: Dict
        :param integration_timesteps: Integration timesteps for LTV models (shape: B, 1 or B)
        :type integration_timesteps: Tensor, optional
        :return: Contains logits tensor of shape (B, 1, vocab_size)
        :rtype: namedtuple
        """
        # get hidden states
        hidden_states = self.backbone.step(
            input_ids, caches, integration_timesteps
        )

        # compute logits
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        inference_params: Optional[Dict] = None,
        num_last_tokens: int = 0,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        **mixer_kwargs,
    ) -> namedtuple:
        """
        Forward pass of the language model.

        :param input_ids: Input token IDs of shape (B, L)
        :type input_ids: Tensor
        :param position_ids: Position IDs (unused, for compatibility)
        :type position_ids: Tensor, optional
        :param inference_params: Parameters for inference mode
        :type inference_params: Dict, optional
        :param num_last_tokens: If > 0, only return logits for last n tokens
        :type num_last_tokens: int
        :param integration_timesteps: Timesteps for LTV models (shape: B, L)
        :type integration_timesteps: Tensor, optional
        :param lengths: Sequence lengths for variable-length sequences (shape: B)
        :type lengths: Tensor, optional
        :param mixer_kwargs: Additional keyword arguments passed to mixer layers
        :return: Contains logits tensor of shape (B, L, vocab_size)
        :rtype: namedtuple
        """
        # get hidden states from backbone
        hidden_states = self.backbone(
            input_ids,
            inference_params=inference_params,
            integration_timesteps=integration_timesteps,
            lengths=lengths,
            **mixer_kwargs,
        )

        # only keep last n tokens if specified
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]

        # compute logits
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        :param save_directory: Directory path where model and config will be saved
        :type save_directory: str
        """
        # create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # save model state dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # save configuration
        config_dict = {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "n_layer": self.n_layer,
            "vocab_size": self.vocab_size,
            "mixer_types": self.mixer_types,
            "d_intermediate": self.d_intermediate,
            "norm_epsilon": self.norm_epsilon,
            "rms_norm": self.rms_norm,
            "fused_add_norm": self.fused_add_norm,
            "residual_in_fp32": self.residual_in_fp32,
            "tie_embeddings": self.tie_embeddings,
            "pad_vocab_size_multiple": self.pad_vocab_size_multiple,
        }

        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        mixer_kwargs: Optional[Dict] = None,
        mlp_cls=None,
        initializer_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "LRNNLMHeadModel":
        """
        Load a pretrained model from a directory.

        :param pretrained_model_path: Path to directory containing saved model and config
        :type pretrained_model_path: str
        :param mixer_kwargs: Additional keyword arguments for mixer
        :type mixer_kwargs: dict, optional
        :param mlp_cls: MLP class to use
        :type mlp_cls: class, optional
        :param initializer_cfg: Configuration for weight initialization
        :type initializer_cfg: dict, optional
        :param device: Device to place tensors on
        :type device: torch.device, optional
        :param dtype: Data type for tensors
        :type dtype: torch.dtype, optional
        :param kwargs: Additional keyword arguments passed to model constructor
        :return: Loaded model instance
        :rtype: LRNNLMHeadModel
        """
        # load configuration
        config_path = os.path.join(pretrained_model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # create model
        model = cls(
            d_model=config_dict["d_model"],
            d_state=config_dict["d_state"],
            n_layer=config_dict["n_layer"],
            vocab_size=config_dict["vocab_size"],
            mixer_types=config_dict["mixer_types"],
            d_intermediate=config_dict.get("d_intermediate", 0),
            mixer_kwargs=mixer_kwargs,
            mlp_cls=mlp_cls,
            norm_epsilon=config_dict.get("norm_epsilon", 1e-5),
            rms_norm=config_dict.get("rms_norm", True),
            fused_add_norm=config_dict.get("fused_add_norm", True),
            residual_in_fp32=config_dict.get("residual_in_fp32", False),
            tie_embeddings=config_dict.get("tie_embeddings", True),
            pad_vocab_size_multiple=config_dict.get(
                "pad_vocab_size_multiple", 8
            ),
            initializer_cfg=initializer_cfg,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        # load state dict
        model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        return model
