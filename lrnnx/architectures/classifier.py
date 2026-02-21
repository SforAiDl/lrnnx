"""Classifier using Linear RNN models with support for token embeddings.

Reference: https://github.com/Efficient-Scalable-Machine-Learning/event-ssm
"""

from typing import List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from lrnnx.architectures.embedding import TokenEmbedding
from lrnnx.models.lti.centaurus import Centaurus
from lrnnx.models.lti.lru import LRU
from lrnnx.models.lti.s5 import S5


def _get_mixer_class_from_string(mixer_type: str):
    """Helper to convert string names to model classes."""
    mixer_registry = {
        "LRU": LRU,
        "S5": S5,
        "Centaurus": Centaurus,
        # "Mamba": Mamba,
    }
    if mixer_type not in mixer_registry:
        raise ValueError(
            f"Unknown mixer type: {mixer_type}. Available: {list(mixer_registry.keys())}"
        )
    return mixer_registry[mixer_type]


class SequencePooling(nn.Module):
    """
    Pooling layer for sequence data with support for variable lengths.

    Handles both intermediate pooling (reducing sequence length) and
    final pooling (creating a single vector representation).
    """

    def __init__(self, pooling_type="last", stride=1):
        """
        Initialize the pooling layer.

        Args:
            pooling_type (str): Pooling mode ("last", "mean", "max", "stride")
            stride (int): Stride for pooling (only used for intermediate pooling)
        """
        super().__init__()
        self.pooling_type = pooling_type
        self.stride = stride

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        integration_timesteps: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Pool sequences, either reducing length (intermediate) or to a single vector (final).

        Args:
            x (torch.Tensor): Input of shape ``(B, L, D)``.
            lengths (torch.Tensor, optional): Actual sequence lengths of shape ``(B,)``.
                Defaults to None.
            integration_timesteps (torch.Tensor, optional): Timesteps of shape ``(B, L)``.
                Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: Pooled tensor (and updated
                timesteps / lengths for intermediate pooling).
        """

        B, L, D = x.shape

        # Intermediate pooling (reducing sequence length)
        if self.stride > 1:
            if self.pooling_type == "stride":
                x = x[:, :: self.stride, :]
                if integration_timesteps is not None:
                    integration_timesteps = integration_timesteps[
                        :, :: self.stride
                    ]
                if lengths is not None:
                    lengths = torch.ceil(lengths.float() / self.stride).long()
                    max_len = x.shape[1]
                    lengths = torch.clamp(lengths, max=max_len)

            elif self.pooling_type in ["mean", "max"]:
                # Mask for valid tokens
                if lengths is not None:
                    mask = torch.arange(L, device=x.device).unsqueeze(
                        0
                    ) < lengths.unsqueeze(1)
                    mask = mask.float().unsqueeze(2)  # (B, L, 1)
                    x = x * mask

                x_pooled = x.transpose(1, 2)  # (B, L, D) -> (B, D, L)
                if self.pooling_type == "mean":
                    x_pooled = nn.functional.avg_pool1d(
                        x_pooled,
                        kernel_size=self.stride,
                        stride=self.stride,
                    )
                else:  # "max"
                    # For max, set padded values to a large negative number so they don't affect max
                    if lengths is not None:
                        x_masked = x + (mask - 1) * 1e9  # (B, L, D)
                        x_pooled = x_masked.transpose(1, 2)
                    x_pooled = nn.functional.max_pool1d(
                        x_pooled,
                        kernel_size=self.stride,
                        stride=self.stride,
                    )
                x = x_pooled.transpose(1, 2)  # (B, D, L') -> (B, L', D)

                # Update timesteps by summing over pooling windows
                if integration_timesteps is not None:
                    ts_unfolded = integration_timesteps.unfold(
                        1, self.stride, self.stride
                    )
                    integration_timesteps = ts_unfolded.sum(dim=2)

                if lengths is not None:
                    # Pool the mask to get new valid lengths
                    mask = mask.transpose(1, 2)  # (B, 1, L)
                    pooled_mask = nn.functional.avg_pool1d(
                        mask,
                        kernel_size=self.stride,
                        stride=self.stride,
                    )
                    pooled_mask = pooled_mask.transpose(1, 2)  # (B, L', 1)
                    # New lengths: count windows with any valid token
                    new_lengths = (pooled_mask.squeeze(2) > 0).sum(dim=1)
                    lengths = new_lengths

            else:
                raise ValueError(
                    f"Unknown intermediate pooling strategy: {self.pooling_type}"
                )

            return x, integration_timesteps, lengths

        # Final pooling (sequence -> single vector)
        else:
            if self.pooling_type == "last":
                if lengths is not None:
                    # Use actual last timestep for variable-length sequences
                    batch_indices = torch.arange(B, device=x.device)
                    last_indices = torch.clamp(lengths - 1, 0, L - 1)
                    pooled = x[batch_indices, last_indices, :]
                else:
                    pooled = x[:, -1, :]

            elif self.pooling_type == "mean":
                if lengths is not None:
                    # Masked mean for variable-length sequences
                    mask = torch.arange(L, device=x.device).unsqueeze(
                        0
                    ) < lengths.unsqueeze(1)
                    mask = mask.unsqueeze(2).float()
                    pooled = (x * mask).sum(dim=1) / torch.clamp(
                        lengths.unsqueeze(1).float(), min=1
                    )
                else:
                    pooled = x.mean(dim=1)

            elif self.pooling_type == "max":
                if lengths is not None:
                    # Masked max for variable-length sequences
                    mask = torch.arange(L, device=x.device).unsqueeze(
                        0
                    ) < lengths.unsqueeze(1)
                    mask = mask.unsqueeze(2).float()
                    masked_x = x * mask + (1 - mask) * (-1e9)
                    pooled = masked_x.max(dim=1)[0]
                else:
                    pooled = x.max(dim=1)[0]

            else:
                raise ValueError(
                    f"Unknown pooling strategy: {self.pooling_type}"
                )

            return pooled, integration_timesteps, lengths


class ClassifierBlock(nn.Module):
    """
    A single processing block in the Classifier.

    Each block contains:
    - LRNN layer for temporal processing (instantiated from lrnn_cls)
    - Optional intermediate pooling for sequence length reduction
    - Dropout for regularization
    - Residual connection
    - Layer normalization
    """

    def __init__(
        self,
        d_model,
        d_state,
        lrnn_cls: Type[nn.Module],
        num_classes: int = 0,
        output_dim: int = 1,
        pooling: Literal["mean", "last", "max"] = "last",
        dropout: float = 0.1,
        intermediate_pooling: Literal[
            "none", "stride", "mean", "max"
        ] = "none",
        pooling_factor: int = 2,
        is_final: bool = False,
        **lrnn_params,
    ):
        """
        Initialize a processing block used inside the classifier.

        The block performs sequence processing and when is_final=True
        produces a single output vector. Set num_classes > 0 to enable
        classification (the block returns logits over num_classes); otherwise
        the block produces regression outputs of shape output_dim.
        """
        super().__init__()

        # Instantiate LRNN layer directly from lrnn_params.
        # The user must provide all required constructor arguments in lrnn_params.
        # Examples:
        #   - LRU: lrnn_params={"d_model": d_model, "d_state": d_state}
        #   - S5: lrnn_params={"d_model": d_model, "d_state": d_state, "discretization": "zoh"}
        #   - Centaurus: lrnn_params={"d_model": d_model, "d_state": d_state, "sub_state_dim": d_state}
        #   - Mamba: lrnn_params={"d_model": d_model, "d_state": d_state}
        try:
            self.lrnn = lrnn_cls(**lrnn_params)
        except TypeError as e:
            raise TypeError(
                f"Could not instantiate {getattr(lrnn_cls, '__name__', str(lrnn_cls))} "
                f"with provided lrnn_params: {lrnn_params}. "
                f"Ensure you pass all required constructor arguments for the LRNN class. "
                f"Error: {e}"
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.intermediate_pooling = intermediate_pooling
        self.pooling_factor = pooling_factor
        self.is_final = is_final

        # Create pooling layer if needed. Annotate as Optional to allow None assignment.
        self.pooler: Optional[SequencePooling] = None
        if intermediate_pooling != "none":
            self.pooler = SequencePooling(
                pooling_type=intermediate_pooling, stride=pooling_factor
            )

        # Final pooling and output head for the last block
        self.final_pooler: Optional[SequencePooling] = None
        if is_final:
            self.final_pooler = SequencePooling(pooling_type=pooling)
            if num_classes > 0:
                self.output_proj = nn.Linear(d_model, num_classes)
            else:
                self.output_proj = nn.Linear(d_model, output_dim)

    def forward(
        self,
        x: Tensor,
        integration_timesteps: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
        """
        Forward pass through a single classifier block.

        Args:
            x (torch.Tensor): Input of shape ``(B, L, D)``.
            integration_timesteps (torch.Tensor, optional): Timesteps for LTV models.
                Defaults to None.
            lengths (torch.Tensor, optional): Actual sequence lengths.
                Defaults to None.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
                Final block returns logits ``(B, num_classes)``;
                non-final blocks return ``(x, integration_timesteps, lengths)``.
        """
        # Standard block processing
        x_res = x
        x = self.lrnn(x, integration_timesteps, lengths)
        x = self.dropout(x)
        x = self.norm(x + x_res)
        # Apply intermediate pooling if specified
        if self.intermediate_pooling != "none" and self.pooler is not None:
            x, integration_timesteps, lengths = self.pooler(
                x, lengths, integration_timesteps
            )

        if self.is_final:
            # Final pooling and output head
            # final_pooler is Optional but only set when is_final is True
            assert self.final_pooler is not None
            pooled, _, _ = self.final_pooler(x, lengths, integration_timesteps)
            return self.output_proj(pooled)
        else:
            return x, integration_timesteps, lengths


class Classifier(nn.Module):
    """
    Classifier: Sequence classifier or regressor...

    Args:
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.
        d_model (int): Hidden dimension of the model.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 0,
        output_dim: int = 1,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        lrnn_cls: Union[Type[nn.Module], List[Type[nn.Module]]] = LRU,
        pooling: Literal["mean", "last", "max"] = "last",
        dropout: float = 0.1,
        intermediate_pooling: Union[
            Literal["none", "stride", "mean", "max"],
            List[Literal["none", "stride", "mean", "max"]],
        ] = "none",
        pooling_factor: Union[int, List[int]] = 2,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        padding_idx: Optional[int] = 0,
        lrnn_params: Optional[dict] = None,
    ):
        """
        Initializes the Classifier.

        Args:
            input_dim (int): Number of input features (ignored when vocab_size is provided).
            num_classes (int, optional): Number of output classes. Defaults to 0.
            output_dim (int, optional): Number of regression outputs. Defaults to 1.
            d_model (int, optional): Hidden dimension of the model. Defaults to 128.
            d_state (int, optional): State dimension for the LRNN layers. Defaults to 64.
            n_layers (int, optional): Number of LRNN layers. Defaults to 4.
            lrnn_cls (type | list[type], optional): Custom LRNN class or list of classes
                (one per layer) to use. Defaults to LRU.
            pooling (str, optional): Pooling strategy for sequence outputs. Defaults to ``"last"``.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            intermediate_pooling (str | list[str], optional): Pooling strategy for
                each layer. Defaults to ``"none"``.
            pooling_factor (int | list[int], optional): Factor by which to reduce
                sequence length. Defaults to 2.
            vocab_size (int, optional): Size of vocabulary for token embeddings. Defaults to None.
            embedding_dim (int, optional): Dimension of embeddings (defaults to d_model). Defaults to None.
            max_position_embeddings (int, optional): Max sequence length for positional
                embeddings. Defaults to None.
            padding_idx (int, optional): Index of padding token for embedding layer. Defaults to 0.
            lrnn_params (dict, optional): Additional parameters for LRNN modules. Defaults to None.
        """
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling

        # Determine if using token embeddings
        if vocab_size is not None:
            # Create token embedding layer
            emb_dim = embedding_dim if embedding_dim is not None else d_model
            self.embedding = TokenEmbedding(
                vocab_size=vocab_size,
                embedding_dim=emb_dim,
                padding_idx=padding_idx,
                max_position_embeddings=max_position_embeddings,
                use_position=False,  # set True if you want learned positional embeddings
                dropout=dropout,
            )
            self.has_embedding = True

            # Project embeddings to model dimension if needed
            self.embed_proj = (
                nn.Linear(emb_dim, d_model)
                if emb_dim != d_model
                else nn.Identity()
            )  # type: nn.Module
        else:
            # For raw features, use standard projection
            self.has_embedding = False
            self.input_proj = nn.Linear(input_dim, d_model)

        # Handle pooling configuration - cast to proper types
        if isinstance(intermediate_pooling, str):
            intermediate_pooling_list: List[
                Literal["none", "stride", "mean", "max"]
            ] = [intermediate_pooling] * n_layers
        else:
            intermediate_pooling_list = intermediate_pooling

        if isinstance(pooling_factor, int):
            pooling_factor_list = [pooling_factor] * n_layers
        else:
            pooling_factor_list = pooling_factor

        lrnn_params = lrnn_params or {}

        # Normalize lrnn_cls to a list
        if not isinstance(lrnn_cls, list):
            lrnn_cls = [lrnn_cls] * n_layers

        if len(lrnn_cls) != n_layers:
            raise ValueError(
                f"lrnn_cls list length ({len(lrnn_cls)}) must match n_layers ({n_layers})"
            )

        # Convert any strings to classes using the helper
        lrnn_cls_list = []
        for item in lrnn_cls:
            if isinstance(item, str):
                lrnn_cls_list.append(_get_mixer_class_from_string(item))
            else:
                lrnn_cls_list.append(item)

        # Stack of blocks (all but last have no output head)
        self.blocks = nn.ModuleList()
        for i in range(n_layers - 1):
            self.blocks.append(
                ClassifierBlock(
                    d_model=d_model,
                    d_state=d_state,
                    num_classes=0,
                    output_dim=0,
                    pooling=pooling,
                    lrnn_cls=lrnn_cls_list[i],
                    dropout=dropout,
                    intermediate_pooling=intermediate_pooling_list[i],
                    pooling_factor=pooling_factor_list[i],
                    is_final=False,
                    **lrnn_params,
                )
            )
        # Last block has output head
        self.final_block = ClassifierBlock(
            d_model=d_model,
            d_state=d_state,
            num_classes=num_classes,
            output_dim=output_dim,
            pooling=pooling,
            lrnn_cls=lrnn_cls_list[-1],
            dropout=dropout,
            intermediate_pooling=intermediate_pooling_list[-1],
            pooling_factor=pooling_factor_list[-1],
            is_final=True,
            **lrnn_params,
        )

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        integration_timesteps: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
        """
        Forward pass of the classifier/regressor.

        Args:
            x (torch.Tensor): Input tensor. Token IDs of shape ``(B, L)`` when using embeddings,
                or continuous features of shape ``(B, L, input_dim)`` otherwise.
            lengths (torch.Tensor, optional): Actual sequence lengths of shape ``(B,)``.
                Defaults to None.
            integration_timesteps (torch.Tensor, optional): Timesteps of shape ``(B, L)``
                for LTV models. Defaults to None.

        Returns:
            torch.Tensor: Logits of shape ``(B, num_classes)`` or regression values
                of shape ``(B, output_dim)``.
        """
        # Process input based on type and model configuration
        if self.has_embedding:
            # If input is token ids (B, L) -> embed then project
            if x.dim() == 2:
                x = self.embedding(x)
                x = self.embed_proj(x)
            else:
                # Already-embedded inputs (B, L, D) -> project with embed_proj
                x = self.embed_proj(x)
        else:
            # Raw continuous inputs -> project from input_dim -> d_model
            x = self.input_proj(x)

        # Pass through all but last block (non-final blocks must return '(x, integration_timesteps, lengths)').
        for block in self.blocks:
            x, integration_timesteps, lengths = block(
                x, integration_timesteps, lengths
            )
        # Final block returns output
        return self.final_block(x, integration_timesteps, lengths)
