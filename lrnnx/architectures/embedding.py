"""Embedding modules for sequence models."""

from typing import Optional

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """
    Learned positional embeddings (position indices -> vectors).

    :param max_position_embeddings: Maximum sequence length supported.
    :type max_position_embeddings: int
    :param embedding_dim: Dimension of the embedding vectors.
    :type embedding_dim: int
    """

    def __init__(self, max_position_embeddings: int, embedding_dim: int):
        super().__init__()
        self.position_embedding = nn.Embedding(
            max_position_embeddings, embedding_dim
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional embeddings.

        :param positions: Tensor of position indices.
        :type positions: torch.Tensor
        :return: Positional embeddings.
        :rtype: torch.Tensor
        """
        return self.position_embedding(positions)


class TokenEmbedding(nn.Module):
    """
    Token embedding module. Positional embeddings are optional and explicit.

    By default this returns token lookups only. Enable learned positional
    embeddings with `use_position=True` and providing `max_position_embeddings`.

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param embedding_dim: Dimension of the embedding vectors.
    :type embedding_dim: int
    :param padding_idx: Index for padding tokens. Defaults to None.
    :type padding_idx: int, optional
    :param max_position_embeddings: Max sequence length for positional 
        embeddings. Required if `use_position=True`. Defaults to None.
    :type max_position_embeddings: int, optional
    :param use_position: Whether to include learned positional 
        embeddings. Defaults to False.
    :type use_position: bool, optional
    :param dropout: Dropout probability. Defaults to 0.1.
    :type dropout: float, optional
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        use_position: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )

        # Optional positional embeddings
        self.position_embedding: Optional[PositionEmbedding] = None
        if use_position:
            if max_position_embeddings is None:
                raise ValueError(
                    "max_position_embeddings must be set when use_position=True"
                )
            self.position_embedding = PositionEmbedding(
                max_position_embeddings, embedding_dim
            )

        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        :param token_ids: Tensor of token IDs of shape `(batch_size, seq_len)`.
        :type token_ids: torch.Tensor
        :return: Embedded tokens of shape `(batch_size, seq_len, embedding_dim)`.
        :rtype: torch.Tensor
        """
        embeddings = self.token_embedding(token_ids)

        if self.position_embedding is not None:
            seq_len = token_ids.size(1)
            positions = (
                torch.arange(seq_len, device=token_ids.device)
                .unsqueeze(0)
                .expand_as(token_ids)
            )
            pos_emb = self.position_embedding(positions)
            embeddings = embeddings + pos_emb

        return self.dropout(embeddings)