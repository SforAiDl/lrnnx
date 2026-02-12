"""Embedding modules for sequence models."""

from typing import Optional

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """Learned positional embeddings (position indices -> vectors)."""

    def __init__(self, max_position_embeddings: int, embedding_dim: int):
        super().__init__()
        self.position_embedding = nn.Embedding(
            max_position_embeddings, embedding_dim
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.position_embedding(positions)


class TokenEmbedding(nn.Module):
    """
    Token embedding module. Positional embeddings are optional and explicit.

    By default this returns token lookups only. Enable learned positional
    embeddings with use_position=True and providing max_position_embeddings.
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

        Args
        ----
            token_ids: Tensor of token IDs [batch_size, seq_len]

        Returns
        -------
            Tensor: [batch_size, seq_len, embedding_dim]
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
