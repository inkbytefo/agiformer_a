# Developer: inkbytefo
# Modified: 2025-11-07

"""
Language Expert

Morphology- and semantics-aware language expert, integrating the core ideas
from the former TMA1Model directly into the AGIFORMER MoE stack.

- Supports AgglutinativeAttention for Turkish-specific modeling
- Optionally enriches token embeddings with:
  - morpho_types: morphological categories
  - semantic_categories: semantic role / category hints
"""

import torch
import torch.nn as nn
from typing import Optional, Any

from ..language.attention import AgglutinativeAttention
from ..core.attention import MultiHeadAttention
from ..core.base_components import FeedForward, LayerNorm

# These constants mirror the design used in TMA-style morphology/semantics encoding.
NUM_MORPHEME_TYPES = 23
NUM_SEMANTIC_CATEGORIES = 12


class LanguageExpert(nn.Module):
    """
    Morphology- and semantics-aware Language Expert.

    Responsibilities:
    - Optionally inject morpho_types and semantic_categories embeddings into the hidden states
    - Run AgglutinativeAttention (or standard MHA) over enriched representations
    - Return transformed states preserving [batch, seq_len, d_model] shape
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        n_heads: int = 12,
        dropout: float = 0.1,
        use_agglutinative_attention: bool = True,
        use_morpho_semantic_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or d_model * 4
        self.use_agglutinative_attention = use_agglutinative_attention
        self.use_morpho_semantic_embeddings = use_morpho_semantic_embeddings

        # Core attention mechanism
        if use_agglutinative_attention:
            self.attention = AgglutinativeAttention(
                hidden_size=d_model,
                num_heads=n_heads,
                verb_bias=2.0,
                root_bias=1.5,
                suffix_bias=1.2,
            )
        else:
            self.attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

        # Optional morphology / semantics embeddings (TMA-style enrichment)
        if self.use_morpho_semantic_embeddings:
            self.morpho_embedding = nn.Embedding(NUM_MORPHEME_TYPES, d_model)
            self.semantic_embedding = nn.Embedding(NUM_SEMANTIC_CATEGORIES, d_model)

        self.ffn = FeedForward(d_model, self.d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        morpho_types: Optional[torch.Tensor] = None,
        semantic_categories: Optional[torch.Tensor] = None,
        **kwargs: Any,  # ignore unused MoE kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] base hidden states
            attention_mask: [batch, seq_len] or broadcastable mask
            morpho_types: [batch, seq_len] (optional)
            semantic_categories: [batch, seq_len] (optional)

        Returns:
            Updated hidden states [batch, seq_len, d_model]
        """
        hidden_states = x
        morpho_types_clamped = None

        # Inject morpho/semantic embeddings if available
        if (
            self.use_morpho_semantic_embeddings
            and morpho_types is not None
            and semantic_categories is not None
        ):
            # Clamp indices to the valid range to prevent out-of-bounds errors
            morpho_types_clamped = torch.clamp(
                morpho_types, 0, NUM_MORPHEME_TYPES - 1
            )
            semantic_categories_clamped = torch.clamp(
                semantic_categories, 0, NUM_SEMANTIC_CATEGORIES - 1
            )

            morpho_emb = self.morpho_embedding(morpho_types_clamped)
            semantic_emb = self.semantic_embedding(semantic_categories_clamped)

            hidden_states = hidden_states + morpho_emb + semantic_emb

        # Self-attention with potential morphological biasing
        residual = hidden_states

        if self.use_agglutinative_attention:
            # Use clamped morpho_types if they were computed, otherwise pass original
            morpho_types_for_attention = morpho_types_clamped if morpho_types_clamped is not None else morpho_types
            
            attn_output, _ = self.attention(
                hidden_states=self.norm1(hidden_states),
                attention_mask=attention_mask,
                morpho_types=morpho_types_for_attention,
            )
        else:
            attn_output = self.attention(
                query=self.norm1(hidden_states),
                key=self.norm1(hidden_states),
                value=self.norm1(hidden_states),
                mask=attention_mask,
            )

        x = residual + self.dropout(attn_output)

        # Feed-forward
        residual = x
        ffn_output = self.ffn(self.norm2(x))
        x = residual + self.dropout(ffn_output)

        return x
