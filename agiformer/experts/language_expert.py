# Developer: inkbytefo
# Modified: 2025-11-06

"""
Language Expert
Specialized for language understanding and generation
Supports both AgglutinativeAttention and standard MultiHeadAttention for comparison
"""

import torch
import torch.nn as nn
from typing import Optional

from ..language.attention import AgglutinativeAttention
from ..core.attention import MultiHeadAttention
from ..core.base_components import FeedForward, LayerNorm

class LanguageExpert(nn.Module):
    """
    Language Expert that can use either AgglutinativeAttention or standard MultiHeadAttention
    based on configuration. This enables fair comparison between the two approaches.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        n_heads: int = 12,
        dropout: float = 0.1,
        use_agglutinative_attention: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or d_model * 4
        self.use_agglutinative_attention = use_agglutinative_attention

        # Initialize attention mechanism based on configuration
        if use_agglutinative_attention:
            # Morphologically-aware attention for Turkish
            self.attention = AgglutinativeAttention(
                hidden_size=d_model,
                num_heads=n_heads,
                verb_bias=2.0,
                root_bias=1.5,
                suffix_bias=1.2
            )
        else:
            # Standard multi-head attention for baseline comparison
            self.attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )

        # Feed-forward network
        self.ffn = FeedForward(d_model, self.d_ff, dropout)

        # Layer norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        morpho_types: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        residual = x

        if self.use_agglutinative_attention:
            # Morphologically-aware attention
            attn_output, _ = self.attention(
                hidden_states=self.norm1(x),
                attention_mask=attention_mask,
                morpho_types=morpho_types
            )
        else:
            # Standard self-attention (query=key=value=x)
            attn_output = self.attention(
                query=self.norm1(x),
                key=self.norm1(x),
                value=self.norm1(x),
                mask=attention_mask
            )

        x = residual + self.dropout(attn_output)

        # Feed-forward
        residual = x
        ffn_output = self.ffn(self.norm2(x))
        x = residual + self.dropout(ffn_output)

        return x
