"""
Language Expert
Specialized for language understanding and generation
Now powered by AgglutinativeAttention for morphological awareness
"""

import torch
import torch.nn as nn
from typing import Optional

from ..language.attention import AgglutinativeAttention
from ..core.base_components import FeedForward, LayerNorm

class LanguageExpert(nn.Module):
    """
    Language Expert that uses AgglutinativeAttention for morphological awareness.
    This provides Turkish-specific language understanding without external LLM dependencies.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        n_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or d_model * 4

        # Morfolojik farkındalıklı attention
        self.attention = AgglutinativeAttention(
            hidden_size=d_model,
            num_heads=n_heads,
            verb_bias=2.0,
            root_bias=1.5,
            suffix_bias=1.2
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

        # Morfolojik farkındalıklı attention
        attn_output, _ = self.attention(
            hidden_states=self.norm1(x),
            attention_mask=attention_mask,
            morpho_types=morpho_types
        )

        x = residual + self.dropout(attn_output)

        # Feed-forward
        residual = x
        ffn_output = self.ffn(self.norm2(x))
        x = residual + self.dropout(ffn_output)

        return x
