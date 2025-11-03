"""
Logic Expert
Specialized for logical and mathematical reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..core.attention import MultiHeadAttention


class LogicExpert(nn.Module):
    """
    Logic Expert for logical and mathematical reasoning
    Uses structured attention patterns for logical relationships
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
        self.d_ff = d_ff or (d_model * 4)
        
        # Multi-head attention for logical relationships
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Logical structure encoder
        self.logic_encoder = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Relational reasoning module
        self.relation_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
        """
        # Self-attention for logical patterns
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, x, x, mask)
        x = residual + self.dropout(attn_out)

        # Memory-efficient relational reasoning
        batch_size, seq_len, d_model = x.size()

        # Use chunked processing to avoid O(seq_len²) memory
        chunk_size = min(64, seq_len)  # Process in chunks of 64 tokens max
        relations_list = []

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            x_chunk = x[:, i:end_i]  # [batch, chunk_size, d_model]

            # Create relations only within this chunk to reduce memory
            chunk_pairs_1 = x_chunk.unsqueeze(2)  # [batch, chunk_size, 1, d_model]
            chunk_pairs_2 = x_chunk.unsqueeze(1)  # [batch, 1, chunk_size, d_model]

            # Concatenate pairs within chunk
            pairs = torch.cat([chunk_pairs_1.expand(-1, -1, end_i-i, -1),
                              chunk_pairs_2.expand(-1, end_i-i, -1, -1)], dim=-1)
            # [batch, chunk_size, chunk_size, 2*d_model]

            # Apply relation network
            chunk_relations = self.relation_net(pairs)  # [batch, chunk_size, chunk_size, d_model]
            chunk_relations = chunk_relations.mean(dim=2)  # [batch, chunk_size, d_model]

            relations_list.append(chunk_relations)

        # Concatenate all chunk relations
        relations = torch.cat(relations_list, dim=1)  # [batch, seq_len, d_model]

        x = x + relations

        # Logic encoder
        residual = x
        x = self.norm2(x)
        logic_out = self.logic_encoder(x)
        x = residual + logic_out

        return x
