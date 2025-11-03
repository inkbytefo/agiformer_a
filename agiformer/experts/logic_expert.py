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
        
        # Relational reasoning
        batch_size, seq_len = x.size(0), x.size(1)
        # Create pair-wise combinations
        x_expanded1 = x.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq, seq, d_model]
        x_expanded2 = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq, seq, d_model]
        pairs = torch.cat([x_expanded1, x_expanded2], dim=-1)  # [batch, seq, seq, 2*d_model]
        
        # Apply relation net and aggregate
        relations = self.relation_net(pairs)  # [batch, seq, seq, d_model]
        relations = relations.mean(dim=2)  # [batch, seq, d_model] - aggregate over pairs
        
        x = x + relations
        
        # Logic encoder
        residual = x
        x = self.norm2(x)
        logic_out = self.logic_encoder(x)
        x = residual + logic_out
        
        return x

