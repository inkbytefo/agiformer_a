"""
Causal Expert
Specialized for causal reasoning and cause-effect relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..core.attention import MultiHeadAttention


class CausalExpert(nn.Module):
    """
    Causal Expert for understanding cause-effect relationships
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
        
        # Causal attention (asymmetric, directed)
        self.causal_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Temporal/directional encoding
        self.temporal_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Causal structure encoder
        self.causal_encoder = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Cause-effect relationship network
        self.causal_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def compute_causal_mask(self, seq_len: int, causal: bool = True) -> Optional[torch.Tensor]:
        """
        Create causal mask (lower triangular) for directed causal attention
        """
        if not causal:
            return None
        
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            causal: Whether to use causal masking
            mask: Optional additional mask
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Add temporal encoding
        if seq_len <= self.temporal_encoding.size(1):
            temporal_enc = self.temporal_encoding[:, :seq_len, :]
            x = x + temporal_enc
        
        # Causal attention
        causal_mask = self.compute_causal_mask(seq_len, causal)
        if mask is not None and causal_mask is not None:
            # Combine masks
            mask = mask & causal_mask.squeeze(0).squeeze(0)
        elif causal_mask is not None:
            mask = causal_mask.squeeze(0).squeeze(0)
        
        residual = x
        x = self.norm1(x)
        attn_out = self.causal_attention(x, x, x, mask)
        x = residual + self.dropout(attn_out)
        
        # Causal relationship computation
        # Create cause-effect pairs
        x_cause = x.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq, seq, d_model]
        x_effect = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq, seq, d_model]
        causal_pairs = torch.cat([x_cause, x_effect], dim=-1)  # [batch, seq, seq, 2*d_model]
        
        # Apply causal net
        causal_relations = self.causal_net(causal_pairs)  # [batch, seq, seq, d_model]
        
        # Only consider forward direction (cause -> effect)
        if causal:
            causal_mask_2d = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            causal_mask_2d = causal_mask_2d.unsqueeze(0).unsqueeze(-1)
            causal_relations = causal_relations * causal_mask_2d
        
        # Aggregate
        causal_features = causal_relations.mean(dim=2)  # [batch, seq, d_model]
        x = x + causal_features
        
        # Causal encoder
        residual = x
        x = self.norm2(x)
        causal_out = self.causal_encoder(x)
        x = residual + causal_out
        
        return x

