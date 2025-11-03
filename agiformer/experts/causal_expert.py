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
        
        self.causal_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.temporal_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        self.causal_encoder = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.causal_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    # --- DEĞİŞİKLİK: `device` parametresi eklendi ---
    def compute_causal_mask(self, seq_len: int, causal: bool = True, device: torch.device = None) -> Optional[torch.Tensor]:
        if not causal:
            return None
        
        # Maskeyi doğru cihazda oluştur
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    # --- BİTTİ ---

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        if seq_len <= self.temporal_encoding.size(1):
            temporal_enc = self.temporal_encoding[:, :seq_len, :]
            x = x + temporal_enc
        
        # --- DEĞİŞİKLİK: `device=x.device` parametresi eklendi ---
        causal_mask = self.compute_causal_mask(seq_len, causal, device=x.device)
        # --- BİTTİ ---

        if mask is not None and causal_mask is not None:
            mask = mask & causal_mask.squeeze(0).squeeze(0)
        elif causal_mask is not None:
            mask = causal_mask.squeeze(0).squeeze(0)
        
        residual = x
        x = self.norm1(x)
        attn_out = self.causal_attention(x, x, x, mask)
        x = residual + self.dropout(attn_out)
        
        x_cause = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_effect = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        causal_pairs = torch.cat([x_cause, x_effect], dim=-1)
        
        causal_relations = self.causal_net(causal_pairs)
        
        if causal:
            # --- DEĞİŞİKLİK: `device=x.device` parametresi eklendi ---
            causal_mask_2d = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            causal_mask_2d = causal_mask_2d.unsqueeze(0).unsqueeze(-1)
            # --- BİTTİ ---
            causal_relations = causal_relations * causal_mask_2d
        
        causal_features = causal_relations.mean(dim=2)
        x = x + causal_features
        
        residual = x
        x = self.norm2(x)
        causal_out = self.causal_encoder(x)
        x = residual + causal_out
        
        return x
