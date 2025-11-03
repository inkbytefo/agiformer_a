"""
Spatial Expert
Specialized for spatial relationships and geometric reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..core.attention import MultiHeadAttention


class SpatialExpert(nn.Module):
    """
    Spatial Expert for spatial and geometric reasoning
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
        
        # Spatial attention
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Position encoding for spatial relationships
        self.spatial_encoding = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Geometric reasoning module
        self.geometric_net = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Distance/angle computation (for geometric features)
        self.distance_proj = nn.Linear(1, d_model // 4)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def compute_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial relationship features
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute pairwise distances (using L2 norm)
        x_expanded1 = x.unsqueeze(2)  # [batch, seq, 1, d_model]
        x_expanded2 = x.unsqueeze(1)  # [batch, 1, seq, d_model]
        
        distances = torch.norm(x_expanded1 - x_expanded2, dim=-1, keepdim=True)  # [batch, seq, seq, 1]
        distances = self.distance_proj(distances)  # [batch, seq, seq, d_model//4]
        
        # Aggregate distances
        spatial_features = distances.mean(dim=2)  # [batch, seq, d_model//4]
        
        # Expand to full dimension
        spatial_features = F.pad(spatial_features, (0, self.d_model - self.d_model // 4))
        
        return spatial_features
    
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
        # Compute spatial features
        spatial_features = self.compute_spatial_features(x)
        
        # Add spatial encoding
        x = x + self.spatial_encoding(spatial_features)
        
        # Spatial attention
        residual = x
        x = self.norm1(x)
        attn_out = self.attention(x, x, x, mask)
        x = residual + self.dropout(attn_out)
        
        # Geometric reasoning
        residual = x
        x = self.norm2(x)
        geometric_out = self.geometric_net(x)
        x = residual + geometric_out
        
        return x

