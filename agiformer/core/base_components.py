"""
Base components for AGIFORMER architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    """Layer Normalization with optional learnable parameters"""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with learnable adjustments"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
        # Learnable adjustments
        self.learnable_adjust = nn.Parameter(torch.zeros(1, max_len, d_model))
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len, :] + self.learnable_adjust[:, :seq_len, :]
        return self.dropout(x + pos_encoding)


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and residual scaling"""
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Residual scaling (from GPT-style models)
        self.residual_scale = math.sqrt(0.5)
        
    def forward(self, x):
        residual = x
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        return (x + residual) * self.residual_scale


class ResidualConnection(nn.Module):
    """Residual connection with pre-normalization"""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """Apply sublayer with residual connection and pre-normalization"""
        return x + self.dropout(sublayer(self.norm(x)))

