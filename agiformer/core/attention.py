## Developer: inkbytefo
## Modified: 2025-11-07
"""
Advanced Attention Mechanisms for AGIFORMER
Includes standard, linear, syntax-aware, and cross-modal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Self-Attention"""
    
    def __init__(self, d_model, n_heads, dropout=0.1, bias=False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: Optional attention mask
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections and split into heads
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # CRITICAL FIX: Properly handle attention mask dimensions
        if mask is not None:
            # Ensure mask is properly sized for broadcasting
            if mask.dim() == 1:  # [seq_len] - broadcast to all heads
                # Expand to [batch, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 2:  # [batch, seq_len]
                # Expand to [batch, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1).expand(batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 3:  # [batch, seq_len, seq_len] or similar
                # Ensure the last two dimensions match seq_len_q and seq_len_k
                if mask.size(1) != seq_len_q or mask.size(2) != seq_len_k:
                    mask = mask[:, :seq_len_q, :seq_len_k]
                # Expand to [batch, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:  # [batch, heads, seq_len, seq_len]
                # Keep as is but ensure dimensions match
                mask = mask[:, :, :seq_len_q, :seq_len_k]
            
            # Apply mask - scores should be [batch, n_heads, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.w_o(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output


class LinearAttention(nn.Module):
    """
    Linear Attention: O(n) complexity instead of O(n^2)
    Based on Performer and Linformer principles
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1, feature_dim=None):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.feature_dim = feature_dim or max(self.d_k, 64)
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Random features for kernel approximation
        self.register_buffer('random_proj', torch.randn(self.d_k, self.feature_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-8
        
    def kernel_function(self, x):
        """Positive random feature map for linear attention"""
        # Using ReLU-based feature map
        return F.relu(x @ self.random_proj) + self.eps
    
    def forward(self, query, key, value, mask=None):
        """
        Linear attention with feature maps
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply kernel function
        Q_kernel = self.kernel_function(Q)  # [batch, n_heads, seq_q, feature_dim]
        K_kernel = self.kernel_function(K)  # [batch, n_heads, seq_k, feature_dim]
        
        # Linear attention: Q_kernel @ (K_kernel^T @ V)
        # This gives O(n) complexity instead of O(n^2)
        KV = torch.einsum('bhkd,bhvd->bhkv', K_kernel, V)  # [batch, n_heads, feature_dim, d_k]
        output = torch.einsum('bhqd,bhkd->bhqd', Q_kernel, KV)  # [batch, n_heads, seq_q, d_k]
        
        # Normalization
        Z = torch.einsum('bhqd->bhq', Q_kernel).unsqueeze(-1)  # [batch, n_heads, seq_q, 1]
        output = output / (Z + self.eps)
        
        # CRITICAL FIX: Properly handle attention mask for linear attention
        if mask is not None:
            if mask.dim() == 1:  # [seq_len]
                mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_heads, seq_len_q)
            elif mask.dim() == 2:  # [batch, seq_len]
                mask = mask.unsqueeze(1).expand(batch_size, self.n_heads, seq_len_q)
            
            # Apply mask to prevent attention to padded positions
            output = output * mask.unsqueeze(-1)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        output = self.w_o(output)
        
        return output


class SyntaxAwareAttention(nn.Module):
    """
    Syntax-aware attention that considers syntactic structure
    Inspired by TMA-1's morphological and semantic awareness
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.base_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Syntax-aware bias network
        self.syntax_bias_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_heads),
            nn.Softmax(dim=-1)
        )
        
        # Morphological embedding
        self.morpho_embed = nn.Embedding(100, d_model)  # Placeholder for morphological features
        
    def forward(self, query, key, value, syntax_features=None, mask=None):
        """
        Args:
            syntax_features: Optional syntactic structure features
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Standard attention
        base_output = self.base_attention(query, key, value, mask)
        
        if syntax_features is not None:
            # Compute syntax-aware bias
            syntax_bias = self.syntax_bias_net(syntax_features)  # [batch, seq_len, n_heads]
            
            # Apply syntax-aware modulation
            # Reshape for broadcasting
            syntax_bias = syntax_bias.unsqueeze(1)  # [batch, 1, seq_len, n_heads]
            # Modulate the attention output (simplified - could be more sophisticated)
            base_output = base_output * (1 + 0.1 * syntax_bias.mean(-1).unsqueeze(-1))
        
        return base_output


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for multimodal fusion
    Allows different modalities to attend to each other
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
    def forward(self, query_modal, key_modal, value_modal, mask=None):
        """
        Args:
            query_modal: Query from one modality [batch, seq_q, d_model]
            key_modal: Key from another modality [batch, seq_k, d_model]
            value_modal: Value from another modality [batch, seq_v, d_model]
        """
        # Self-attention within query modality
        query_self = self.self_attn(query_modal, query_modal, query_modal, mask)
        
        # Cross-attention to other modality
        cross_output = self.cross_attn(query_modal, key_modal, value_modal, mask)
        
        # Fusion
        output = query_self + cross_output
        return output
