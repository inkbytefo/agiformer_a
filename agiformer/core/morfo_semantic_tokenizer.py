"""
Learnable Morfo-Semantic Tokenizer
Simplified Charformer-inspired implementation for Kıvılcım phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GBSTBlock(nn.Module):
    """
    Simplified Gradient-Based Subword Tokenization Block
    Charformer-inspired: single convolution for character n-gram learning
    """
    
    def __init__(
        self,
        vocab_size: int = 256,  # Character vocabulary size
        d_model: int = 768,
        kernel_size: int = 3,  # Fixed kernel size for simplicity
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, d_model)
        
        # Simplified convolution for n-gram learning (no LayerNorm inside conv)
        self.conv_layer = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        
        # Layer normalization after convolution
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Simple output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, char_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            char_ids: [batch_size, seq_len] - character IDs
        
        Returns:
            token_embeddings: [batch_size, seq_len, d_model] - enriched character embeddings
            token_boundaries: [batch_size, seq_len] - dummy boundaries (all True)
        """
        batch_size, seq_len = char_ids.size()
        
        # Character embeddings: [batch, seq_len, d_model]
        char_emb = self.char_embedding(char_ids)
        char_emb = self.dropout(char_emb)
        
        # Transpose for conv1d: [batch, d_model, seq_len]
        char_emb_t = char_emb.transpose(1, 2)
        
        # Apply convolution to capture character n-grams
        conv_out = self.conv_layer(char_emb_t)  # [batch, d_model, seq_len]
        
        # Apply LayerNorm after convolution
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, d_model]
        conv_out = self.layer_norm(conv_out)
        
        # Transpose back for output projection
        conv_out = conv_out.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # Transpose back to original format: [batch, seq_len, d_model]
        token_embeddings = conv_out.transpose(1, 2)
        
        # Apply activation and dropout
        token_embeddings = F.gelu(token_embeddings)
        token_embeddings = self.dropout(token_embeddings)
        
        # Final projection
        token_embeddings = self.output_proj(token_embeddings)
        
        # Dummy token boundaries (all positions are boundaries for now)
        token_boundaries = torch.ones(
            batch_size, seq_len,
            dtype=torch.bool, device=token_embeddings.device
        )
        
        return token_embeddings, token_boundaries


class MorfoSemanticTokenizer(nn.Module):
    """
    Simplified Morfo-Semantic Tokenizer
    Single GBST block for Kıvılcım phase
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 768,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Single GBST block for simplicity
        self.gbst_block = GBSTBlock(vocab_size, d_model, kernel_size, dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        char_ids: torch.Tensor,
        return_boundaries: bool = False
    ) -> torch.Tensor:
        """
        Args:
            char_ids: [batch_size, seq_len] - character IDs
            return_boundaries: Whether to return token boundaries
        
        Returns:
            token_embeddings: [batch_size, seq_len, d_model] - enriched token embeddings
            token_boundaries: (optional) [batch_size, seq_len] - predicted boundaries
        """
        x, token_boundaries = self.gbst_block(char_ids)
        
        # Normalize
        x = self.layer_norm(x)
        
        if return_boundaries:
            return x, token_boundaries
        return x
