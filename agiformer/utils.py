"""
Utility functions for AGIFORMER
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import math


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create causal (lower triangular) mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


def create_padding_mask(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask from sequence lengths
    
    Args:
        seq_lengths: [batch_size] - actual length of each sequence
        max_len: Maximum sequence length
    Returns:
        mask: [batch_size, 1, 1, max_len] - 1 for valid, 0 for padding
    """
    batch_size = seq_lengths.size(0)
    mask = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < seq_lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    return mask.float()


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in megabytes"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb


class WarmupScheduler:
    """Learning rate scheduler with warmup"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        d_model: int,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.factor = factor
        self._step = 0
        
    def step(self):
        """Update learning rate"""
        self._step += 1
        lr = self.factor * (self.d_model ** -0.5) * min(
            self._step ** -0.5,
            self._step * (self.warmup_steps ** -1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {'step': self._step}
    
    def load_state_dict(self, state_dict):
        self._step = state_dict['step']


def format_number(num: int) -> str:
    """Format large numbers (e.g., 1000000 -> 1M)"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

