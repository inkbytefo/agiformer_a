"""
Meta-Learning Components
Enable the model to learn how to learn and adapt quickly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from ..core.attention import MultiHeadAttention


class MetaLearner(nn.Module):
    """
    Meta-Learner: Learns optimal learning strategies
    Implements MAML (Model-Agnostic Meta-Learning) inspired approach
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Meta-parameter network (learns how to update parameters)
        self.meta_param_net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Task embedding network
        self.task_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Fast adaptation network
        self.fast_adapt = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Attention for task similarity
        self.task_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        current_state: torch.Tensor,
        task_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Meta-learning forward pass
        
        Args:
            current_state: [batch_size, seq_len, d_model]
            task_context: [batch_size, task_seq_len, d_model]
        
        Returns:
            adapted_state: [batch_size, seq_len, d_model]
            meta_info: Dict with meta-learning statistics
        """
        meta_info = {}
        
        # Encode task context
        if task_context is not None:
            task_embedding = self.task_encoder(task_context.mean(dim=1, keepdim=True))  # [batch, 1, d_model]
            
            # Attend to task context
            task_attended = self.task_attention(
                current_state,
                task_context,
                task_context
            )
            
            # Fast adaptation
            adaptation_input = torch.cat([current_state, task_attended], dim=-1)
            adaptation = self.fast_adapt(adaptation_input)
            
            adapted_state = self.norm(current_state + self.dropout(adaptation))
        else:
            adapted_state = current_state
        
        meta_info['has_task_context'] = task_context is not None
        
        return adapted_state, meta_info
    
    def compute_meta_gradient(
        self,
        loss: torch.Tensor,
        parameters: nn.ParameterList
    ) -> Dict[str, torch.Tensor]:
        """
        Compute meta-gradients for fast adaptation
        """
        # This would be used in MAML-style training
        # For now, return empty dict
        return {}

