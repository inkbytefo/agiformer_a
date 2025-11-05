"""
Mixture of Experts (MoE) System
Dynamically routes inputs to specialized expert networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


class ExpertRouter(nn.Module):
    """
    Router that decides which experts to activate for each token
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        k: int = 2,  # Top-k experts to select
        load_balancing_loss_weight: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = min(k, n_experts)  # Ensure k doesn't exceed n_experts
        self.load_balancing_loss_weight = load_balancing_loss_weight
        
        # Router network
        self.router = nn.Linear(d_model, n_experts)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Route inputs to experts

        Args:
            hidden_states: [batch_size, seq_len, d_model]
            routing_bias: Optional[torch.Tensor] - external bias for routing. Shape: [batch_size, n_experts]

        Returns:
            expert_weights: [batch_size, seq_len, n_experts] - routing weights
            expert_indices: [batch_size, seq_len, k] - indices of top-k experts
            router_info: Dict with routing statistics
        """
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)

        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, n_experts]

        if routing_bias is not None:
            # routing_bias shape: [batch, n_experts]
            # [batch, 1, n_experts] olarak genişletip ekle
            router_logits = router_logits + routing_bias.unsqueeze(1)

        # Get top-k experts
        expert_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(expert_weights, k=self.k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute load balancing loss (encourages uniform expert usage)
        router_info = {}
        if self.training:
            # Average routing weights across batch and sequence
            avg_router_weights = expert_weights.mean(dim=(0, 1))  # [n_experts]
            # Load balancing loss: encourage uniform distribution
            load_balancing_loss = self.load_balancing_loss_weight * (
                self.n_experts * (avg_router_weights ** 2).sum()
            )
            router_info['load_balancing_loss'] = load_balancing_loss

        router_info['expert_usage'] = top_k_indices.float().mean(dim=(0, 1)).cpu().tolist()
        router_info['avg_router_confidence'] = top_k_weights.mean().item()

        return top_k_weights, top_k_indices, router_info


class Expert(nn.Module):
    """
    Base class for expert networks
    Each expert is a specialized feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        return self.w2(self.dropout(self.activation(self.w1(x))))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer
    Routes inputs to multiple experts and combines their outputs
    """
    
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        d_ff: int = None,
        k: int = 2,
        dropout: float = 0.1,
        load_balancing_loss_weight: float = 0.01,
        custom_experts: Optional[List[nn.Module]] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.d_ff = d_ff or (d_model * 4)
        
        # Router
        self.router = ExpertRouter(d_model, n_experts, k, load_balancing_loss_weight)
        
        # Experts
        if custom_experts is not None:
            assert len(custom_experts) == n_experts
            self.experts = nn.ModuleList(custom_experts)
        else:
            # Standard feed-forward experts
            self.experts = nn.ModuleList([
                Expert(d_model, self.d_ff, dropout)
                for _ in range(n_experts)
            ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            routing_bias: Optional[torch.Tensor] - external bias for routing. Shape: [batch_size, n_experts]

        Returns:
            output: [batch_size, seq_len, d_model]
            expert_info: Dict with expert statistics
        """
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)

        # Route to experts
        expert_weights, expert_indices, router_info = self.router(hidden_states, routing_bias=routing_bias)
        
        # Process through each expert and combine with weights
        expert_outputs = []
        expert_infos = []
        for i, expert in enumerate(self.experts):
            expert_result = expert(hidden_states)
            if isinstance(expert_result, tuple):
                expert_out, expert_info = expert_result
                expert_infos.append(expert_info)
            else:
                expert_out = expert_result
                expert_infos.append({})
            expert_outputs.append(expert_out)
        
        # Stack expert outputs: [n_experts, batch, seq_len, d_model]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # Gather outputs for top-k experts
        # expert_indices: [batch, seq_len, k]
        batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).unsqueeze(2)
        seq_indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).unsqueeze(2)
        
        # Gather: [batch, seq_len, k, d_model]
        gathered_outputs = expert_outputs[expert_indices, batch_indices, seq_indices]
        
        # Apply weights: [batch, seq_len, k, d_model]
        weighted_outputs = gathered_outputs * expert_weights.unsqueeze(-1)
        
        # Sum over k experts: [batch, seq_len, d_model]
        output = weighted_outputs.sum(dim=2)
        
        # Residual connection
        output = output + hidden_states
        
        expert_info = {
            'router_info': router_info,
            'num_active_experts': expert_indices.size(-1),
            'expert_details': expert_infos
        }
        
        return output, expert_info
