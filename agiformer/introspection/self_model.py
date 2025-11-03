"""
Self-Model and Introspection Loop
Allows the model to observe and reason about its own thought processes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from ..core.attention import MultiHeadAttention


class ThoughtTrace(nn.Module):
    """
    Represents a trace of the model's thought process
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.traces = []
        
    def add_step(self, hidden_state: torch.Tensor, metadata: Dict = None):
        """Add a step to the thought trace"""
        self.traces.append({
            'hidden_state': hidden_state.detach().clone(),
            'metadata': metadata or {}
        })
    
    def get_trace(self) -> List[Dict]:
        """Get all traces"""
        return self.traces
    
    def clear(self):
        """Clear traces"""
        self.traces = []


class SelfModel(nn.Module):
    """
    Self-Model: The model's representation of itself
    Allows introspection and meta-reasoning
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Attention to observe own states
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Meta-reasoning network
        self.meta_reasoner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # current state + past states
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Error detection network
        self.error_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Thought trace
        self.thought_trace = None
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        current_state: torch.Tensor,
        previous_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Observe and reason about current state
        
        Args:
            current_state: [batch_size, seq_len, d_model]
            previous_states: [batch_size, prev_seq_len, d_model]
        
        Returns:
            introspected_state: [batch_size, seq_len, d_model]
            introspection_info: Dict with self-observation statistics
        """
        introspection_info = {}
        
        # Self-attention
        residual = current_state
        current_state_norm = self.norm(current_state)
        introspected = self.self_attention(
            current_state_norm,
            current_state_norm,
            current_state_norm
        )
        introspected = residual + self.dropout(introspected)
        
        # Meta-reasoning with previous states
        if previous_states is not None:
            # Combine current and previous
            combined = torch.cat([previous_states, introspected], dim=1)
            meta_input = torch.cat([
                introspected,
                previous_states.mean(dim=1, keepdim=True).expand_as(introspected)
            ], dim=-1)
            meta_out = self.meta_reasoner(meta_input)
            introspected = introspected + meta_out
        
        # Error detection
        error_scores = self.error_detector(introspected)  # [batch, seq_len, 1]
        introspection_info['error_scores'] = error_scores  # Keep tensor, not mean
        
        # Confidence estimation
        confidence_scores = self.confidence_estimator(introspected)  # [batch, seq_len, 1]
        introspection_info['confidence_scores'] = confidence_scores  # Keep tensor, not mean
        
        introspection_info['needs_correction'] = error_scores.mean().item() > 0.5
        introspection_info['high_confidence'] = confidence_scores.mean().item() > 0.7
        
        return introspected, introspection_info


class IntrospectionLoop(nn.Module):
    """
    Introspection Loop: Iterative self-reflection and improvement
    """
    
    def __init__(
        self,
        d_model: int,
        max_iterations: int = 3,
        n_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_iterations = max_iterations
        
        # Self-model
        self.self_model = SelfModel(d_model, n_heads, dropout)
        
        # Correction network (for self-correction)
        self.correction_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # current + error signal
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Decision network (when to stop iterating)
        self.decision_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        initial_state: torch.Tensor,
        previous_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Iterative introspection loop
        
        Args:
            initial_state: [batch_size, seq_len, d_model]
            previous_states: [batch_size, prev_seq_len, d_model]
        
        Returns:
            final_state: [batch_size, seq_len, d_model]
            introspection_history: Dict with iteration history
        """
        current_state = initial_state
        introspection_history = {
            'iterations': [],
            'final_error': None,
            'final_confidence': None
        }
        
        for iteration in range(self.max_iterations):
            # Self-observation
            introspected, introspection_info = self.self_model(
                current_state,
                previous_states
            )
            
            introspection_history['iterations'].append({
                'iteration': iteration,
                'error_score': introspection_info['error_scores'].mean().item(),
                'confidence': introspection_info['confidence_scores'].mean().item()
            })
            
            # Check if correction is needed
            if introspection_info['needs_correction']:
                # Apply correction using error signal tensor
                error_signal = introspection_info['error_scores']  # [batch, seq_len, 1]
                correction_input = torch.cat([
                    current_state,
                    error_signal.expand(-1, -1, current_state.size(-1))
                ], dim=-1)
                correction = self.correction_net(correction_input)
                current_state = self.norm(current_state + correction)
            else:
                current_state = introspected
            
            # Decision: should we continue?
            continue_prob = self.decision_net(current_state.mean(dim=1))  # [batch, 1]
            if continue_prob.mean().item() < 0.5:
                # Model is confident, stop early
                break
        
        introspection_history['final_error'] = introspection_info['error_scores'].mean().item()
        introspection_history['final_confidence'] = introspection_info['confidence_scores'].mean().item()
        introspection_history['num_iterations'] = len(introspection_history['iterations'])
        
        return current_state, introspection_history
