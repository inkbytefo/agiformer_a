"""
Unified Memory Backbone
Manages both short-term (working memory) and long-term (episodic/semantic memory)
Inspired by Transformer-XL's segment-level recurrence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class MemoryBank(nn.Module):
    """
    External memory bank for long-term storage
    Implements a differentiable memory mechanism
    """
    
    def __init__(self, memory_size: int, d_model: int):
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        
        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(memory_size, d_model))
        
        # Memory access mechanisms
        self.read_head = nn.Linear(d_model, memory_size)
        self.write_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid()
        )
        
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using content-based addressing
        
        Args:
            query: [batch_size, seq_len, d_model]
        Returns:
            retrieved: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Compute attention weights over memory
        # Memory: [memory_size, d_model], Query: [batch, seq_len, d_model]
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute similarity
        similarity = torch.matmul(query, memory_expanded.transpose(1, 2))  # [batch, seq_len, memory_size]
        attention_weights = F.softmax(similarity / math.sqrt(self.d_model), dim=-1)
        
        # Retrieve
        retrieved = torch.matmul(attention_weights, memory_expanded)  # [batch, seq_len, d_model]
        
        return retrieved, attention_weights
    
    def write(self, key: torch.Tensor, value: torch.Tensor):
        """
        Write to memory
        
        Args:
            key: [batch_size, seq_len, d_model] - what to write
            value: [batch_size, seq_len, d_model] - content to write
        """
        # Compute write locations
        write_weights = F.softmax(self.read_head(key), dim=-1)  # [batch, seq_len, memory_size]
        
        # Prepare values to write
        values_to_write = self.write_head(value)  # [batch, seq_len, d_model]
        
        # Update memory (in-place, but differentiable)
        # Aggregate writes across batch and sequence
        write_weights_sum = write_weights.sum(dim=0).sum(dim=0)  # [memory_size]
        write_weights_normalized = write_weights / (write_weights_sum.unsqueeze(0).unsqueeze(0) + 1e-8)
        
        # Weighted update
        memory_update = torch.matmul(
            write_weights_normalized.transpose(1, 2),  # [batch, memory_size, seq_len]
            values_to_write  # [batch, seq_len, d_model]
        )  # [batch, memory_size, d_model]
        
        # Update memory (sum across batch)
        memory_update = memory_update.mean(dim=0)  # [memory_size, d_model]
        
        # Gated update
        current_memory = self.memory.data
        gate_input = torch.cat([current_memory, memory_update], dim=-1)  # [memory_size, 2*d_model]
        gate = self.update_gate(gate_input)  # [memory_size, 1]
        
        self.memory.data = current_memory * (1 - gate) + memory_update * gate


class WorkingMemory(nn.Module):
    """
    Short-term working memory (recurrent segment-level memory)
    Similar to Transformer-XL's recurrence mechanism
    """
    
    def __init__(self, d_model: int, max_segment_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_segment_len = max_segment_len
        
        # Segment-level state
        self.register_buffer('segment_memory', None)
        
    def update_segment_memory(self, hidden_states: torch.Tensor):
        """
        Update segment memory with new hidden states
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
        """
        if self.segment_memory is None:
            self.segment_memory = hidden_states
        else:
            # Concatenate with existing memory
            # Keep only last max_segment_len states
            combined = torch.cat([self.segment_memory, hidden_states], dim=1)
            if combined.size(1) > self.max_segment_len:
                combined = combined[:, -self.max_segment_len:, :]
            self.segment_memory = combined
    
    def get_context(self, seq_len: int) -> Optional[torch.Tensor]:
        """
        Get context from segment memory
        
        Args:
            seq_len: Length of current sequence
        Returns:
            context: [batch_size, context_len, d_model] or None
        """
        if self.segment_memory is None:
            return None
        
        # Return last N states as context
        context_len = min(self.segment_memory.size(1), seq_len)
        return self.segment_memory[:, -context_len:, :]
    
    def reset(self):
        """Reset segment memory"""
        self.segment_memory = None


class UnifiedMemoryBackbone(nn.Module):
    """
    Unified memory system combining working memory and long-term memory
    """
    
    def __init__(
        self,
        d_model: int = 768,
        memory_size: int = 10000,
        max_segment_len: int = 512,
        memory_update_freq: int = 10
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_update_freq = memory_update_freq
        
        # Memory components
        self.working_memory = WorkingMemory(d_model, max_segment_len)
        self.long_term_memory = MemoryBank(memory_size, d_model)
        
        # Memory fusion
        self.memory_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # current + working + long-term
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Memory gates
        self.working_gate = nn.Linear(d_model * 2, 1)
        self.longterm_gate = nn.Linear(d_model * 2, 1)
        
        self._step_count = 0
        
    def forward(
        self,
        current_states: torch.Tensor,
        use_working_memory: bool = True,
        use_longterm_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through memory system
        
        Args:
            current_states: [batch_size, seq_len, d_model]
            use_working_memory: Whether to use working memory
            use_longterm_memory: Whether to use long-term memory
        
        Returns:
            enhanced_states: [batch_size, seq_len, d_model]
            memory_info: Dict with memory statistics
        """
        batch_size, seq_len, _ = current_states.size()
        
        memory_info = {}
        
        # Working memory (short-term)
        working_context = None
        if use_working_memory:
            working_context = self.working_memory.get_context(seq_len)
            
            if working_context is not None:
                # Gate working memory
                gate_input = torch.cat([current_states, working_context[:, -1:, :].expand(-1, seq_len, -1)], dim=-1)
                working_gate = torch.sigmoid(self.working_gate(gate_input))
                
                # Use first token of working context as context
                working_context_rep = working_context[:, -1:, :].expand(-1, seq_len, -1)
                current_states = current_states + working_gate * working_context_rep
        
        # Long-term memory (retrieval)
        longterm_context = None
        if use_longterm_memory:
            longterm_context, read_weights = self.long_term_memory.read(current_states)
            memory_info['memory_read_weights'] = read_weights.mean().item()
            
            # Gate long-term memory
            gate_input = torch.cat([current_states, longterm_context], dim=-1)
            longterm_gate = torch.sigmoid(self.longterm_gate(gate_input))
            
            current_states = current_states + longterm_gate * longterm_context
        
        # Fusion
        if working_context is not None and longterm_context is not None:
            # Combine all three
            combined = torch.cat([current_states, working_context[:, -1:, :].expand(-1, seq_len, -1), longterm_context], dim=-1)
            enhanced_states = self.memory_fusion(combined)
        elif working_context is not None:
            combined = torch.cat([current_states, working_context[:, -1:, :].expand(-1, seq_len, -1), current_states], dim=-1)
            enhanced_states = self.memory_fusion(combined)
        elif longterm_context is not None:
            combined = torch.cat([current_states, current_states, longterm_context], dim=-1)
            enhanced_states = self.memory_fusion(combined)
        else:
            enhanced_states = current_states
        
        # Update working memory
        if use_working_memory:
            self.working_memory.update_segment_memory(current_states)
        
        # Update long-term memory (periodically)
        self._step_count += 1
        if use_longterm_memory and self._step_count % self.memory_update_freq == 0:
            # Write important states to long-term memory
            # Use a simple heuristic: states with high activation
            importance = current_states.norm(dim=-1)  # [batch, seq_len]
            top_k_indices = torch.topk(importance.flatten(), k=min(100, importance.numel()), dim=0)[1]
            batch_indices = top_k_indices // seq_len
            seq_indices = top_k_indices % seq_len
            
            selected_states = current_states[batch_indices, seq_indices, :].unsqueeze(1)
            self.long_term_memory.write(selected_states, selected_states)
        
        memory_info['step_count'] = self._step_count
        
        return enhanced_states, memory_info
    
    def reset_working_memory(self):
        """Reset working memory (e.g., for new sequence)"""
        self.working_memory.reset()
    
    def clear_longterm_memory(self):
        """Clear long-term memory (use with caution)"""
        with torch.no_grad():
            self.long_term_memory.memory.data = torch.randn_like(self.long_term_memory.memory.data)

