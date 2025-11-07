## Developer: inkbytefo
## Modified: 2025-11-04

"""
Unified Memory Backbone
Manages both short-term (working memory) and long-term (episodic/semantic memory)
*** UPDATED WITH .detach() TO PREVENT RUNTIMEERROR ***
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

# (MemoryBank sınıfı aynı kalır)
class MemoryBank(nn.Module):
    def __init__(self, memory_size: int, d_model: int):
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        self.memory = nn.Parameter(torch.randn(memory_size, d_model))
        self.read_head = nn.Linear(d_model, memory_size)
        self.write_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.update_gate = nn.Sequential(nn.Linear(d_model * 2, 1), nn.Sigmoid())
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add input validation
        if query.dim() != 3:
            raise ValueError(f"Expected query to be 3D tensor, got {query.dim()}D")
        
        batch_size, seq_len, d_model = query.size()
        
        # Validate dimensions match
        if d_model != self.d_model:
            raise ValueError(f"Query d_model ({d_model}) doesn't match memory d_model ({self.d_model})")
        
        # Clamp seq_len to prevent out of bounds
        max_seq_len = 1024  # Reasonable limit
        if seq_len > max_seq_len:
            query = query[:, :max_seq_len, :]
            seq_len = max_seq_len
        
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add bounds checking before matrix multiplication
        if memory_expanded.size(1) != self.memory_size:
            raise ValueError(f"Memory size mismatch: expected {self.memory_size}, got {memory_expanded.size(1)}")
        
        # Ensure all tensors are on the same device
        if query.device != memory_expanded.device:
            memory_expanded = memory_expanded.to(query.device)
        
        similarity = torch.matmul(query, memory_expanded.transpose(1, 2))
        attention_weights = F.softmax(similarity / math.sqrt(self.d_model), dim=-1)
        retrieved = torch.matmul(attention_weights, memory_expanded)
        return retrieved, attention_weights
    def write(self, key: torch.Tensor, value: torch.Tensor):
        write_weights = F.softmax(self.read_head(key), dim=-1)
        values_to_write = self.write_head(value)
        write_weights_sum = write_weights.sum(dim=(0, 1))
        write_weights_normalized = write_weights / (write_weights_sum.unsqueeze(0).unsqueeze(0) + 1e-8)
        memory_update = torch.matmul(write_weights_normalized.transpose(1, 2), values_to_write).mean(dim=0)
        gate_input = torch.cat([self.memory.data, memory_update], dim=-1)
        gate = self.update_gate(gate_input)
        self.memory.data = self.memory.data * (1 - gate) + memory_update * gate

class WorkingMemory(nn.Module):
    def __init__(self, d_model: int, max_segment_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_segment_len = max_segment_len
        self.register_buffer('segment_memory', None, persistent=False) # persistent=False önerilir

    def update_segment_memory(self, hidden_states: torch.Tensor):
        if self.segment_memory is None:
            # --- DEĞİŞİKLİK: Belleğe ilk kayıtta da detach() kullan ---
            self.segment_memory = hidden_states.detach()
        else:
            combined = torch.cat([self.segment_memory, hidden_states], dim=1)
            if combined.size(1) > self.max_segment_len:
                combined = combined[:, -self.max_segment_len:, :]
            # --- DEĞİŞİKLİK: Gradyan geçmişini ayırarak belleğe kaydet ---
            self.segment_memory = combined.detach()
            # --- BİTTİ ---

    def get_context(self) -> Optional[torch.Tensor]:
        return self.segment_memory

    def reset(self):
        self.segment_memory = None

class UnifiedMemoryBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 768, memory_size: int = 10000, max_segment_len: int = 512,
        memory_update_freq: int = 10
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_update_freq = memory_update_freq
        self.working_memory = WorkingMemory(d_model, max_segment_len)
        self.long_term_memory = MemoryBank(memory_size, d_model)
        self.memory_fusion = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.working_gate = nn.Linear(d_model * 2, d_model) # Boyut düzeltmesi
        self.longterm_gate = nn.Linear(d_model * 2, d_model) # Boyut düzeltmesi
        self._step_count = 0

    def forward(
        self, current_states: torch.Tensor, use_working_memory: bool = True,
        use_longterm_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len, _ = current_states.size()
        memory_info = {}

        # Orijinal girdiyi sakla
        original_states = current_states

        # Çalışma belleğinden bağlam al
        if use_working_memory:
            working_context = self.working_memory.get_context()
            if working_context is not None:
                context_len = working_context.size(1)
                gate_input = torch.cat([current_states, working_context[:, -1:, :].expand(-1, seq_len, -1)], dim=-1)
                working_gate = torch.sigmoid(self.working_gate(gate_input))
                current_states = current_states + working_gate * working_context[:, -1:, :].expand(-1, seq_len, -1)

        # Uzun süreli bellekten oku
        if use_longterm_memory:
            longterm_context, read_weights = self.long_term_memory.read(current_states)
            memory_info['memory_read_weights'] = read_weights.mean().item()
            gate_input = torch.cat([current_states, longterm_context], dim=-1)
            longterm_gate = torch.sigmoid(self.longterm_gate(gate_input))
            current_states = current_states + longterm_gate * longterm_context

        # Çalışma belleğini GÜNCELLE
        if use_working_memory:
            self.working_memory.update_segment_memory(original_states)

        # Uzun süreli belleği GÜNCELLE
        self._step_count += 1
        if use_longterm_memory and self.training and self._step_count % self.memory_update_freq == 0:
            self.long_term_memory.write(original_states, original_states)

        memory_info['step_count'] = self._step_count
        return current_states, memory_info

    def reset_working_memory(self):
        self.working_memory.reset()
