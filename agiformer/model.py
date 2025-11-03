"""
AGIFORMER: Main Model Architecture
Enhanced for Gözlemci phase - MoE + Memory + Introspection + Multimodal activated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union

from .core.morfo_semantic_tokenizer import MorfoSemanticTokenizer
from .core.attention import MultiHeadAttention, LinearAttention
from .core.base_components import FeedForward, LayerNorm, ResidualConnection
from .core.memory_backbone import UnifiedMemoryBackbone
from .core.multimodal_perception import MultimodalPerceptionCore
from .experts.moe import MixtureOfExperts
from .experts.language_expert import LanguageExpert
from .experts.logic_expert import LogicExpert
from .experts.spatial_expert import SpatialExpert
from .experts.causal_expert import CausalExpert
from .introspection.self_model import IntrospectionLoop


class AGIFORMERBlock(nn.Module):
    """
    Enhanced AGIFORMER transformer block for Gözlemci phase
    Includes attention, MoE experts, and introspection capability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int,
        expert_types: list = None,
        dropout: float = 0.1,
        use_linear_attention: bool = False,
        use_introspection: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.use_introspection = use_introspection
        
        # Attention mechanism
        if use_linear_attention:
            self.attention = LinearAttention(d_model, n_heads, dropout)
        else:
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.attn_residual = ResidualConnection(d_model, dropout)
        
        # Mixture of Experts
        if expert_types is None:
            expert_types = ['standard'] * n_experts
        
        custom_experts = []
        for exp_type in expert_types[:n_experts]:
            if exp_type == 'language':
                custom_experts.append(LanguageExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'logic':
                custom_experts.append(LogicExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'spatial':
                custom_experts.append(SpatialExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'causal':
                custom_experts.append(CausalExpert(d_model, d_ff, n_heads, dropout))
            else:
                # Standard expert
                from .experts.moe import Expert
                custom_experts.append(Expert(d_model, d_ff, dropout))
        
        self.moe = MixtureOfExperts(
            d_model,
            n_experts,
            d_ff,
            k=2,  # Top-2 experts
            dropout=dropout,
            custom_experts=custom_experts if custom_experts else None
        )
        
        # Introspection loop (only for last layer)
        if use_introspection:
            self.introspection = IntrospectionLoop(
                d_model=d_model,
                max_iterations=2,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # Layer norms
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        previous_states: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            previous_states: Optional previous hidden states for introspection
        
        Returns:
            x: [batch_size, seq_len, d_model]
            block_info: Dictionary with block information
        """
        # Self-attention with residual connection
        x = self.attn_residual(x, lambda x: self.attention(x, x, x, mask))
        
        # Mixture of Experts
        x, moe_info = self.moe(x)
        
        # Introspection (if enabled)
        introspection_info = {}
        if self.use_introspection:
            x, introspection_history = self.introspection(x, previous_states)
            introspection_info = introspection_history
        
        block_info = {
            'type': 'enhanced_block',
            'moe': moe_info,
            'introspection': introspection_info
        }
        
        return x, block_info


class AGIFORMER(nn.Module):
    """
    AGIFORMER: Enhanced main model architecture for Gözlemci phase
    
    Components:
    1. Multimodal Perception Core (Text + Image + Audio + Video)
    2. Morfo-Semantic Tokenizer (for text-only input)
    3. Memory system (Working Memory + Long-term Memory)
    4. Stack of AGIFORMER Blocks (with MoE + Introspection)
    5. Output projection
    """
    
    def __init__(
        self,
        vocab_size: int = 256,  # Character vocabulary
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        n_experts: int = 4,  # Activated for Düşünür
        expert_types: list = None,
        memory_size: int = 10000,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_linear_attention: bool = False,
        use_memory: bool = True,  # Activated for Düşünür
        use_introspection: bool = True,  # Activated for Gözlemci
        use_multimodal: bool = True  # Activated for Gözlemci
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Multimodal Perception Core
        if use_multimodal:
            self.multimodal_perception = MultimodalPerceptionCore(
                d_model=d_model,
                vocab_size=vocab_size,
                n_cross_modal_layers=2,
                dropout=dropout
            )
        else:
            self.multimodal_perception = None
        
        # Morfo-Semantic Tokenizer (for text-only input)
        self.tokenizer = MorfoSemanticTokenizer(
            vocab_size=vocab_size,
            d_model=d_model,
            kernel_size=3,  # Simplified to single kernel size
            dropout=dropout
        )
        
        # Memory System (Working + Long-term)
        if use_memory:
            self.memory = UnifiedMemoryBackbone(
                d_model=d_model,
                memory_size=memory_size,
                max_segment_len=max_seq_len // 2,
                memory_update_freq=10
            )
        else:
            self.memory = None
        
        # AGIFORMER Blocks (enhanced with MoE + Introspection)
        if expert_types is None:
            expert_types = ['language', 'logic', 'spatial', 'causal'][:n_experts]
        
        self.blocks = nn.ModuleList([
            AGIFORMERBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_experts=n_experts,
                expert_types=expert_types,
                dropout=dropout,
                use_linear_attention=use_linear_attention,
                use_introspection=(use_introspection and (i == n_layers - 1))  # Only last layer
            )
            for i in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Layer norm for final output
        self.final_norm = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass
        
        Args:
            text: [batch_size, seq_len] - character IDs
            image: [batch_size, 3, H, W] - image tensor
            audio: [batch_size, audio_length] - audio waveform
            video: [batch_size, num_frames, 3, H, W] - video tensor
            mask: Optional attention mask
            return_embeddings: Whether to return intermediate embeddings
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            info: (optional) Dict with model information
        """
        model_info = {}
        
        # Multimodal perception or text tokenization
        if self.multimodal_perception is not None and (image is not None or audio is not None or video is not None):
            # Multimodal input
            modality_embeds, unified_embed = self.multimodal_perception(
                text=text,
                image=image,
                audio=audio,
                video=video
            )
            x = unified_embed
            model_info['multimodal'] = True
            model_info['modalities'] = list(modality_embeds.keys())
        elif text is not None:
            # Text-only input
            x = self.tokenizer(text)  # [batch, seq_len, d_model]
            model_info['multimodal'] = False
            model_info['tokenizer'] = 'simplified_morfo_semantic'
        else:
            raise ValueError("At least one input modality must be provided")
        
        # Memory system (Working + Long-term)
        if self.memory is not None:
            x, memory_info = self.memory(
                x, 
                use_working_memory=True, 
                use_longterm_memory=True
            )
            model_info['memory'] = memory_info
        
        # AGIFORMER blocks
        all_block_info = []
        previous_states = None
        
        for i, block in enumerate(self.blocks):
            x, block_info = block(x, mask=mask, previous_states=previous_states)
            all_block_info.append(block_info)
            
            # Update previous states for introspection
            if previous_states is not None:
                previous_states = torch.cat([previous_states, x], dim=1)
            else:
                previous_states = x
        
        model_info['blocks'] = all_block_info
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        if return_embeddings:
            return x, model_info
        
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        return logits, model_info
    
    def reset_memory(self):
        """Reset memory system"""
        if self.memory is not None:
            self.memory.reset_working_memory()
    
    def generate(
        self,
        text: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            text: [batch_size, seq_len] - initial text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        generated = text.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits
                logits, _ = self.forward(text=generated)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
