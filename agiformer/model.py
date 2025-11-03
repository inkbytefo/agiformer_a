## Developer: inkbytefo
## Modified: 2025-11-03
"""
AGIFORMER: Main Model Architecture
Enhanced for Gözlemci phase - MoE + Memory + Introspection + Multimodal activated
*** UPDATED WITH GRADIENT CHECKPOINTING SUPPORT ***
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
from torch.utils.checkpoint import checkpoint # <-- DEĞİŞİKLİK: Checkpoint'i import et

from .core.morfo_semantic_tokenizer import MorfoSemanticTokenizer
from .core.attention import MultiHeadAttention, LinearAttention
from .core.base_components import FeedForward, LayerNorm, ResidualConnection
from .core.memory_backbone import UnifiedMemoryBackbone
from .core.multimodal_perception import MultimodalPerceptionCore
from .experts.moe import MixtureOfExperts
from .introspection.self_model import IntrospectionLoop

class AGIFORMERBlock(nn.Module):
    # ... (Bu sınıfın içeriği aynı kalabilir, değişiklik yapmaya gerek yok)
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
        
        if use_linear_attention:
            self.attention = LinearAttention(d_model, n_heads, dropout)
        else:
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        self.attn_residual = ResidualConnection(d_model, dropout)
        
        from .experts.language_expert import LanguageExpert
        from .experts.logic_expert import LogicExpert
        from .experts.spatial_expert import SpatialExpert
        from .experts.causal_expert import CausalExpert
        from .experts.moe import Expert

        custom_experts = []
        for exp_type in expert_types[:n_experts]:
            if exp_type == 'language': custom_experts.append(LanguageExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'logic': custom_experts.append(LogicExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'spatial': custom_experts.append(SpatialExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'causal': custom_experts.append(CausalExpert(d_model, d_ff, n_heads, dropout))
            else: custom_experts.append(Expert(d_model, d_ff, dropout))
        
        self.moe = MixtureOfExperts(
            d_model, n_experts, d_ff, k=2, dropout=dropout,
            custom_experts=custom_experts if custom_experts else None
        )
        
        if use_introspection:
            self.introspection = IntrospectionLoop(d_model, max_iterations=2, n_heads=n_heads, dropout=dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def forward( self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, previous_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        x = self.attn_residual(x, lambda y: self.attention(y, y, y, mask))
        x, moe_info = self.moe(x)
        
        introspection_info = {}
        if self.use_introspection:
            x, introspection_history = self.introspection(x, previous_states)
            introspection_info = introspection_history
        
        block_info = {'type': 'enhanced_block', 'moe': moe_info, 'introspection': introspection_info}
        return x, block_info


class AGIFORMER(nn.Module):
    def __init__(
        self,
        # ... (diğer parametreler aynı kalır)
        vocab_size: int = 256, d_model: int = 768, n_layers: int = 12, n_heads: int = 12,
        d_ff: int = 3072, n_experts: int = 4, expert_types: list = None, memory_size: int = 10000,
        max_seq_len: int = 2048, dropout: float = 0.1, use_linear_attention: bool = False,
        use_memory: bool = True, use_introspection: bool = True, use_multimodal: bool = True,
        # --- DEĞİŞİKLİK: Yeni parametre eklendi ---
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing # <-- DEĞİŞİKLİK: Değeri sakla

        if use_multimodal: self.multimodal_perception = MultimodalPerceptionCore(d_model, vocab_size, 2, n_heads, dropout)
        else: self.multimodal_perception = None
        
        self.tokenizer = MorfoSemanticTokenizer(vocab_size, d_model, kernel_size=3, dropout=dropout)
        
        if use_memory: self.memory = UnifiedMemoryBackbone(d_model, memory_size, max_seq_len // 2, 10)
        else: self.memory = None
        
        if expert_types is None: expert_types = ['language', 'logic', 'spatial', 'causal'][:n_experts]
        
        self.blocks = nn.ModuleList([
            AGIFORMERBlock(
                d_model, n_heads, d_ff, n_experts, expert_types, dropout,
                use_linear_attention, (use_introspection and (i == n_layers - 1))
            ) for i in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.final_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
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
        model_info = {}
        
        if self.multimodal_perception and (image is not None or audio is not None or video is not None):
            modality_embeds, x = self.multimodal_perception(text=text, image=image, audio=audio, video=video)
            model_info['multimodal'] = True
            model_info['modalities'] = list(modality_embeds.keys())
        elif text is not None:
            x = self.tokenizer(text)
            model_info['multimodal'] = False
            model_info['tokenizer'] = 'simplified_morfo_semantic'
        else:
            raise ValueError("At least one input modality must be provided")
            
        if self.memory:
            x, memory_info = self.memory(x, use_working_memory=True, use_longterm_memory=True)
            model_info['memory'] = memory_info
            
        all_block_info = []
        previous_states = None
        
        # --- DEĞİŞİKLİK: Gradient Checkpointing burada uygulanır ---
        for i, block in enumerate(self.blocks):
            # Eğer eğitim modundaysak ve checkpointing aktifse, checkpoint'i kullan
            if self.training and self.use_gradient_checkpointing:
                # `checkpoint` fonksiyonu, `block`'un forward'ını çağırır
                # ama ara aktivasyonları saklamaz, bunun yerine geri yayılımda yeniden hesaplar.
                # Bu, bellekten tasarruf sağlar.
                x, block_info = checkpoint(block, x, mask, previous_states, use_reentrant=False)
            else:
                # Normal forward pass (eğitimde değilsek veya checkpointing kapalıysa)
                x, block_info = block(x, mask=mask, previous_states=previous_states)
            
            all_block_info.append(block_info)
            if previous_states is not None: previous_states = torch.cat([previous_states, x], dim=1)
            else: previous_states = x
        # --- BİTTİ ---
            
        model_info['blocks'] = all_block_info
        x = self.final_norm(x)
        
        if return_embeddings:
            return x, model_info
            
        logits = self.output_proj(x)
        return logits, model_info

    def reset_memory(self):
        if self.memory: self.memory.reset_working_memory()

    def generate(
        self,
        text: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0,
        top_k: int = 50, top_p: float = 0.9
    ) -> torch.Tensor:
        self.eval()
        generated = text.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.forward(text=generated)
                next_token_logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        return generated
