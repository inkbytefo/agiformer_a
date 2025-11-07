## Developer: inkbytefo
## Modified: 2025-11-07
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

from agiformer.language.tokenizer import MorphoPiece
from .core.attention import MultiHeadAttention, LinearAttention
from .core.base_components import FeedForward, LayerNorm, ResidualConnection
from .core.memory_backbone import UnifiedMemoryBackbone
from .core.multimodal_perception import MultimodalPerceptionCore
from .experts.moe import MixtureOfExperts
from .experts.knowledge_graph import GlobalKnowledgeGraph
from .experts.relations import NUM_RELATIONS
from .experts.task_classifier import TaskTypeClassifier, EXPERT_DOMAINS
from .introspection.self_model import IntrospectionLoop

class AGIFORMERBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_experts: int,
        expert_types: list = None,
        dropout: float = 0.1,
        use_linear_attention: bool = False,
        use_introspection: bool = False,
        use_agglutinative_attention: bool = True,
        global_knowledge_graph=None
    ):
        super().__init__()
        self.d_model = d_model
        self.use_introspection = use_introspection
        self.use_agglutinative_attention = use_agglutinative_attention
        # CRITICAL FIX: Ensure k doesn't exceed n_experts
        self.n_experts = n_experts
        self.k = min(2, n_experts)  # Top-k should not exceed number of experts

        if use_linear_attention:
            self.attention = LinearAttention(d_model, n_heads, dropout)
        else:
            self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        self.attn_residual = ResidualConnection(d_model, dropout)

        # Yeni: Görev Türü Sınıflandırıcı
        self.task_classifier = TaskTypeClassifier(d_model)

        # Uzmanların hangi alana ait olduğunu belirten bir harita
        self.expert_to_domain_map = self.map_experts_to_domains(expert_types)

        from .experts.language_expert import LanguageExpert
        from .experts.logic_expert import LogicExpert
        from .experts.spatial_expert import SpatialExpert
        from .experts.causal_expert import CausalExpert
        from .experts.neuro_symbolic_expert import NeuroSymbolicExpert
        from .experts.moe import Expert

        custom_experts = []
        for exp_type in expert_types[:n_experts]:
            if exp_type == 'language':
                custom_experts.append(LanguageExpert(d_model, d_ff, n_heads, dropout, use_agglutinative_attention))
            elif exp_type == 'logic': custom_experts.append(LogicExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'spatial': custom_experts.append(SpatialExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'causal': custom_experts.append(CausalExpert(d_model, d_ff, n_heads, dropout))
            elif exp_type == 'neuro_symbolic': custom_experts.append(NeuroSymbolicExpert(d_model, d_ff, n_heads, dropout, global_knowledge_graph=global_knowledge_graph))
            else: custom_experts.append(Expert(d_model, d_ff, dropout))

        self.moe = MixtureOfExperts(
            d_model, n_experts, d_ff, k=self.k, dropout=dropout,
            custom_experts=custom_experts if custom_experts else None
        )

        if use_introspection:
            self.introspection = IntrospectionLoop(d_model, max_iterations=2, n_heads=n_heads, dropout=dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def map_experts_to_domains(self, expert_types: list) -> torch.Tensor:
        # Hangi uzmanın hangi alana ait olduğunu belirler.
        # Örn: LanguageExpert -> LINGUISTIC, NeuroSymbolicExpert -> SYMBOLIC
        mapping = []
        for expert_type in expert_types:
            if expert_type in ["language", "spatial"]: # Dilsel ve mekansal görevler
                mapping.append(EXPERT_DOMAINS["LINGUISTIC"])
            elif expert_type in ["neuro_symbolic", "logic", "causal"]: # Mantıksal görevler
                mapping.append(EXPERT_DOMAINS["SYMBOLIC"])
            else:
                mapping.append(EXPERT_DOMAINS["LINGUISTIC"]) # Varsayılan
        return torch.tensor(mapping[:self.n_experts], dtype=torch.long)  # Ensure mapping matches n_experts
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        morpho_types: Optional[torch.Tensor] = None,
        semantic_categories: Optional[torch.Tensor] = None,
        previous_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        # --- YENİ MANTIK: AKILLI YÖNLENDİRME ---
        # 1. Girdinin görev türünü tahmin et
        task_logits = self.task_classifier(x) # Shape: [batch, num_domains]

        # 2. Router için bir bias oluştur
        # Her uzman için, tahmin edilen görev türüyle ne kadar uyumlu olduğuna dair bir skor
        routing_bias = torch.zeros(x.size(0), self.n_experts, device=x.device)  # Ensure correct size

        # Uzman-alan haritasını cihaza taşı
        self.expert_to_domain_map = self.expert_to_domain_map.to(x.device)

        # Her bir alan (domain) için, o alana ait uzmanları teşvik et
        for domain_idx in range(len(EXPERT_DOMAINS)):
            # Bu alana ait uzmanların maskesi
            expert_mask = (self.expert_to_domain_map == domain_idx)
            # Bu alanın tahmin edilen skoru
            domain_score = task_logits[:, domain_idx]
            # Skoru ilgili uzmanlara bias olarak ekle
            routing_bias += expert_mask * domain_score.unsqueeze(-1)

        # 3. MoE'yi bias ile çağır
        x = self.attn_residual(x, lambda y: self.attention(y, y, y, attention_mask))
        # Pass through MoE with enriched routing context and auxiliary annotations
        x, moe_info = self.moe(
            x,
            routing_bias=routing_bias,
            attention_mask=attention_mask,
            morpho_types=morpho_types,
            semantic_categories=semantic_categories,
        )

        introspection_info = {}
        if self.use_introspection:
            x, introspection_history = self.introspection(x, previous_states)
            introspection_info = introspection_history

        # Eğitim için görev sınıflandırma logit'lerini de döndür
        moe_info['task_logits'] = task_logits

        block_info = {'type': 'enhanced_block', 'moe': moe_info, 'introspection': introspection_info}
        return x, block_info


class AGIFORMER(nn.Module):
    def __init__(
        self,
        tokenizer: MorphoPiece,  # YENİ: Tokenizer parametre olarak alınacak
        d_model: int = 768, n_layers: int = 12, n_heads: int = 12,
        d_ff: int = 3072, n_experts: int = 4, expert_types: list = None, memory_size: int = 10000,
        max_seq_len: int = 2048, dropout: float = 0.1, use_linear_attention: bool = False,
        use_memory: bool = True, use_introspection: bool = True, use_multimodal: bool = True,
        use_agglutinative_attention: bool = True,
        # --- DEĞİŞİKLİK: Yeni parametre eklendi ---
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()

        # Tokenizer'ı sakla ve vocab_size'ı ondan al
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size() if hasattr(tokenizer, 'vocab_size') and callable(getattr(tokenizer, 'vocab_size')) else tokenizer.vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing # <-- DEĞİŞİKLİK: Değeri sakla
        self.use_agglutinative_attention = use_agglutinative_attention
        # CRITICAL FIX: Ensure expert configuration is consistent
        self.n_experts = n_experts
        self.k = min(2, n_experts)  # Top-k selection should not exceed available experts

        # Token embedding layer (MorphoPiece sadece tokenization yapıyor)
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)

        if use_multimodal: self.multimodal_perception = MultimodalPerceptionCore(d_model, self.vocab_size, 2, n_heads, dropout)
        else: self.multimodal_perception = None
        
        if use_memory: self.memory = UnifiedMemoryBackbone(d_model, memory_size, max_seq_len // 2, 10)
        else: self.memory = None
        
        # CRITICAL FIX: Ensure expert_types matches n_experts
        if expert_types is None:
            expert_types = ['language', 'logic', 'spatial', 'causal'][:n_experts]
        else:
            expert_types = expert_types[:n_experts]  # Truncate to n_experts

        # Global Bilgi Grafiğini burada oluştur
        self.global_knowledge_graph = GlobalKnowledgeGraph(num_concepts=1024, d_model=d_model, num_relations=NUM_RELATIONS)

        # LanguageExpert inside each block can leverage morpho/semantic signals via MoE.
        # If you later expose per-expert configs, thread them in here.
        self.blocks = nn.ModuleList([
            AGIFORMERBlock(
                d_model,
                n_heads,
                d_ff,
                n_experts,
                expert_types,
                dropout,
                use_linear_attention,
                (use_introspection and (i == n_layers - 1)),
                use_agglutinative_attention,
                global_knowledge_graph=self.global_knowledge_graph,
            )
            for i in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, self.vocab_size)
        self.final_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        morpho_types: Optional[torch.Tensor] = None,
        semantic_categories: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        model_info = {}

        if self.multimodal_perception and (image is not None or audio is not None or video is not None):
            modality_embeds, x = self.multimodal_perception(text=input_ids, image=image, audio=audio, video=video)
            model_info['multimodal'] = True
            model_info['modalities'] = list(modality_embeds.keys())
        elif input_ids is not None:
            # Input_ids zaten tokenize edilmiş, embedding'e çevir
            # --- DEĞİŞİKLİK: Hatalı token ID'lerini güvenli aralığa sınırla ---
            # Clamp input_ids to prevent out-of-bounds embedding lookups
            input_ids_clamped = torch.clamp(input_ids, 0, self.vocab_size - 1)
            x = self.token_embedding(input_ids_clamped)  # Token embedding
            # --- BİTTİ ---
            model_info['multimodal'] = False
            model_info['tokenizer'] = 'morphopiece'
        else:
            raise ValueError("At least one input modality must be provided")
            
        if self.memory:
            x, memory_info = self.memory(x, use_working_memory=True, use_longterm_memory=True)
            model_info['memory'] = memory_info
            
        all_block_info = []
        previous_states = None
        
        # --- Gradient Checkpointing + MoE-aware experts ---
        for i, block in enumerate(self.blocks):
            if self.training and self.use_gradient_checkpointing:
                # Positional args into:
                # AGIFORMERBlock.forward(x, attention_mask, morpho_types, semantic_categories, previous_states)
                x, block_info = checkpoint(
                    block,
                    x,
                    attention_mask,
                    morpho_types,
                    semantic_categories,
                    previous_states,
                    use_reentrant=False,
                )
            else:
                x, block_info = block(
                    x,
                    attention_mask=attention_mask,
                    morpho_types=morpho_types,
                    semantic_categories=semantic_categories,
                    previous_states=previous_states,
                )

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
                logits, _ = self.forward(input_ids=generated)
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
