# Developer: inkbytefo
# Modified: 2025-11-06

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.attention import MultiHeadAttention
from ..core.base_components import FeedForward, LayerNorm
from .knowledge_graph import DynamicKnowledgeGraph
from .relations import RELATION_TYPES, NUM_RELATIONS

class RelationClassifier(nn.Module):
    """İki kavram arasındaki ilişkiyi sınıflandırır."""
    def __init__(self, d_model: int, num_relations: int):
        super().__init__()
        self.layer1 = nn.Linear(d_model * 2, d_model)
        self.layer2 = nn.Linear(d_model, num_relations)

    def forward(self, concept1: torch.Tensor, concept2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept1 (torch.Tensor): İlk kavramın embedding'i. Shape: [num_edges, d_model]
            concept2 (torch.Tensor): İkinci kavramın embedding'i. Shape: [num_edges, d_model]

        Returns:
            torch.Tensor: İlişki türleri üzerine logit'ler. Shape: [num_edges, num_relations]
        """
        combined = torch.cat([concept1, concept2], dim=1)
        hidden = F.relu(self.layer1(combined))
        logits = self.layer2(hidden)
        return logits

class NeuroSymbolicExpert(nn.Module):
    """
    AGIFORMER için Nöro-Sembolik Uzman.
    Nöral temsilleri sembolik kavramlara çevirir, mantıksal akıl yürütür
    ve sonucu tekrar nöral uzaya yansıtır.
    """
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1, global_knowledge_graph=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # 1. Nöral Taraf: Artık Sequential değil, manuel kontrol için
        self.pre_attn_norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # 2. Köprü: Nöral'den Sembolik'e (Concept Extractor)
        # Şimdilik, en önemli token'ları seçen basit bir projeksiyon
        self.neural_to_symbolic_bridge = nn.Linear(d_model, 1)

        # 3. Sembolik Taraf: Global Bilgi Grafiği (dışarıdan alınır)
        self.knowledge_graph = global_knowledge_graph

        # Yeni: İlişki Sınıflandırıcı
        self.relation_classifier = RelationClassifier(d_model, NUM_RELATIONS)

        # 4. Köprü: Sembolik'ten Nöral'e (Logic Embedding)
        # Sonuçları d_model boyutuna getiren bir embedding katmanı
        self.symbolic_to_neural_bridge = nn.Linear(d_model, d_model) # Basit bir projeksiyon

        self.final_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def prototype_logic_engine(self, concepts: torch.Tensor, concept_indices: torch.Tensor) -> torch.Tensor:
        """
        Basit bir sembolik akıl yürütme simülasyonu.
        Örn: "Eğer 'kral' ve 'kadın' kavramları varsa, 'kraliçe' kavramını güçlendir."
        Bu sadece bir yer tutucudur (placeholder).
        """
        # Bu prototipte, sadece en önemli kavramı alıp ona bir "mantıksal" dönüşüm uygulayalım.
        # Gerçekte burada bir bilgi grafiği veya mantık motoru çalışır.

        # Örneğin, seçilen kavramların embedding'lerini alalım
        output_concepts = concepts.clone()

        # Basit bir kural: En önemli kavramın embedding'ini tersine çevir (sembolik bir işlem gibi)
        if concept_indices.numel() > 0:
            # En önemli kavramın embedding'ini al
            main_concept_embedding = concepts.view(-1, self.d_model)[concept_indices[0]]
            # Basit bir dönüşüm uygula
            inferred_embedding = torch.flip(main_concept_embedding, dims=[0])

            # Sonucu orijinal tensöre geri yaz (sadece bir örnek)
            # Bu kısım daha sofistike hale getirilecek
            output_concepts = output_concepts + inferred_embedding * 0.1

        return output_concepts

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        İlişkileri sınıflandırarak anlamsal graf oluşturan nöro-sembolik akıl yürütme döngüsü.
        """
        residual = x

        # 1. Nöral Anlama ve Attention Ağırlıklarını Çıkarma
        x_norm = self.pre_attn_norm(x)
        attn_output, attention_weights = self.attention(x_norm, x_norm, x_norm, mask, return_attention=True)
        neural_understanding = self.ffn(attn_output)

        # 2. İLİŞKİLERİ SINIFLANDIRARAK GRAF OLUŞTURMA
        batch_size, seq_len, _ = neural_understanding.shape

        # Ortalama attention matrisini al
        attn_matrix = attention_weights.mean(dim=1)
        threshold = 0.1

        # Eşik üzerindeki potansiyel kenarları bul
        adj_matrix = (attn_matrix > threshold)

        batch_edges = []
        concepts1_for_classifier = []
        concepts2_for_classifier = []

        for b in range(batch_size):
            offset = b * seq_len
            edges = adj_matrix[b].nonzero(as_tuple=False)

            for edge in edges:
                u, v = edge[0].item(), edge[1].item()
                if u != v:
                    batch_edges.append([u + offset, v + offset])
                    # Sınıflandırıcı için kavram çiftlerini topla
                    concepts1_for_classifier.append(neural_understanding[b, u])
                    concepts2_for_classifier.append(neural_understanding[b, v])

        if not batch_edges:
            return neural_understanding, {}  # Akıl yürütme adımını atla

        # Toplu halde ilişki sınıflandırması yap
        c1 = torch.stack(concepts1_for_classifier)
        c2 = torch.stack(concepts2_for_classifier)
        relation_logits = self.relation_classifier(c1, c2)
        predicted_edge_types = torch.argmax(relation_logits, dim=1)

        # Geçerli ilişkileri filtrele (NONE olmayanlar)
        valid_indices = (predicted_edge_types != RELATION_TYPES["NONE"]).nonzero(as_tuple=True)[0]

        if valid_indices.numel() == 0:
            return neural_understanding, {}  # Anlamlı ilişki bulunamadıysa atla

        final_edges = torch.tensor(batch_edges, device=x.device)[valid_indices]
        final_edge_types = predicted_edge_types[valid_indices]

        edge_index = final_edges.t().contiguous()

        # Global Bilgi Grafiğini çağır
        concepts_flat = neural_understanding.view(batch_size * seq_len, self.d_model)

        reasoned_concepts_flat = self.knowledge_graph(
            input_concepts=concepts_flat,
            edge_index=edge_index,
            edge_type=final_edge_types
        )

        reasoned_output = reasoned_concepts_flat.view(batch_size, seq_len, self.d_model)

        # Sonucu entegre et
        neural_result = self.symbolic_to_neural_bridge(reasoned_output)
        output = self.final_norm(residual + self.dropout(neural_result))

        # Sınıflandırıcıdan gelen bilgileri dışarıya aktar
        expert_info = {
            "relation_logits": relation_logits,
            "classified_edges": batch_edges  # Sınıflandırıcıya giden orijinal kenarlar
        }

        return output, expert_info
