# Developer: inkbytefo
# Modified: 2025-11-06

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

class GlobalKnowledgeGraph(nn.Module):
    """
    Eğitim boyunca öğrenilen kavramları ve ilişkileri depolayan,
    tüm model tarafından paylaşılan kalıcı bir bilgi grafiği.
    """
    def __init__(self, num_concepts: int, d_model: int, num_relations: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.d_model = d_model
        self.num_relations = num_relations

        # 1. Global Kavram Hafızası (Öğrenilebilir Düğümler)
        # Bu, modelin "dünya bilgisi"nin depolandığı yer.
        self.global_concepts = nn.Embedding(num_concepts, d_model)

        # 2. GNN Tabanlı Akıl Yürütme Motoru - RGCN ile
        self.reasoning_conv1 = RGCNConv(d_model, d_model, num_relations)
        self.reasoning_conv2 = RGCNConv(d_model, d_model, num_relations)

        # 3. Yazma/Güncelleme Mekanizması
        # Girdiden gelen yeni bilgiyi global grafa nasıl entegre edeceğimizi öğrenir.
        self.write_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.write_transform = nn.Linear(d_model, d_model)

    def forward(self, input_concepts: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """
        Graf üzerinde ilişkisel akıl yürütür.

        Args:
            input_concepts (torch.Tensor): Düğüm (kavram) embedding'leri. Shape: [num_concepts, d_model]
            edge_index (torch.Tensor): Kenar bağlantıları. Shape: [2, num_edges]
            edge_type (torch.Tensor): Her kenarın ilişki türü. Shape: [num_edges]

        Returns:
            torch.Tensor: Akıl yürütme sonrası güncellenmiş kavram embedding'leri.
        """
        # --- Okuma (Read) Aşaması ---
        # Girdi kavramlarının global hafızadaki en benzer kavramları bulmasını sağla.
        # Cosine similarity ile en ilgili global kavramları çek.
        global_concepts_norm = F.normalize(self.global_concepts.weight, dim=1)
        input_concepts_norm = F.normalize(input_concepts, dim=1)

        similarity = torch.matmul(input_concepts_norm, global_concepts_norm.t()) # [num_concepts, num_global_concepts]

        # En ilgili K global kavramı al (Attention-like retrieval)
        top_k = min(10, self.num_concepts)
        retrieval_scores, retrieved_indices = torch.topk(similarity, k=top_k, dim=-1)
        retrieval_weights = F.softmax(retrieval_scores, dim=-1)

        retrieved_concepts = self.global_concepts(retrieved_indices) # [num_concepts, top_k, d_model]

        # Ağırlıklı ortalama ile "çağrılan" bilgiyi oluştur
        context_from_memory = torch.sum(retrieval_weights.unsqueeze(-1) * retrieved_concepts, dim=1) # [num_concepts, d_model]

        # Girdi kavramlarını, hafızadan gelen bağlamla zenginleştir
        reasoning_input = input_concepts + context_from_memory

        # --- Akıl Yürütme (Reasoning) Aşaması ---
        # RGCN ile ilişkisel mesajlaşma
        x = self.reasoning_conv1(reasoning_input, edge_index, edge_type)
        x = F.relu(x)
        x = self.reasoning_conv2(x, edge_index, edge_type)

        reasoned_output = reasoning_input + x # Residual connection

        # --- Yazma (Write) Aşaması (Eğer model eğitim modundaysa) ---
        if self.training:
            # Girdiden gelen en önemli kavramları global hafızaya yaz/güncelle.
            # Basit bir strateji: En çok bağlantılı düğümü güncelle.
            if edge_index.numel() > 0:
                # En çok kenara sahip düğümü bul
                unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
                most_connected_idx = unique_nodes[counts.argmax()].item()

                concept_to_write = input_concepts[most_connected_idx]

                # En benzer global kavramı bul
                target_global_idx = similarity[most_connected_idx].argmax()

                # Kapılı güncelleme (gated update) ile yavaşça öğren
                gate = self.write_gate(torch.cat([self.global_concepts.weight[target_global_idx], concept_to_write], dim=0))
                updated_value = self.global_concepts.weight[target_global_idx] * (1 - gate) + self.write_transform(concept_to_write) * gate

                # In-place güncelleme (eğitim için gerekli)
                self.global_concepts.weight.data[target_global_idx] = updated_value

        return reasoned_output


class DynamicKnowledgeGraph(nn.Module):
    """
    GNN tabanlı dinamik bir bilgi grafiği.
    Kavramları (düğümler) ve ilişkileri (kenarlar) öğrenir ve günceller.
    (Eski versiyon, geriye uyumluluk için tutuluyor)
    """
    def __init__(self, d_model: int, num_relations: int = 10):
        super().__init__()
        self.d_model = d_model
        self.num_relations = num_relations

        # GNN katmanları: Graf üzerinde mesajlaşma ve öğrenme için
        self.conv1 = GCNConv(d_model, d_model)
        self.conv2 = GCNConv(d_model, d_model)

        # İlişki türlerini temsil eden öğrenilebilir embedding'ler
        self.relation_embeddings = nn.Embedding(num_relations, d_model)

    def forward(self, concepts: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """
        Graf üzerinde akıl yürütür.

        Args:
            concepts (torch.Tensor): Düğüm (kavram) embedding'leri. Shape: [num_concepts, d_model]
            relations (torch.Tensor): Kenar (ilişki) bilgisi. Shape: [2, num_edges]

        Returns:
            torch.Tensor: Akıl yürütme sonrası güncellenmiş kavram embedding'leri.
        """
        # PyG için veri yapısı oluştur
        # Şimdilik kenar özelliklerini (edge_attr) basit tutuyoruz
        graph_data = Data(x=concepts, edge_index=relations)

        # GNN ile 2 katmanlı mesajlaşma
        x = self.conv1(graph_data.x, graph_data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, graph_data.edge_index)

        # Orijinal kavramlarla birleştir (residual connection)
        return concepts + x
