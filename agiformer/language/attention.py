#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
Agglutinative Attention - SOV Yapıya Göre Attention
Türkçe'nin eklemeli yapısına özel attention mekanizması
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from src.morpho_splitter import MorphoSplitter

class AgglutinativeAttention(nn.Module):
    """Türkçe eklemeli yapıya özel attention mekanizması"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        morpho_splitter: Optional[MorphoSplitter] = None,
        verb_bias: float = 2.0,
        root_bias: float = 1.5,
        suffix_bias: float = 1.2
    ):
        """
        AgglutinativeAttention başlat
        
        Args:
            hidden_size: Hidden dimension
            num_heads: Attention head sayısı
            morpho_splitter: Morfem ayrımı için splitter
            verb_bias: Yüklem token'ına verilecek ekstra bias
            root_bias: Kök token'larına verilecek bias
            suffix_bias: Ek token'larına verilecek bias
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.morpho_splitter = morpho_splitter or MorphoSplitter()
        self.verb_bias = verb_bias
        self.root_bias = root_bias
        self.suffix_bias = suffix_bias
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Morfolojik tip embeddings (23 kategori - TMA1Model ile uyumlu)
        # Not: Bu embedding şu an kullanılmıyor (sadece bias maskesi kullanılıyor)
        # Gelecekte kullanılabilir, şimdilik uyumluluk için 23 olarak bırakılıyor
        NUM_MORPHEME_TYPES = 23
        self.morpho_embedding = nn.Embedding(NUM_MORPHEME_TYPES, hidden_size)
        
    def _identify_morpho_types(
        self,
        token_ids: torch.Tensor,
        token_texts: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Token'ların morfolojik tiplerini belirle
        
        Args:
            token_ids: Token ID tensor [batch, seq_len]
            token_texts: Token string listesi (opsiyonel)
        
        Returns:
            Morfolojik tip tensor [batch, seq_len] (0=root, 1=suffix, 2=verb, 3=other, 4=pad)
        """
        batch_size, seq_len = token_ids.shape
        morpho_types = torch.zeros(batch_size, seq_len, dtype=torch.long, device=token_ids.device)
        
        # Eğer token text'leri verilmişse analiz et
        if token_texts:
            for b in range(batch_size):
                for s in range(seq_len):
                    if s < len(token_texts[b]):
                        word = token_texts[b][s]
                        analysis = self.morpho_splitter.split_word(word)
                        
                        # Yüklem kontrolü (basit: son eklerde zaman eki varsa)
                        if any('zaman' in ek.get('tür', '') for ek in analysis['morfemler']):
                            morpho_types[b, s] = 2  # Verb
                        elif analysis['kök'] == word:
                            morpho_types[b, s] = 0  # Root
                        elif analysis['ekler']:
                            morpho_types[b, s] = 1  # Suffix
                        else:
                            morpho_types[b, s] = 3  # Other
        
        return morpho_types
    
    def _create_morpho_bias_mask(
        self,
        seq_len: int,
        morpho_types: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Morfolojik tip bazlı bias maskesi oluştur (training için - tüm sequence)
        
        Args:
            seq_len: Sequence length
            morpho_types: Morfolojik tip tensor [batch, seq_len]
        
        Returns:
            Bias maskesi [batch, num_heads, seq_len, seq_len]
        """
        batch_size = morpho_types.shape[0]
        bias_mask = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=device)
        
        for b in range(batch_size):
            # Yüklem pozisyonlarını bul (SOV yapısında genelde sonda)
            verb_positions = (morpho_types[b] == 2).nonzero(as_tuple=False).squeeze(-1)
            
            # Her token için bias hesapla
            for i in range(seq_len):
                morpho_type_i = morpho_types[b, i].item()
                
                # Yüklem token'ına ekstra dikkat (SOV yapısı)
                if len(verb_positions) > 0:
                    # En yakın yüklem pozisyonunu bul
                    if len(verb_positions.shape) == 0:
                        nearest_verb = verb_positions.item()
                    else:
                        nearest_verb = verb_positions[torch.argmin(torch.abs(verb_positions - i))]
                    
                    # Yüklem token'ına bias ver
                    if isinstance(nearest_verb, torch.Tensor):
                        nearest_verb = nearest_verb.item()
                    bias_mask[b, :, i, nearest_verb] += self.verb_bias
                
                # Kök token'larına bias
                if morpho_type_i == 0:  # Root
                    bias_mask[b, :, :, i] += self.root_bias * 0.5
                
                # Ek token'larına daha az bias
                elif morpho_type_i == 1:  # Suffix
                    bias_mask[b, :, :, i] += self.suffix_bias * 0.3
        
        return bias_mask
    
    def _create_morpho_bias_mask_for_new_tokens(
        self,
        new_morpho_types: torch.Tensor,
        kv_seq_len: int,
        new_seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Yeni token'lar için morfolojik bias maskesi oluştur (generation için)
        
        Args:
            new_morpho_types: Yeni token'ların morfolojik tipleri [batch, new_seq_len]
            kv_seq_len: KV cache dahil toplam sequence uzunluğu
            new_seq_len: Yeni token sayısı (genelde 1)
            device: Device
        
        Returns:
            Bias maskesi [batch, num_heads, new_seq_len, kv_seq_len]
        """
        batch_size = new_morpho_types.shape[0]
        bias_mask = torch.zeros(batch_size, self.num_heads, new_seq_len, kv_seq_len, device=device)
        
        for b in range(batch_size):
            for i in range(new_seq_len):
                morpho_type_i = new_morpho_types[b, i].item()
                new_token_pos = kv_seq_len - new_seq_len + i
                
                # Yüklem token'ına ekstra dikkat
                if morpho_type_i == 2:  # Verb
                    # Yeni token yüklem ise, tüm önceki token'lara dikkat çek
                    bias_mask[b, :, i, :new_token_pos] += self.verb_bias * 0.5
                
                # Kök token'larına bias
                elif morpho_type_i == 0:  # Root
                    # Kök token'larına genel bias
                    bias_mask[b, :, i, :] += self.root_bias * 0.3
                
                # Ek token'larına daha az bias
                elif morpho_type_i == 1:  # Suffix
                    bias_mask[b, :, i, :] += self.suffix_bias * 0.2
        
        return bias_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        morpho_types: Optional[torch.Tensor] = None,
        token_texts: Optional[List[List[str]]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len] or [batch, seq_len, seq_len]
            morpho_types: Morfolojik tip tensor [batch, seq_len] (0=root, 1=suffix, 2=verb, 3=other, 4=pad)
                          Öncelikli: Preprocessing ile önceden hesaplanmış tensor kullanılmalı
            token_texts: Token string listesi (fallback, sadece morpho_types yoksa kullanılır)
            past_key_value: KV cache
            use_cache: Cache kullanılsın mı?
        
        Returns:
            output: [batch, seq_len, hidden_size]
            present_key_value: KV cache (if use_cache)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Q, K, V projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head: [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use past key/value if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        # kv_seq_len'i her zaman key'in uzunluğundan al
        kv_seq_len = key.shape[2]
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, kv_seq_len]
        
        # Morfolojik bias ekle
        if morpho_types is not None:
            # Preprocessing ile önceden hesaplanmış morpho_types kullan (ÖNCELİKLİ)
            # morpho_types: [batch, seq_len] - sadece yeni token'lar için
            
            # KV cache varsa, past morpho_types ile birleştirmemiz gerekir
            # Şimdilik sadece yeni token'lar için bias uygula
            if past_key_value is not None:
                # Generate sırasında: sadece yeni token'ın bias'ını ekle
                # Past KV için morpho_types bilgisi yok, sadece yeni token'a bias uygula
                morpho_bias = self._create_morpho_bias_mask_for_new_tokens(
                    morpho_types, kv_seq_len, seq_len, hidden_states.device
                )
            else:
                # Training sırasında: tüm sequence için bias
                morpho_bias = self._create_morpho_bias_mask(kv_seq_len, morpho_types, hidden_states.device)
            
            # Bias'ı ekle
            if scores.shape == morpho_bias.shape:
                scores = scores + morpho_bias
            elif morpho_bias.shape[-2] == seq_len and morpho_bias.shape[-1] == kv_seq_len:
                scores = scores + morpho_bias
        elif token_texts:
            # Fallback: token_texts ile runtime analiz (YAVAŞ - sadece gerektiğinde)
            # Token ID'leri, tüm metnin uzunluğuna göre (kv_seq_len) oluşturulmalı
            token_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            
            # Runtime morphological analysis (SLOW)
            computed_morpho_types = self._identify_morpho_types(token_ids, token_texts)
            
            # Maskeyi tam boyutta (kv_seq_len x kv_seq_len) oluştur
            if past_key_value is not None:
                # Generate sırasında: sadece yeni token için
                morpho_bias = self._create_morpho_bias_mask_for_new_tokens(
                    computed_morpho_types[:, -seq_len:], kv_seq_len, seq_len, hidden_states.device
                )
            else:
                morpho_bias = self._create_morpho_bias_mask(kv_seq_len, computed_morpho_types, hidden_states.device)
            
            # Bias'ı ekle
            if scores.shape == morpho_bias.shape:
                scores = scores + morpho_bias
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Maskeyi doğru boyuta getir
            attention_mask = attention_mask[:, :, :seq_len, :kv_seq_len]
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, value)  # [batch, heads, seq_len, head_dim]
        
        # Concatenate heads: [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        # Prepare cache
        present_key_value = None
        if use_cache:
            present_key_value = (key, value)
        
        return output, present_key_value

def main():
    """Test fonksiyonu"""
    print("\n" + "=" * 60)
    print("    Agglutinative Attention Test")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 10
    hidden_size = 768
    num_heads = 12
    
    attention = AgglutinativeAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        verb_bias=2.0,
        root_bias=1.5
    )
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    token_texts = [
        ["ev", "ler", "im", "de", "ki", "ler", "gittim", "dün"],
        ["yapay", "zeka", "gelecek", "in", "teknoloji", "sidir"]
    ]
    
    output, _ = attention(hidden_states, token_texts=token_texts)
    
    print(f"\n✅ Input shape: {hidden_states.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Agglutinative attention working!")

if __name__ == "__main__":
    main()

