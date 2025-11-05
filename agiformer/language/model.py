#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
TMA-1 Model - Türkçe Mantık Ağı
Morfem farkındalıklı, eklemeli yapıya özel Transformer modeli
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

from src.model import ModelConfig, FeedForward
from src.agglutinative_attention import AgglutinativeAttention
from src.grammar_engine import GrammarEngine
from src.morpho_splitter import MorphoSplitter

class TMA1TransformerLayer(nn.Module):
    """TMA-1 Transformer Layer (Morfem farkındalıklı)"""
    
    def __init__(self, config: ModelConfig, morpho_splitter: Optional[MorphoSplitter] = None):
        super().__init__()
        self.config = config
        
        # Pre-attention layer norm
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Agglutinative attention (SOV yapısına göre)
        self.attention = AgglutinativeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            morpho_splitter=morpho_splitter,
            verb_bias=2.0,
            root_bias=1.5,
            suffix_bias=1.2
        )
        
        # Pre-FFN layer norm
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Feed-forward network
        self.ffn = FeedForward(config)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        morpho_types: Optional[torch.Tensor] = None,
        token_texts: Optional[List[List[str]]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-attention layer norm + residual
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attn_output, present_key_value = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            morpho_types=morpho_types,
            token_texts=token_texts,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + attn_output
        hidden_states = self.dropout(hidden_states)
        
        # Pre-FFN layer norm + residual
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ffn_output = self.ffn(hidden_states)
        hidden_states = residual + ffn_output
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, present_key_value

class TMA1Model(nn.Module):
    """TMA-1: Türkçe Mantık Ağı - Morfem farkındalıklı Transformer"""
    
    def __init__(
        self,
        config: ModelConfig,
        morpho_splitter: Optional[MorphoSplitter] = None,
        grammar_engine: Optional[GrammarEngine] = None,
        use_grammar_bias: bool = True
    ):
        """
        TMA-1 model başlat
        
        Args:
            config: Model yapılandırması
            morpho_splitter: Morfem ayrımı için splitter
            grammar_engine: Dilbilgisi motoru
            use_grammar_bias: Dilbilgisi bias'ı kullanılsın mı?
        """
        super().__init__()
        self.config = config
        self.use_grammar_bias = use_grammar_bias
        
        self.morpho_splitter = morpho_splitter or MorphoSplitter()
        self.grammar_engine = grammar_engine or GrammarEngine() if use_grammar_bias else None
        
        # Embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Morfolojik tip embedding (detaylı - 23 kategori)
        # 0=pad, 1=special, 2-5=kökler, 6-11=iyelik, 12-16=durum, 17-20=zaman, 21=çoğul, 22=other
        NUM_MORPHEME_TYPES = 23
        self.morpho_embedding = nn.Embedding(NUM_MORPHEME_TYPES, config.hidden_size)
        
        # Semantic category embedding (12 kategori: pad, special, mekan, zaman, insan, hayvan, duygu, eşya, yiyecek, fiil_eylem, sıfat, belirsiz)
        # MORFOSEMANTIK TOKENIZATION
        NUM_SEMANTIC_CATEGORIES = 12
        self.semantic_embedding = nn.Embedding(NUM_SEMANTIC_CATEGORIES, config.hidden_size)
        
        # TMA-1 Transformer layers
        self.layers = nn.ModuleList([
            TMA1TransformerLayer(config, morpho_splitter=self.morpho_splitter)
            for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_final = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Positional encoding (learned)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_seq_len, config.hidden_size) * 0.02
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _get_token_texts(
        self,
        input_ids: torch.Tensor,
        vocab: Optional[List[str]] = None
    ) -> Optional[List[List[str]]]:
        """
        Token ID'lerden token text'lerini al (morfem analizi için)
        
        Args:
            input_ids: Token ID tensor [batch, seq_len]
            vocab: Vocabulary listesi
        
        Returns:
            Token text listesi veya None
        """
        if vocab is None:
            return None
        
        batch_size, seq_len = input_ids.shape
        token_texts = []
        
        for b in range(batch_size):
            batch_texts = []
            for s in range(seq_len):
                token_id = input_ids[b, s].item()
                if token_id < len(vocab):
                    batch_texts.append(vocab[token_id])
                else:
                    batch_texts.append("")
            token_texts.append(batch_texts)
        
        return token_texts
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        morpho_types: Optional[torch.Tensor] = None,
        semantic_categories: Optional[torch.Tensor] = None,
        token_texts: Optional[List[List[str]]] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        vocab: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            morpho_types: Morfolojik tip tensor [batch, seq_len] (öncelikli)
            semantic_categories: Anlamsal kategori tensor [batch, seq_len] (MORFOSEMANTIK TOKENIZATION)
            token_texts: Token string listesi (fallback, sadece morpho_types yoksa)
            past_key_values: KV cache
            use_cache: Cache kullanılsın mı?
            vocab: Vocabulary (token_texts için, fallback)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            past_key_values: KV cache (if use_cache)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding lookup
        hidden_states = self.embedding(input_ids)  # [batch, seq_len, hidden_size]
        
        # Add morpheme type embeddings (EXPLICIT MORPHEME REPRESENTATIONS)
        if morpho_types is not None:
            # Clamp morpho_types to valid range [0, 22]
            morpho_types_clamped = torch.clamp(morpho_types, 0, 22)
            morpho_emb = self.morpho_embedding(morpho_types_clamped)  # [batch, seq_len, hidden_size]
            # Additive combination (embedding + morpheme type embedding)
            hidden_states = hidden_states + morpho_emb
        else:
            # Fallback: if morpho_types not provided, use pad (0) for all tokens
            pad_morpho_types = torch.zeros_like(input_ids, dtype=torch.long)
            morpho_emb = self.morpho_embedding(pad_morpho_types)
            hidden_states = hidden_states + morpho_emb
        
        # Add semantic category embeddings (MORFOSEMANTIK TOKENIZATION)
        if semantic_categories is not None:
            # Clamp semantic_categories to valid range [0, 11]
            semantic_categories_clamped = torch.clamp(semantic_categories, 0, 11)
            semantic_emb = self.semantic_embedding(semantic_categories_clamped)  # [batch, seq_len, hidden_size]
            # Additive combination (embedding + morpho embedding + semantic embedding)
            hidden_states = hidden_states + semantic_emb
        else:
            # Fallback: if semantic_categories not provided, use belirsiz (11) for all tokens
            pad_semantic_categories = torch.full_like(input_ids, 11, dtype=torch.long)  # belirsiz
            semantic_emb = self.semantic_embedding(pad_semantic_categories)
            hidden_states = hidden_states + semantic_emb
        
        # Add positional encoding
        hidden_states = hidden_states + self.pos_encoding[:, :seq_len, :]
        
        # Get token texts if not provided and needed (fallback only)
        if token_texts is None and vocab is not None and morpho_types is None:
            token_texts = self._get_token_texts(input_ids, vocab)
        
        # Process through TMA-1 layers
        present_key_values = [] if use_cache else None
        
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values else None
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                morpho_types=morpho_types,
                token_texts=token_texts,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.ln_final(hidden_states)
        
        # Output projection
        logits = self.output_proj(hidden_states)  # [batch, seq_len, vocab_size]
        
        # Apply grammar bias if enabled
        if self.use_grammar_bias and self.grammar_engine and vocab:
            # Basit implementasyon: sadece son token için
            previous_tokens_list = []
            for b in range(batch_size):
                if token_texts:
                    previous_tokens_list.append(token_texts[b][:-1] if len(token_texts[b]) > 1 else [])
                else:
                    previous_tokens_list.append([])
            
            # Grammar bias uygula (basit: son pozisyon için)
            if seq_len > 0:
                logits = self.grammar_engine.apply_grammar_bias(
                    logits,
                    vocab,
                    previous_tokens_list[0] if previous_tokens_list else [],
                    None  # morpho_analysis
                )
        
        return logits, present_key_values if use_cache else None
    
    def get_num_params(self) -> int:
        """Toplam parametre sayısı"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        vocab: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        Metin üretimi (morfem farkındalıklı)
        
        Args:
            input_ids: [batch, seq_len] başlangıç token'ları
            vocab: Vocabulary listesi
            max_new_tokens: Maksimum üretilecek token
            temperature: Sampling sıcaklığı
            top_k: Top-k sampling
            top_p: Nucleus sampling
            eos_token_id: EOS token ID
            pad_token_id: Padding token ID
        
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        past_key_values = None
        generated = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                logits, past_key_values = self(
                    generated[:, -self.config.max_seq_len:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    vocab=vocab
                )
                
                # Son token için logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) sampling
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
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # EOS check
                if (next_token == eos_token_id).all():
                    break
        
        return generated

def main():
    """Test fonksiyonu"""
    print("\n" + "=" * 60)
    print("    TMA-1 Model Test")
    print("=" * 60)
    
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        ffn_size=3072,
        max_seq_len=512
    )
    
    model = TMA1Model(config)
    
    print(f"\n✅ TMA-1 Model created")
    print(f"   Parameters: {model.get_num_params():,} ({model.get_num_params() / 1e6:.2f}M)")
    print(f"   Layers: {config.num_layers}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Heads: {config.num_heads}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, _ = model(input_ids)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output shape: {logits.shape}")
    print(f"   ✅ TMA-1 model working!")

if __name__ == "__main__":
    main()

