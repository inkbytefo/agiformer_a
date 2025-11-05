#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Grammar Engine - Türkçe Dilbilgisi Kuralları
Ünlü uyumu, ek sırası, yasak kombinasyonlar için bias sistemi
============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import re

class GrammarEngine:
    """Türkçe dilbilgisi kuralları motoru"""
    
    def __init__(self, penalty: float = -100.0, reward: float = 5.0):
        """
        GrammarEngine başlat
        
        Args:
            penalty: Yasak kombinasyonlar için ceza
            reward: Doğru kombinasyonlar için ödül
        """
        self.penalty = penalty
        self.reward = reward
        
        # Ünlü uyumu kuralları
        self.back_vowels = {'a', 'ı', 'o', 'u'}
        self.front_vowels = {'e', 'i', 'ö', 'ü'}
        
        # Yasak ek kombinasyonları
        self.forbidden_combinations = [
            ('a', 'e'),  # Kalın-ince uyumsuzluğu
            ('ı', 'i'),
            ('o', 'ö'),
            ('u', 'ü'),
            ('de', 'da'),  # Ünlü uyumu ile uyumsuz
            ('te', 'ta'),
            ('den', 'dan'),
            ('ten', 'tan'),
        ]
        
        # Doğru ek sıralaması (öncelik sırası)
        self.suffix_order = {
            'çoğul': 1,
            'iyelik': 2,
            'bulunma': 3,
            'ayrılma': 4,
            'yönelme': 5,
            'belirtme': 6,
            'geçmiş_zaman': 7,
            'şimdiki_zaman': 8,
            'gelecek_zaman': 9,
        }
        
        # Vocabulary cache for vectorized operations
        self._vocab_cache: Optional[Dict] = None
        self._vocab_first_vowels: Optional[torch.Tensor] = None
        self._vocab_contains_forbidden_a: Optional[torch.Tensor] = None  # For each forbidden pair
        self._vocab_contains_forbidden_b: Optional[torch.Tensor] = None
    
    def check_vowel_harmony(self, root: str, suffix: str) -> bool:
        """
        Ünlü uyumu kontrolü
        
        Args:
            root: Kök kelime
            suffix: Ek
        
        Returns:
            Ünlü uyumu doğru mu?
        """
        if not root or not suffix:
            return True
        
        # Kökün son ünlüsünü bul
        root_vowels = [c for c in root if c in 'aeıiouöü']
        if not root_vowels:
            return True
        
        last_vowel = root_vowels[-1].lower()
        
        # Ekin ünlülerini bul
        suffix_vowels = [c for c in suffix.lower() if c in 'aeıiouöü']
        if not suffix_vowels:
            return True
        
        first_vowel = suffix_vowels[0].lower()
        
        # Kalın-ince uyumu
        if last_vowel in self.back_vowels:
            # Kalın ünlüden sonra kalın veya belirli ünlüler gelmeli
            if first_vowel in self.front_vowels and first_vowel not in {'i'}:
                return False
        elif last_vowel in self.front_vowels:
            # İnce ünlüden sonra ince ünlüler gelmeli
            if first_vowel in self.back_vowels and first_vowel not in {'a'}:
                return False
        
        # Düz-yuvarlak uyumu (basit)
        if last_vowel in {'o', 'u'} and first_vowel in {'ö', 'ü', 'e', 'i'}:
            return False
        elif last_vowel in {'ö', 'ü'} and first_vowel in {'a', 'ı', 'o', 'u'}:
            return False
        
        return True
    
    def check_suffix_order(self, suffixes: List[str]) -> bool:
        """
        Ek sırası kontrolü
        
        Args:
            suffixes: Ek listesi
        
        Returns:
            Ek sırası doğru mu?
        """
        # Basit kontrol: yüklem ekleri genelde sonda olmalı
        time_suffixes = ['geçmiş_zaman', 'şimdiki_zaman', 'gelecek_zaman']
        
        time_indices = []
        other_indices = []
        
        for i, suffix in enumerate(suffixes):
            if any(ts in suffix for ts in time_suffixes):
                time_indices.append(i)
            else:
                other_indices.append(i)
        
        # Zaman ekleri diğer eklerden sonra olmalı
        if time_indices and other_indices:
            if min(time_indices) < max(other_indices):
                return False
        
        return True
    
    def _build_vocab_cache(self, vocab: List[str], device: torch.device) -> None:
        """
        Vocabulary için ünlü bilgisini cache'le (vektörize işlemler için)
        
        Args:
            vocab: Vocabulary listesi
            device: Tensor device
        """
        vocab_size = len(vocab)
        
        # Her token'ın ilk ünlüsünü bul (0=back, 1=front, 2=no vowel, 3=invalid)
        first_vowels = torch.zeros(vocab_size, dtype=torch.long, device=device)
        
        # Yasak kombinasyon bilgisi (her forbidden pair için)
        # Her pattern için ayrı ayrı kontrol edelim
        forbidden_mask_a = torch.zeros(
            vocab_size, len(self.forbidden_combinations), dtype=torch.bool, device=device
        )
        forbidden_mask_b = torch.zeros(
            vocab_size, len(self.forbidden_combinations), dtype=torch.bool, device=device
        )
        
        for v_idx, token in enumerate(vocab):
            if v_idx >= vocab_size:
                break
            
            token_lower = token.lower()
            
            # İlk ünlüyü bul
            token_vowels = [c for c in token_lower if c in 'aeıiouöü']
            if token_vowels:
                first_vowel = token_vowels[0]
                if first_vowel in self.back_vowels:
                    first_vowels[v_idx] = 0  # back vowel
                elif first_vowel in self.front_vowels:
                    first_vowels[v_idx] = 1  # front vowel
                else:
                    first_vowels[v_idx] = 2  # no vowel (shouldn't happen)
            else:
                first_vowels[v_idx] = 2  # no vowel
            
            # Yasak kombinasyonları kontrol et (a ve b'yi ayrı ayrı)
            for f_idx, (forbidden_a, forbidden_b) in enumerate(self.forbidden_combinations):
                if forbidden_a in token_lower:
                    forbidden_mask_a[v_idx, f_idx] = True
                if forbidden_b in token_lower:
                    forbidden_mask_b[v_idx, f_idx] = True
        
        self._vocab_cache = {
            'vocab': vocab,
            'vocab_size': vocab_size
        }
        self._vocab_first_vowels = first_vowels
        self._vocab_contains_forbidden_a = forbidden_mask_a
        self._vocab_contains_forbidden_b = forbidden_mask_b
    
    def _extract_last_vowel_vectorized(
        self,
        tokens: List[str],
        device: torch.device
    ) -> torch.Tensor:
        """
        Token listesinden son ünlüleri vektörize şekilde çıkar
        
        Args:
            tokens: Token string listesi
            device: Tensor device
        
        Returns:
            Son ünlü tensor [len(tokens)] (0=back, 1=front, 2=no vowel)
        """
        last_vowels = torch.zeros(len(tokens), dtype=torch.long, device=device)
        
        for i, token in enumerate(tokens):
            if not token:
                last_vowels[i] = 2  # no vowel
                continue
            
            token_lower = token.lower()
            token_vowels = [c for c in token_lower if c in 'aeıiouöü']
            
            if token_vowels:
                last_vowel = token_vowels[-1]
                if last_vowel in self.back_vowels:
                    last_vowels[i] = 0  # back vowel
                elif last_vowel in self.front_vowels:
                    last_vowels[i] = 1  # front vowel
                else:
                    last_vowels[i] = 2  # no vowel
            else:
                last_vowels[i] = 2  # no vowel
        
        return last_vowels
    
    def _check_vowel_harmony_vectorized(
        self,
        last_vowels: torch.Tensor,  # [batch, seq_len] veya [batch*seq_len]
        vocab_first_vowels: torch.Tensor  # [vocab_size]
    ) -> torch.Tensor:
        """
        Ünlü uyumunu vektörize şekilde kontrol et
        
        Args:
            last_vowels: Son ünlü tensor [batch*seq_len] (0=back, 1=front, 2=no vowel)
            vocab_first_vowels: Vocab ilk ünlü tensor [vocab_size]
        
        Returns:
            Uyum maskesi [batch*seq_len, vocab_size] (1=uyumlu, 0=uyumsuz)
        """
        batch_seq_len = last_vowels.shape[0]
        vocab_size = vocab_first_vowels.shape[0]
        
        # Expand dimensions for broadcasting
        # last_vowels: [batch_seq_len, 1]
        # vocab_first_vowels: [1, vocab_size]
        last_vowels_expanded = last_vowels.unsqueeze(1)  # [batch_seq_len, 1]
        vocab_first_vowels_expanded = vocab_first_vowels.unsqueeze(0)  # [1, vocab_size]
        
        # Initialize harmony mask (all 1 = compatible)
        harmony_mask = torch.ones(
            batch_seq_len, vocab_size, dtype=torch.bool, device=last_vowels.device
        )
        
        # Kalın-ince uyumu kontrolü
        # Back vowel (0) + front vowel (1) = incompatible (except 'i' and 'a' cases handled separately)
        back_vowel_mask = (last_vowels_expanded == 0)  # [batch_seq_len, 1]
        front_vowel_vocab = (vocab_first_vowels_expanded == 1)  # [1, vocab_size]
        
        # Front vowel (1) + back vowel (0) = incompatible (except 'a' case)
        front_vowel_mask = (last_vowels_expanded == 1)  # [batch_seq_len, 1]
        back_vowel_vocab = (vocab_first_vowels_expanded == 0)  # [1, vocab_size]
        
        # Simple rule: back+front or front+back = incompatible
        # (Simplified - detailed rules with 'i' and 'a' exceptions would need more complex logic)
        incompatible = (back_vowel_mask & front_vowel_vocab) | (front_vowel_mask & back_vowel_vocab)
        harmony_mask[incompatible] = False
        
        # Düz-yuvarlak uyumu (simplified - would need actual vowel characters for full accuracy)
        # For now, we rely on the back/front classification
        
        # No vowel cases: set to compatible (1)
        no_vowel_mask = (last_vowels_expanded == 2) | (vocab_first_vowels_expanded == 2)
        harmony_mask[no_vowel_mask.expand(-1, vocab_size)] = True
        
        return harmony_mask
    
    def apply_grammar_bias(
        self,
        logits: torch.Tensor,
        vocab: List[str],
        previous_tokens: List[str],
        morpho_analysis: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        """
        Dilbilgisi kurallarına göre logit bias'ı uygula (VEKTÖRİZE)
        
        Args:
            logits: Logit tensor [batch, seq_len, vocab_size]
            vocab: Vocabulary listesi
            previous_tokens: Önceki token'lar (batch, seq_len için list of lists veya flat list)
            morpho_analysis: Morfem analizi (opsiyonel)
        
        Returns:
            Bias uygulanmış logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Build vocab cache if needed or vocab changed
        if (self._vocab_cache is None or 
            self._vocab_cache['vocab'] != vocab or 
            self._vocab_cache['vocab_size'] != vocab_size):
            self._build_vocab_cache(vocab, device)
        
        # Flatten batch and sequence dimensions for vectorized operations
        logits_flat = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
        bias_flat = torch.zeros_like(logits_flat)
        
        # Extract last vowels from previous tokens (vectorized)
        # previous_tokens can be List[List[str]] (batch-wise) or flat List[str]
        if previous_tokens and len(previous_tokens) > 0:
            # Check if it's nested (batch-wise) or flat
            if isinstance(previous_tokens[0], list):
                # Batch-wise: flatten
                flat_tokens = [token for batch in previous_tokens for token in batch]
            else:
                # Flat list
                flat_tokens = previous_tokens
            
            # Ensure we have enough tokens for all positions
            # For each position (batch*seq_len), we need the previous token
            num_positions = batch_size * seq_len
            last_tokens_list = []
            
            for pos in range(num_positions):
                if pos < len(flat_tokens) and flat_tokens[pos]:
                    last_tokens_list.append(flat_tokens[pos])
                else:
                    last_tokens_list.append("")
            
            # Extract last vowels vectorized
            last_vowels = self._extract_last_vowel_vectorized(last_tokens_list, device)
            
            # Check vowel harmony vectorized
            harmony_mask = self._check_vowel_harmony_vectorized(
                last_vowels, self._vocab_first_vowels
            )  # [batch*seq_len, vocab_size]
            
            # Apply vowel harmony bias: penalty for incompatible, reward for compatible
            vowel_harmony_bias = torch.where(
                harmony_mask,
                torch.full_like(bias_flat, self.reward * 0.1),
                torch.full_like(bias_flat, self.penalty)
            )
            bias_flat += vowel_harmony_bias
            
            # Yasak kombinasyon kontrolü (vectorized)
            # For each forbidden combination, check if it appears in last_token + vocab_token
            last_tokens_lower = [t.lower() for t in last_tokens_list]
            forbidden_bias = torch.zeros_like(bias_flat)
            
            for f_idx, (forbidden_a, forbidden_b) in enumerate(self.forbidden_combinations):
                # Get cached vocab patterns
                vocab_contains_a = self._vocab_contains_forbidden_a[:, f_idx].unsqueeze(0)  # [1, vocab_size]
                vocab_contains_b = self._vocab_contains_forbidden_b[:, f_idx].unsqueeze(0)  # [1, vocab_size]
                
                # Check which last tokens contain forbidden parts (vectorized)
                last_contains_a = torch.tensor(
                    [forbidden_a in t for t in last_tokens_lower],
                    dtype=torch.bool, device=device
                ).unsqueeze(1)  # [batch*seq_len, 1]
                last_contains_b = torch.tensor(
                    [forbidden_b in t for t in last_tokens_lower],
                    dtype=torch.bool, device=device
                ).unsqueeze(1)  # [batch*seq_len, 1]
                
                # Check: (last_contains_a & vocab_contains_b) | (last_contains_b & vocab_contains_a)
                # Broadcasting: [batch*seq_len, 1] & [1, vocab_size] = [batch*seq_len, vocab_size]
                forbidden_match = (
                    (last_contains_a & vocab_contains_b) | 
                    (last_contains_b & vocab_contains_a)
                )  # [batch*seq_len, vocab_size]
                
                forbidden_bias[forbidden_match] += self.penalty * 0.5
            
            bias_flat += forbidden_bias
        
        # Reshape back to [batch, seq_len, vocab_size]
        bias = bias_flat.view(batch_size, seq_len, vocab_size)
        biased_logits = logits + bias
        
        return biased_logits
    
    def get_vowel_harmony_mask(
        self,
        vocab: List[str],
        last_vowel: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Ünlü uyumu maskesi oluştur
        
        Args:
            vocab: Vocabulary listesi
            last_vowel: Son ünlü
            device: Tensor device
        
        Returns:
            Mask tensor [vocab_size] (1=uyumlu, 0=uyumsuz)
        """
        mask = torch.ones(len(vocab), device=device)
        
        if not last_vowel or last_vowel not in 'aeıiouöü':
            return mask
        
        last_vowel_lower = last_vowel.lower()
        
        for i, token in enumerate(vocab):
            token_vowels = [c for c in token.lower() if c in 'aeıiouöü']
            if not token_vowels:
                continue
            
            first_vowel = token_vowels[0].lower()
            
            # Ünlü uyumu kontrolü
            if not self.check_vowel_harmony(last_vowel, first_vowel):
                mask[i] = 0.0
        
        return mask
    
    def validate_sequence(self, tokens: List[str]) -> Tuple[bool, List[str]]:
        """
        Token dizisinin dilbilgisi kurallarına uygunluğunu kontrol et
        
        Args:
            tokens: Token listesi
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        for i in range(1, len(tokens)):
            prev_token = tokens[i - 1]
            curr_token = tokens[i]
            
            # Ünlü uyumu kontrolü
            if not self.check_vowel_harmony(prev_token, curr_token):
                errors.append(f"Ünlü uyumu hatası: '{prev_token}' + '{curr_token}'")
            
            # Yasak kombinasyon kontrolü
            for forbidden in self.forbidden_combinations:
                if (forbidden[0] in prev_token.lower() and forbidden[1] in curr_token.lower()) or \
                   (forbidden[1] in prev_token.lower() and forbidden[0] in curr_token.lower()):
                    errors.append(f"Yasak kombinasyon: '{prev_token}' + '{curr_token}'")
        
        return len(errors) == 0, errors

def main():
    """Test fonksiyonu"""
    print("\n" + "=" * 60)
    print("    Grammar Engine Test")
    print("=" * 60)
    
    engine = GrammarEngine()
    
    # Ünlü uyumu testleri
    test_cases = [
        ("ev", "de", True),   # Doğru
        ("ev", "da", True),   # Doğru (alternatif)
        ("ev", "ta", False),  # Yanlış (sert ünsüz uyumsuzluğu)
        ("kitap", "da", True), # Doğru
        ("kitap", "de", False), # Yanlış
        ("güzel", "de", True),  # Doğru
        ("güzel", "da", False), # Yanlış
    ]
    
    print("\n📝 Ünlü Uyumu Testleri:")
    for root, suffix, expected in test_cases:
        result = engine.check_vowel_harmony(root, suffix)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {root} + {suffix} → {result} (beklenen: {expected})")
    
    # Token dizisi validasyonu
    print("\n📝 Dizi Validasyonu:")
    sequences = [
        ["ev", "de", "ki", "ler"],
        ["ev", "da", "ki", "ler"],  # Ünlü uyumu hatası
        ["kitap", "dan", "aldım"],
        ["güzel", "de", "bir", "kitap"],
    ]
    
    for seq in sequences:
        is_valid, errors = engine.validate_sequence(seq)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {' '.join(seq)}")
        if errors:
            for error in errors:
                print(f"      - {error}")

if __name__ == "__main__":
    main()

