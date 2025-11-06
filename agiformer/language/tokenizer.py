#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
MorphoPiece Tokenizer - Türkçe Morfoloji Farkındalıklı Tokenizer
SentencePiece + Morfem ayrımı kombinasyonu
============================================================================
"""

import sentencepiece as spm
from sentencepiece import SentencePieceProcessor
import os
from typing import List, Dict, Optional
import json
import re

class MorphoPiece:
    """Türkçe morfoloji farkındalıklı tokenizer"""
    
    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        morpho_splitter: Optional[MorphoSplitter] = None
    ):
        self.sp_processor = None
        self.morpho_splitter = morpho_splitter or MorphoSplitter()
        self.vocab_size = 0
        self.morpho_aware_separator = "##"
        
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.load(tokenizer_path)
    
    def load(self, tokenizer_path: str) -> bool:
        """SentencePiece tokenizer'ı yükle"""
        if not os.path.exists(tokenizer_path):
            print(f"❌ Tokenizer not found: {tokenizer_path}")
            return False
        
        try:
            self.sp_processor = SentencePieceProcessor(model_file=tokenizer_path)
            self.vocab_size = self.sp_processor.vocab_size()
            print(f"✅ MorphoPiece loaded (vocab size: {self.vocab_size})")
            return True
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            return False
    
    def train(
        self,
        corpus_file: str,
        output_prefix: str = "tokenizer/morphopiece",
        vocab_size: int = 32000,
        morpho_aware: bool = True
    ) -> bool:
        """MorphoPiece tokenizer'ı eğit"""
        if not os.path.exists(corpus_file):
            print(f"❌ Corpus file not found: {corpus_file}")
            return False
        
        print(f"🔤 Training MorphoPiece tokenizer...")
        print(f"   Corpus: {corpus_file}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Morpho-aware: {morpho_aware}")
        
        if morpho_aware:
            print("\n📝 Processing corpus with morphological analysis...")
            processed_corpus = self._preprocess_with_morpho(corpus_file)
            temp_file = output_prefix + "_processed.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for line in processed_corpus:
                    f.write(line + '\n')
            training_file = temp_file
        else:
            training_file = corpus_file
        
        try:
            train_command = (
                f"--input={training_file} "
                f"--model_prefix={output_prefix} "
                f"--vocab_size={vocab_size} "
                f"--model_type=unigram "
                f"--user_defined_symbols={self.morpho_aware_separator} "
                f"--character_coverage=0.9995 "
                f"--input_sentence_size=1000000 "
                f"--shuffle_input_sentence=True "
                f"--seed_sentencepiece_size=1000000 "
                f"--shrinking_factor=0.75 "
                f"--num_threads=4 "
                f"--max_sentence_length=4192 "
                f"--normalization_rule_name=nfkc_cf "
                f"--byte_fallback=True"
            )
            print(f"\n🚀 Running SentencePiece training...")
            spm.SentencePieceTrainer.train(train_command)
            
            if morpho_aware and os.path.exists(training_file):
                os.remove(training_file)
            
            model_file = f"{output_prefix}.model"
            if os.path.exists(model_file):
                self.load(model_file)
                print(f"✅ MorphoPiece training completed!")
                return True
            else:
                print(f"❌ Model file not created")
                return False
        
        except Exception as e:
            import traceback
            print(f"❌ Training error: {e}")
            print(traceback.format_exc())
            return False
    
    def _preprocess_with_morpho(self, corpus_file: str) -> List[str]:
        """Corpus'u morfem ayrımı ile işle"""
        processed_lines = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   Corpus has {len(lines)} lines.")
            for line_num, line in enumerate(lines):
                if line_num % 10000 == 0 and line_num > 0:
                    print(f"   Processed {line_num:,} lines...", end='\r')
                
                line = line.strip()
                if not line:
                    continue
                
                analysis = self.morpho_splitter.split_sentence(line)
                processed_words = []
                for word_analysis in analysis["kelimeler"]:
                    morphemes = [word_analysis["kök"]] + [ek.lstrip('-') for ek in word_analysis["ekler"]]
                    processed_words.append(self.morpho_aware_separator.join(morphemes))
                
                processed_lines.append(' '.join(processed_words))
        
        print(f"\n✅ Processed {len(processed_lines):,} lines")
        return processed_lines
    
    def encode(
        self,
        text: str,
        morpho_aware: bool = True,
        out_type: type = int
    ) -> List:
        """Metni tokenize et"""
        if not self.sp_processor:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
        
        if morpho_aware:
            analysis = self.morpho_splitter.split_sentence(text)
            processed_words = []
            for word_analysis in analysis["kelimeler"]:
                morphemes = [word_analysis["kök"]] + [ek.lstrip('-') for ek in word_analysis["ekler"]]
                processed_words.append(self.morpho_aware_separator.join(morphemes))
            processed_text = ' '.join(processed_words)
        else:
            processed_text = text
        
        if out_type == int:
            return self.sp_processor.encode(processed_text, out_type=int)
        else:
            return self.sp_processor.encode(processed_text, out_type=str)
    
    def decode(self, token_ids: List[int]) -> str:
        """Token ID'lerini metne çevir"""
        if not self.sp_processor:
            raise ValueError("Tokenizer not loaded.")
        
        decoded_text = self.sp_processor.decode(token_ids)
        # Morpheme separator'ı kaldırarak kelimeleri birleştir
        return decoded_text.replace(self.morpho_aware_separator, "").replace(" ", " ").strip()

    def get_morpho_tokens(self, text: str) -> List[Dict]:
        """Metni morfem token'larına ayır (debug/analiz için)"""
        analysis = self.morpho_splitter.split_sentence(text)
        tokens = []
        
        for word_analysis in analysis["kelimeler"]:
            tokens.append({
                "token": word_analysis["kök"],
                "type": "root",
                "word": word_analysis["kelime"]
            })
            for ek in word_analysis["ekler"]:
                tokens.append({
                    "token": ek.lstrip('-'),
                    "type": "suffix",
                    "word": word_analysis["kelime"]
                })
        
        return tokens

def main():
    """Test fonksiyonu"""
    print("\n" + "=" * 60)
    print("    MorphoPiece Tokenizer Test")
    print("=" * 60)
    
    morphopiece = MorphoPiece()
    
    test_texts = [
        "Evlerimdekiler",
        "Dün markete gittim",
        "Türkçe eklemeli bir dildir",
        "Yapay zeka geleceğin teknolojisidir"
    ]
    
    print("\n📝 Morfem Analizi:")
    for text in test_texts:
        print(f"\n   Metin: {text}")
        tokens = morphopiece.get_morpho_tokens(text)
        print(f"   Morfemler: {' | '.join([t['token'] for t in tokens])}")
        print(f"   Kökler: {' | '.join([t['token'] for t in tokens if t['type'] == 'root'])}")
        print(f"   Ekler: {' | '.join([t['token'] for t in tokens if t['type'] == 'suffix'])}")

if __name__ == "__main__":
    main()
