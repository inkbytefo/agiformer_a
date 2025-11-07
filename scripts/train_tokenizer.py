#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Developer: inkbytefo
# Modified: 2025-11-05
"""
============================================================================
MorphoPiece Tokenizer Training Script
1.5 GB Türkçe corpus (C4 + Wikipedia) ile morfem farkındalıklı tokenizer eğitimi
============================================================================
"""

import sys
from pathlib import Path

# Projenin kök dizinini (agiformer_b) Python path'ine ekle
# Bu betik scripts/ dizininde olduğu için, iki üst dizine çıkmamız gerekiyor.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import argparse
from tqdm import tqdm
import sentencepiece as spm

# Morphological analysis
# Path correction above should fix this import
from agiformer.language.morpho_splitter import RegexSplitter

# Data collection imports
try:
    from datasets import load_dataset
except ImportError:
    print("❌ datasets library not found. Install: pip install datasets")
    sys.exit(1)

def download_mc4_turkish(output_file: str, max_size_gb: float = 0.75) -> bool:
    """
    C4 Turkish corpus indir (updated from deprecated mc4)
    
    Args:
        output_file: Çıktı dosyası
        max_size_gb: Maksimum boyut (GB)
    
    Returns:
        Başarılı mı?
    """
    print(f"📥 Downloading C4 Turkish corpus...")
    print(f"   Target size: {max_size_gb} GB")
    
    try:
        dataset = load_dataset("allenai/c4", "tr", streaming=True)
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        total_size = 0
        text_count = 0
        min_text_length = 100
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset['train'], desc="C4"):
                text = item.get('text', '').strip()
                
                if len(text) < min_text_length:
                    continue
                
                # Normalize whitespace
                text = ' '.join(text.split())
                
                if not text:
                    continue
                
                f.write(text + '\n')
                total_size += len(text.encode('utf-8'))
                text_count += 1
                
                if total_size >= max_size_bytes:
                    break
        
        final_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"✅ C4 downloaded: {final_size_gb:.2f} GB, {text_count:,} texts")
        return True
    
    except Exception as e:
        print(f"❌ C4 download error: {e}")
        return False

def download_wikipedia_turkish(output_file: str, max_size_gb: float = 0.75) -> bool:
    """
    Wikipedia Turkish corpus indir
    
    Args:
        output_file: Çıktı dosyası
        max_size_gb: Maksimum boyut (GB)
    
    Returns:
        Başarılı mı?
    """
    print(f"📥 Downloading Wikipedia Turkish corpus...")
    print(f"   Target size: {max_size_gb} GB")
    
    try:
        # Wikipedia dataset from HuggingFace (no authentication required)
        dataset = load_dataset("wikipedia", "20231101.tr", streaming=True)
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        total_size = 0
        text_count = 0
        min_text_length = 100
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset['train'], desc="Wikipedia"):
                text = item.get('text', '').strip()
                
                if len(text) < min_text_length:
                    continue
                
                # Normalize whitespace
                text = ' '.join(text.split())
                
                if not text:
                    continue
                
                f.write(text + '\n')
                total_size += len(text.encode('utf-8'))
                text_count += 1
                
                if total_size >= max_size_bytes:
                    break
        
        final_size_gb = total_size / (1024 * 1024 * 1024)
        print(f"✅ Wikipedia downloaded: {final_size_gb:.2f} GB, {text_count:,} texts")
        return True
    
    except Exception as e:
        print(f"❌ Wikipedia download error: {e}")
        print(f"💡 Tip: Wikipedia dataset requires 'datasets' library")
        return False

def merge_corpus_files(file1: str, file2: str, output_file: str) -> bool:
    """İki corpus dosyasını birleştir"""
    print(f"🔄 Merging corpus files...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as out:
            # File 1
            if os.path.exists(file1):
                with open(file1, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Merging file 1"):
                        if line.strip():
                            out.write(line)
            
            # File 2
            if os.path.exists(file2):
                with open(file2, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc="Merging file 2"):
                        if line.strip():
                            out.write(line)
        
        size_gb = os.path.getsize(output_file) / (1024 * 1024 * 1024)
        print(f"✅ Merged corpus: {size_gb:.2f} GB")
        return True
    
    except Exception as e:
        print(f"❌ Merge error: {e}")
        return False

def preprocess_with_morpho(
    input_file: str,
    output_file: str,
    morpho_splitter,
    max_lines: int = None
) -> bool:
    """
    Corpus'u morfem ayrımı ile işle
    Kök ve ekleri ayrı token'lara ayır

    Args:
        input_file: Input corpus dosyası
        output_file: İşlenmiş çıktı dosyası
        morpho_splitter: Morfem ayrımı için splitter (None ise basit işleme)
        max_lines: İşlenecek maksimum satır sayısı (None = tümü)

    Returns:
        Başarılı mı?
    """
    if morpho_splitter is None:
        print(f"📝 Preprocessing corpus with basic tokenization (no morphological analysis)...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print("")
    else:
        print(f"📝 Preprocessing corpus with morphological analysis...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print("")

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False

    processed_count = 0
    error_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            total_lines = sum(1 for _ in f_in)
            f_in.seek(0)

            for line_num, line in enumerate(tqdm(f_in, total=total_lines, desc="Processing")):
                if max_lines and line_num >= max_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    if morpho_splitter is not None:
                        # Morfem ayrımı ile işleme
                        analysis = morpho_splitter.split_sentence(line)

                        # Her kelime için kök ve ekleri ayır
                        morphemes = []
                        for word_analysis in analysis["kelimeler"]:
                            # Kök
                            root = word_analysis["kök"]
                            if root:
                                morphemes.append(root)

                            # Ekler (eksi işaretini kaldır)
                            for suffix in word_analysis["ekler"]:
                                suffix_clean = suffix.lstrip('-')
                                if suffix_clean:
                                    morphemes.append(suffix_clean)

                        # Morfemlerle yeni cümle oluştur (boşlukla ayrılmış)
                        if morphemes:
                            processed_line = ' '.join(morphemes)
                            f_out.write(processed_line + '\n')
                            processed_count += 1
                    else:
                        # Basit işleme: kelimeleri olduğu gibi bırak
                        # Sadece temel normalizasyon yap
                        words = line.split()
                        if words:
                            # Kelimeleri normalize et ve birleştir
                            normalized_words = []
                            for word in words:
                                # Temel temizlik
                                word = word.strip()
                                if word:
                                    normalized_words.append(word)

                            if normalized_words:
                                processed_line = ' '.join(normalized_words)
                                f_out.write(processed_line + '\n')
                                processed_count += 1

                except Exception as e:
                    error_count += 1
                    if error_count % 1000 == 0:
                        print(f"⚠️  Errors: {error_count}")
                    continue

    print(f"\n✅ Preprocessing completed!")
    print(f"   Processed: {processed_count:,} lines")
    print(f"   Errors: {error_count:,}")

    output_size_gb = os.path.getsize(output_file) / (1024 * 1024 * 1024)
    print(f"   Output size: {output_size_gb:.2f} GB")

    return True

def train_morphopiece(
    corpus_file: str,
    output_prefix: str,
    vocab_size: int = 32000,
    model_type: str = 'unigram',
    character_coverage: float = 1.0
) -> bool:
    """
    MorphoPiece tokenizer'ı eğit
    
    Args:
        corpus_file: İşlenmiş corpus dosyası
        output_prefix: Çıktı dosya öneki
        vocab_size: Vocabulary boyutu
        model_type: SentencePiece model tipi
        character_coverage: Character coverage
    
    Returns:
        Başarılı mı?
    """
    print(f"\n🔤 Training MorphoPiece tokenizer...")
    print(f"   Corpus: {corpus_file}")
    print(f"   Output: {output_prefix}")
    print(f"   Vocab size: {vocab_size:,}")
    print(f"   Model type: {model_type}")
    print(f"   Character coverage: {character_coverage}")
    print("")
    
    if not os.path.exists(corpus_file):
        print(f"❌ Corpus file not found: {corpus_file}")
        return False
    
    # Create output directory
    os.makedirs(os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else ".", exist_ok=True)
    
    try:
        # SentencePiece training parameters
        training_args = {
            'input': corpus_file,
            'model_prefix': output_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': character_coverage,
            'input_sentence_size': 10000000,  # Large for better quality
            'shuffle_input_sentence': True,
            'seed_sentencepiece_size': 10000000,
            'shrinking_factor': 0.75,
            'num_threads': 8,  # Multi-threading
            'max_sentence_length': 4192,
            'split_by_unicode_script': True,
            'split_by_whitespace': True,
            'split_by_number': True,
            'normalization_rule_name': 'nfkc_cf',  # Standard Unicode normalization (casefolded)
            'add_dummy_prefix': True,
            'remove_extra_whitespaces': True,
            'hard_vocab_limit': False,
            'use_all_vocab': False,
            'byte_fallback': True,
            'train_extremely_large_corpus': True,  # For large corpus
        }
        
        print("⏳ Training tokenizer (this may take 10-30 minutes)...")
        print("")
        
        # Train tokenizer
        spm.SentencePieceTrainer.train(**training_args)
        
        # Verify output files
        model_file = f"{output_prefix}.model"
        vocab_file = f"{output_prefix}.vocab"
        
        if not os.path.exists(model_file):
            print(f"❌ Model file not created: {model_file}")
            return False
        
        # Create vocab.json from .vocab file
        vocab_json = {}
        if os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        token = parts[0]
                        score = float(parts[1])
                        vocab_json[token] = {
                            'score': score,
                            'index': len(vocab_json)
                        }
        else:
            # Load from model and create vocab
            sp_processor = spm.SentencePieceProcessor(model_file=model_file)
            for i in range(sp_processor.vocab_size()):
                token = sp_processor.id_to_piece(i)
                vocab_json[token] = {
                    'score': 0.0,
                    'index': i
                }
        
        vocab_json_path = f"{output_prefix}_vocab.json"
        with open(vocab_json_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_json, f, ensure_ascii=False, indent=2)
        
        # Statistics
        model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
        vocab_size_actual = len(vocab_json)
        
        print("")
        print(f"✅ MorphoPiece training completed!")
        print(f"   Model file: {model_file} ({model_size_mb:.2f} MB)")
        print(f"   Vocab file: {vocab_file} ({vocab_size_actual:,} tokens)")
        print(f"   Vocab JSON: {vocab_json_path}")
        print("")
        print("💡 Usage:")
        print(f"   from agiformer.language.tokenizer import MorphoPiece")
        print(f"   morphopiece = MorphoPiece('{model_file}')")
        print(f"   tokens = morphopiece.encode('Dün markete gittim', morpho_aware=True)")
        
        return True
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Train MorphoPiece tokenizer with Turkish corpus')
    
    # Data collection
    parser.add_argument('--download', action='store_true',
                       help='Download corpus from MC4 and Wikipedia')
    parser.add_argument('--mc4-size', type=float, default=0.75,
                       help='MC4 corpus size in GB (default: 0.75)')
    parser.add_argument('--wikipedia-size', type=float, default=0.75,
                       help='Wikipedia corpus size in GB (default: 0.75)')
    parser.add_argument('--corpus-file', type=str, default='data/turkish_corpus/turkish_corpus_phase1.txt',
                       help='Combined corpus file path')
    
    # Preprocessing
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess corpus with morphological analysis')
    parser.add_argument('--preprocessed-file', type=str, default='data/turkish_corpus/turkish_corpus_morpho_processed.txt',
                       help='Preprocessed corpus file path')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum lines to process (None = all)')
    
    # Training
    parser.add_argument('--train', action='store_true',
                       help='Train MorphoPiece tokenizer')
    parser.add_argument('--output', type=str, default='tokenizer/morphopiece',
                       help='Output model prefix')
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size (default: 32000)')
    parser.add_argument('--model-type', type=str, default='unigram',
                       choices=['unigram', 'bpe', 'char', 'word'],
                       help='SentencePiece model type (default: unigram)')
    parser.add_argument('--character-coverage', type=float, default=1.0,
                       help='Character coverage (default: 1.0)')
    
    # All-in-one
    parser.add_argument('--all', action='store_true',
                       help='Run all steps: download + preprocess + train')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("    MorphoPiece Tokenizer Training")
    print("=" * 60)
    print("")

    # Initialize morphological splitter (may be None if import failed)
    morpho_splitter = None
    if RegexSplitter is not None:
        try:
            morpho_splitter = RegexSplitter()
            print("✅ RegexSplitter initialized successfully")
        except Exception as e:
            print(f"⚠️  Failed to initialize RegexSplitter: {e}")
            print("   Continuing with basic tokenization...")
            morpho_splitter = None
    else:
        print("⚠️  RegexSplitter not available, using basic tokenization")
        morpho_splitter = None

    print("")
    
    # Step 1: Download corpus
    if args.all or args.download:
        print("📥 Step 1: Downloading corpus...")
        print("")
        
        mc4_file = 'data/turkish_corpus/mc4_turkish.txt'
        wiki_file = 'data/turkish_corpus/wikipedia_turkish.txt'
        
        os.makedirs('data/turkish_corpus', exist_ok=True)
        
        # Download MC4
        if not os.path.exists(mc4_file) or os.path.getsize(mc4_file) < 1000:
            download_mc4_turkish(mc4_file, args.mc4_size)
        else:
            print(f"✅ MC4 corpus already exists: {mc4_file}")
        
        # Download Wikipedia
        if not os.path.exists(wiki_file) or os.path.getsize(wiki_file) < 1000:
            download_wikipedia_turkish(wiki_file, args.wikipedia_size)
        else:
            print(f"✅ Wikipedia corpus already exists: {wiki_file}")
        
        # Merge
        if os.path.exists(mc4_file) and os.path.exists(wiki_file):
            merge_corpus_files(mc4_file, wiki_file, args.corpus_file)
        elif os.path.exists(mc4_file):
            import shutil
            shutil.copy(mc4_file, args.corpus_file)
        elif os.path.exists(wiki_file):
            import shutil
            shutil.copy(wiki_file, args.corpus_file)
        
        print("")
    
    # Step 2: Preprocess with morphological analysis
    if args.all or args.preprocess:
        print("📝 Step 2: Preprocessing with morphological analysis...")
        print("")
        
        if not os.path.exists(args.corpus_file):
            print(f"❌ Corpus file not found: {args.corpus_file}")
            print("💡 Run with --download first")
            return
        
        preprocess_with_morpho(
            args.corpus_file,
            args.preprocessed_file,
            morpho_splitter,
            args.max_lines
        )
        
        print("")
    
    # Step 3: Train tokenizer
    if args.all or args.train:
        print("🔤 Step 3: Training MorphoPiece tokenizer...")
        print("")
        
        # First check if prepare_real_dataset.py output exists
        real_corpus_file = 'data/turkish_corpus/turkish_corpus_phase1.txt'
        real_corpus_jsonl = 'data/turkish_corpus/turkish_corpus_phase1.jsonl'
        
        # Priority order: real corpus JSONL > real corpus TXT > preprocessed > raw corpus
        if os.path.exists(real_corpus_jsonl):
            # Extract text from JSONL format
            import json
            temp_txt_file = 'data/turkish_corpus/temp_corpus_from_jsonl.txt'
            print(f"📝 Extracting text from JSONL corpus: {real_corpus_jsonl}")
            
            with open(real_corpus_jsonl, 'r', encoding='utf-8') as f_in:
                with open(temp_txt_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            if text.strip():
                                f_out.write(text + '\n')
                        except json.JSONDecodeError:
                            continue
            
            training_file = temp_txt_file
            print(f"✅ Extracted {temp_txt_file} for training")
        elif os.path.exists(real_corpus_file):
            training_file = real_corpus_file
            print(f"✅ Using real corpus file: {training_file}")
        else:
            # Determine input file (preprocessed if exists, otherwise raw)
            training_file = args.preprocessed_file if os.path.exists(args.preprocessed_file) else args.corpus_file
        
        if not os.path.exists(training_file):
            print(f"❌ Training corpus not found: {training_file}")
            print("💡 Run with --preprocess first")
            return
        
        success = train_morphopiece(
            training_file,
            args.output,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage
        )
        
        if success:
            print("\n" + "=" * 60)
            print("✅ MorphoPiece tokenizer training completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ Training failed!")
            print("=" * 60)
            # Use already imported module-level sys; avoid local import overshadowing
            sys.exit(1)
    
    # Default: run all if no specific step requested
    if not args.download and not args.preprocess and not args.train:
        print("💡 No steps specified. Running all steps...")
        print("")
        
        # Run all
        main_args = argparse.Namespace(**vars(args))
        main_args.all = True
        main_args.download = True
        main_args.preprocess = True
        main_args.train = True
        
        # Recursive call using module-level sys to avoid UnboundLocalError
        sys.argv = [sys.argv[0], '--all']
        main()

if __name__ == "__main__":
    main()
