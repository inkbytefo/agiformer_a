#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01

"""
============================================================================
TMA-1 Preprocessing Script - Morfolojik Analizi Ön İşleme
Ham corpus'u okuyup her token için morfo-tip bilgisi ekler
============================================================================
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agiformer.language.morpho_splitter import MorphoSplitter

# Detaylı morfem tipi mapping'i (23 kategori)
MORPHEME_TYPE_MAP = {
    'pad': 0,
    'special': 1,
    'isim_kök': 2,
    'fiil_kök': 3,
    'sıfat_kök': 4,
    'zarf_kök': 5,
    'iyelik_1tekil': 6,
    'iyelik_2tekil': 7,
    'iyelik_3tekil': 8,
    'iyelik_1çoğul': 9,
    'iyelik_2çoğul': 10,
    'iyelik_3çoğul': 11,
    'belirtme': 12,
    'yönelme': 13,
    'bulunma': 14,
    'ayrılma': 15,
    'ilgi': 16,
    'geçmiş_zaman': 17,
    'şimdiki_zaman': 18,
    'gelecek_zaman': 19,
    'geniş_zaman': 20,
    'çoğul': 21,
    'other': 22
}

# Reverse mapping (ID -> name)
MORPHEME_TYPE_NAMES = {v: k for k, v in MORPHEME_TYPE_MAP.items()}

# Semantic category mapping (Anlamsal Kategoriler) - MORFOSEMANTIK TOKENIZATION
SEMANTIC_CATEGORY_MAP = {
    'pad': 0,
    'special': 1,
    # Mekan kategorisi (location)
    'mekan': 2,  # ev, okul, şehir, park, sokak, bina, oda, vb.
    # Zaman kategorisi (time)
    'zaman': 3,  # gün, ay, yıl, saat, dakika, saniye, sabah, akşam, vb.
    # İnsan kategorisi (person)
    'insan': 4,  # adam, kadın, çocuk, öğretmen, doktor, öğrenci, vb.
    # Hayvan kategorisi (animal)
    'hayvan': 5,  # kedi, köpek, kuş, at, inek, vb.
    # Duygu kategorisi (emotion)
    'duygu': 6,  # mutluluk, üzüntü, sevinç, korku, öfke, vb.
    # Eşya kategorisi (object)
    'eşya': 7,  # masa, sandalye, kitap, kalem, telefon, araba, vb.
    # Yiyecek kategorisi (food)
    'yiyecek': 8,  # ekmek, su, çay, kahve, elma, vb.
    # Fiil kategorisi (action verbs)
    'fiil_eylem': 9,  # gitmek, gelmek, yapmak, okumak, yazmak, vb.
    # Sıfat kategorisi (adjective)
    'sıfat': 10,  # büyük, küçük, güzel, iyi, kötü, vb.
    # Bilinmeyen/anlamsal kategori belirlenemeyen
    'belirsiz': 11
}

# Basit keyword-based semantic categorizer
# İleride WordNet veya daha gelişmiş bir yapı ile genişletilebilir
SEMANTIC_KEYWORDS = {
    'mekan': {
        'ev', 'okul', 'şehir', 'kent', 'park', 'sokak', 'cadde', 'bina', 'oda', 'salon',
        'mutfak', 'bahçe', 'mahalle', 'ilçe', 'il', 'ülke', 'köy', 'kasaba', 'hastane',
        'kütüphane', 'mağaza', 'dükkan', 'ofis', 'fabrika', 'restoran', 'cafe', 'market',
        'süpermarket', 'okul', 'üniversite', 'müze', 'sinema', 'tiyatro', 'otopark'
    },
    'zaman': {
        'gün', 'gece', 'sabah', 'öğle', 'akşam', 'ay', 'yıl', 'saat', 'dakika',
        'saniye', 'hafta', 'pazartesi', 'salı', 'çarşamba', 'perşembe', 'cuma', 'cumartesi',
        'pazar', 'ocak', 'şubat', 'mart', 'nisan', 'mayıs', 'haziran', 'temmuz', 'ağustos',
        'eylül', 'ekim', 'kasım', 'aralık', 'bugün', 'dün', 'yarın', 'sabah', 'öğleden',
        'akşamüstü', 'geceyarısı'
    },
    'insan': {
        'adam', 'kadın', 'erkek', 'kız', 'çocuk', 'bebek', 'öğretmen', 'doktor', 'öğrenci',
        'anne', 'baba', 'kardeş', 'amca', 'dayı', 'teyze', 'hala', 'nine', 'dede', 'büyükanne',
        'büyükbaba', 'hasta', 'mühendis', 'avukat', 'polis', 'öğretmen', 'müdür', 'müdire',
        'başkan', 'genç', 'yaşlı', 'yetişkin'
    },
    'hayvan': {
        'kedi', 'köpek', 'kuş', 'at', 'inek', 'koyun', 'keçi', 'tavuk', 'horoz', 'ördek',
        'balık', 'tavşan', 'fare', 'sıçan', 'aslan', 'kaplan', 'ayı', 'kurt', 'tilki',
        'domuz', 'at', 'eşek', 'deve', 'kuzu', 'oğlak', 'yavru'
    },
    'eşya': {
        'masa', 'sandalye', 'kitap', 'kalem', 'defter', 'telefon', 'bilgisayar', 'araba',
        'bisiklet', 'motosiklet', 'gömlek', 'pantolon', 'ayakkabı', 'çanta', 'anahtar',
        'kapı', 'pencere', 'duvar', 'tavan', 'zemin', 'yol', 'köprü', 'bina', 'ev',
        'televizyon', 'radyo', 'kamera', 'saat', 'tablo', 'resim'
    },
    'yiyecek': {
        'ekmek', 'su', 'çay', 'kahve', 'süt', 'peynir', 'yumurta', 'et', 'tavuk', 'balık',
        'elma', 'armut', 'muz', 'portakal', 'domates', 'salatalık', 'soğan', 'patates',
        'pirinç', 'makarna', 'ekmek', 'börek', 'çörek', 'tatlı', 'çikolata', 'şeker',
        'meyve', 'sebze', 'salata'
    },
    'fiil_eylem': {
        # Bu kategori genelde morfolojik analiz ile belirlenir, ama bazı örnekler:
        'git', 'gel', 'yap', 'oku', 'yaz', 'oku', 'söyle', 'düşün', 'anla', 'gör',
        'bil', 'al', 'ver', 'sat', 'satın', 'öğren', 'öğret', 'çalış', 'oyna', 'koş'
    },
    'sıfat': {
        'büyük', 'küçük', 'güzel', 'çirkin', 'iyi', 'kötü', 'uzun', 'kısa', 'geniş',
        'dar', 'yüksek', 'alçak', 'sıcak', 'soğuk', 'sıcak', 'hızlı', 'yavaş', 'akıllı',
        'aptal', 'mutlu', 'üzgün', 'yeni', 'eski', 'temiz', 'kirli', 'aç', 'tok', 'susuz'
    }
}

def _detect_possessive_person(suffix: str) -> str:
    """
    İyelik ekinin şahıs bilgisini ek string'inden çıkar
    
    Args:
        suffix: İyelik eki string'i (örn: "im", "in", "leri")
    
    Returns:
        Şahıs bilgisi: "1tekil", "2tekil", "3tekil", "1çoğul", "2çoğul", "3çoğul", veya None
    """
    suffix_lower = suffix.lower().strip('-')
    
    # 1. tekil: -im, -ım, -um, -üm
    if suffix_lower in ['im', 'ım', 'um', 'üm']:
        return '1tekil'
    
    # 2. tekil: -in, -ın, -un, -ün
    if suffix_lower in ['in', 'ın', 'un', 'ün']:
        return '2tekil'
    
    # 3. tekil: -i, -ı, -u, -ü, -si, -sı, -su, -sü
    if suffix_lower in ['i', 'ı', 'u', 'ü', 'si', 'sı', 'su', 'sü', 'yi', 'yı', 'yu', 'yü']:
        return '3tekil'
    
    # 1. çoğul: -imiz, -ımız, -umuz, -ümüz
    if suffix_lower in ['imiz', 'ımız', 'umuz', 'ümüz']:
        return '1çoğul'
    
    # 2. çoğul: -iniz, -ınız, -unuz, -ünüz
    if suffix_lower in ['iniz', 'ınız', 'unuz', 'ünüz']:
        return '2çoğul'
    
    # 3. çoğul: -leri, -ları
    if suffix_lower in ['leri', 'ları']:
        return '3çoğul'
    
    return None

def get_detailed_morpho_type(token: str, morpho_splitter: MorphoSplitter) -> int:
    """
    Token'ın detaylı morfolojik tipini belirle (23 kategori)
    
    Args:
        token: Token string
        morpho_splitter: MorphoSplitter instance
    
    Returns:
        Detaylı morfolojik tip ID (0-22)
    """
    # Özel token'lar için
    if not token or token.strip() == "":
        return MORPHEME_TYPE_MAP['pad']
    
    if token.startswith("<") and token.endswith(">"):
        return MORPHEME_TYPE_MAP['special']
    
    # Morfolojik analiz yap
    analysis = morpho_splitter.split_word(token.strip())
    
    if not analysis['morfemler']:
        return MORPHEME_TYPE_MAP['other']
    
    # Kök kontrolü: Eğer ek yoksa veya sadece kök varsa
    if not analysis['ekler'] or len(analysis['morfemler']) == 1:
        root_morf = analysis['morfemler'][0]
        root_type = root_morf.get('tür', 'kök').lower()
        
        # Kök tipine göre belirle
        if 'isim' in root_type or 'noun' in root_type:
            return MORPHEME_TYPE_MAP['isim_kök']
        elif 'fiil' in root_type or 'verb' in root_type:
            return MORPHEME_TYPE_MAP['fiil_kök']
        elif 'sıfat' in root_type or 'adj' in root_type or 'adjective' in root_type:
            return MORPHEME_TYPE_MAP['sıfat_kök']
        elif 'zarf' in root_type or 'adv' in root_type or 'adverb' in root_type:
            return MORPHEME_TYPE_MAP['zarf_kök']
        else:
            # Varsayılan olarak isim kök
            return MORPHEME_TYPE_MAP['isim_kök']
    
    # Ekler varsa, son ekten başlayarak kontrol et (en önemli ek genelde sonda)
    # İlk önce zaman eklerine bak (fiiller için)
    for morf in reversed(analysis['morfemler'][1:]):  # Kökü atla, sondan başla
        morf_type = morf.get('tür', '').lower()
        morf_surface = morf.get('morfem', '').lower().strip('-')
        
        # Zaman ekleri (öncelik: yüklem kontrolü)
        if 'geçmiş_zaman' in morf_type or 'görülen_geçmiş' in morf_type or 'past' in morf_type:
            return MORPHEME_TYPE_MAP['geçmiş_zaman']
        elif 'şimdiki_zaman' in morf_type or 'prog' in morf_type or 'cont' in morf_type:
            return MORPHEME_TYPE_MAP['şimdiki_zaman']
        elif 'gelecek_zaman' in morf_type or 'fut' in morf_type:
            return MORPHEME_TYPE_MAP['gelecek_zaman']
        elif 'geniş_zaman' in morf_type or 'aor' in morf_type:
            return MORPHEME_TYPE_MAP['geniş_zaman']
        
        # Çoğul eki
        if 'çoğul' in morf_type or morf_surface in ['ler', 'lar']:
            return MORPHEME_TYPE_MAP['çoğul']
        
        # Durum ekleri
        if 'belirtme' in morf_type or 'acc' in morf_type:
            return MORPHEME_TYPE_MAP['belirtme']
        elif 'yönelme' in morf_type or 'dat' in morf_type:
            return MORPHEME_TYPE_MAP['yönelme']
        elif 'bulunma' in morf_type or 'loc' in morf_type:
            return MORPHEME_TYPE_MAP['bulunma']
        elif 'ayrılma' in morf_type or 'abl' in morf_type:
            return MORPHEME_TYPE_MAP['ayrılma']
        elif 'ilgi' in morf_type or 'gen' in morf_type or morf_surface == 'ki':
            return MORPHEME_TYPE_MAP['ilgi']
        
        # İyelik ekleri (şahıs bilgisi ile)
        if 'iyelik' in morf_type or 'poss' in morf_type:
            person = _detect_possessive_person(morf_surface)
            if person:
                if person == '1tekil':
                    return MORPHEME_TYPE_MAP['iyelik_1tekil']
                elif person == '2tekil':
                    return MORPHEME_TYPE_MAP['iyelik_2tekil']
                elif person == '3tekil':
                    return MORPHEME_TYPE_MAP['iyelik_3tekil']
                elif person == '1çoğul':
                    return MORPHEME_TYPE_MAP['iyelik_1çoğul']
                elif person == '2çoğul':
                    return MORPHEME_TYPE_MAP['iyelik_2çoğul']
                elif person == '3çoğul':
                    return MORPHEME_TYPE_MAP['iyelik_3çoğul']
            # Şahıs belirlenemezse genel iyelik (3. tekil varsayılan)
            return MORPHEME_TYPE_MAP['iyelik_3tekil']
    
    # Hiçbir kategori bulunamazsa other
    return MORPHEME_TYPE_MAP['other']

# Geriye dönük uyumluluk için eski fonksiyon
def get_morpho_type(token: str, morpho_splitter: MorphoSplitter) -> int:
    """
    Token'ın morfolojik tipini belirle (GERİYE DÖNÜK UYUMLULUK - eski 5 kategori)
    
    Args:
        token: Token string
        morpho_splitter: MorphoSplitter instance
    
    Returns:
        Morfolojik tip (0=root, 1=suffix, 2=verb, 3=other, 4=pad)
    """
    detailed_type = get_detailed_morpho_type(token, morpho_splitter)
    
    # Detaylı tipten eski kategoriye dönüştür
    if detailed_type == MORPHEME_TYPE_MAP['pad']:
        return 4
    elif detailed_type == MORPHEME_TYPE_MAP['special']:
        return 3
    elif detailed_type in [MORPHEME_TYPE_MAP['geçmiş_zaman'], 
                           MORPHEME_TYPE_MAP['şimdiki_zaman'],
                           MORPHEME_TYPE_MAP['gelecek_zaman'],
                           MORPHEME_TYPE_MAP['geniş_zaman']]:
        return 2  # Verb
    elif detailed_type in [MORPHEME_TYPE_MAP['isim_kök'],
                           MORPHEME_TYPE_MAP['fiil_kök'],
                           MORPHEME_TYPE_MAP['sıfat_kök'],
                           MORPHEME_TYPE_MAP['zarf_kök']]:
        return 0  # Root
    else:
        return 1  # Suffix

def get_semantic_category(token: str, morpho_splitter: MorphoSplitter) -> int:
    """
    Token'ın anlamsal kategorisini belirle (MORFOSEMANTIK TOKENIZATION)
    
    Args:
        token: Token string
        morpho_splitter: MorphoSplitter instance (kök bulmak için)
    
    Returns:
        Semantic category ID (0-11)
    """
    # Özel token'lar için
    if not token or token.strip() == "":
        return SEMANTIC_CATEGORY_MAP['pad']
    
    if token.startswith("<") and token.endswith(">"):
        return SEMANTIC_CATEGORY_MAP['special']
    
    # Kökü bul (morfem analizi ile)
    analysis = morpho_splitter.split_word(token.strip())
    root = analysis.get('kök', token.strip().lower())
    root_lower = root.lower()
    
    # Keyword-based matching (basit ama etkili)
    for category, keywords in SEMANTIC_KEYWORDS.items():
        if root_lower in keywords:
            return SEMANTIC_CATEGORY_MAP[category]
    
    # Morfolojik tip bilgisi kullanarak fallback
    morpho_type = get_detailed_morpho_type(token, morpho_splitter)
    
    # Fiil kökleri için özel kategori
    if morpho_type == MORPHEME_TYPE_MAP['fiil_kök']:
        return SEMANTIC_CATEGORY_MAP['fiil_eylem']
    
    # Sıfat kökleri için özel kategori
    if morpho_type == MORPHEME_TYPE_MAP['sıfat_kök']:
        return SEMANTIC_CATEGORY_MAP['sıfat']
    
    # Bilinmeyen
    return SEMANTIC_CATEGORY_MAP['belirsiz']

def preprocess_corpus(
    input_file: str,
    output_file: str,
    tokenizer_path: str,
    morpho_splitter: MorphoSplitter,
    max_lines: int = None,
    split_dataset: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> bool:
    """
    Corpus'u ön işleme tabi tutup JSONL formatında kaydet
    
    Args:
        input_file: Ham corpus dosyası
        output_file: Çıktı JSONL dosyası (veya train/val/test dosyaları için base path)
        tokenizer_path: SentencePiece tokenizer model path
        morpho_splitter: MorphoSplitter instance
        max_lines: Maksimum işlenecek satır sayısı (None = hepsi)
        split_dataset: Eğer True ise, corpus'u train/val/test olarak böl
        train_ratio: Train set oranı (default: 0.8)
        val_ratio: Validation set oranı (default: 0.1)
        test_ratio: Test set oranı (default: 0.1)
        random_seed: Random seed for shuffling (default: 42)
    
    Returns:
        Başarılı ise True
    """
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print(f"💡 Train tokenizer first: python src/train_morphopiece.py --train")
        return False
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer: {tokenizer_path}")
    tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size():,})")
    
    # Count lines in input file
    print(f"\n📊 Counting lines in corpus: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    if max_lines:
        total_lines = min(total_lines, max_lines)
    
    print(f"✅ Found {total_lines:,} lines to process")
    
    # Determine output files based on split_dataset flag
    if split_dataset:
        # Split into train/val/test files
        base_path = output_file.replace('.jsonl', '') if output_file.endswith('.jsonl') else output_file
        train_file = f"{base_path}_train.jsonl"
        val_file = f"{base_path}_val.jsonl"
        test_file = f"{base_path}_test.jsonl"
        output_files = {
            'train': train_file,
            'val': val_file,
            'test': test_file
        }
        print(f"\n📊 Dataset will be split:")
        print(f"   Train: {train_file} ({train_ratio*100:.0f}%)")
        print(f"   Validation: {val_file} ({val_ratio*100:.0f}%)")
        print(f"   Test: {test_file} ({test_ratio*100:.0f}%)")
        
        # Validate ratios sum to 1.0
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            print(f"⚠️  Warning: Ratios sum to {train_ratio + val_ratio + test_ratio:.6f}, not 1.0")
            print(f"   Normalizing ratios...")
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        # Open all output files
        f_outs = {
            'train': open(train_file, 'w', encoding='utf-8'),
            'val': open(val_file, 'w', encoding='utf-8'),
            'test': open(test_file, 'w', encoding='utf-8')
        }
    else:
        output_files = {'all': output_file}
        f_outs = {'all': open(output_file, 'w', encoding='utf-8')}
    
    # Process corpus
    print(f"\n🔄 Processing corpus...")
    print(f"   Input: {input_file}")
    
    processed_count = 0
    error_count = 0
    all_lines = []
    
    # First pass: Read all lines into memory (for shuffling if splitting)
    if split_dataset:
        print(f"📖 Reading all lines into memory for shuffling...")
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line_idx, line in enumerate(tqdm(f_in, total=total_lines, desc="Reading")):
                if max_lines and line_idx >= max_lines:
                    break
                all_lines.append((line_idx, line.strip()))
        
        # Shuffle with seed for reproducibility
        print(f"🔀 Shuffling {len(all_lines):,} lines (seed={random_seed})...")
        random.seed(random_seed)
        random.shuffle(all_lines)
        
        # Calculate split indices
        num_lines = len(all_lines)
        train_end = int(num_lines * train_ratio)
        val_end = train_end + int(num_lines * val_ratio)
        
        print(f"📊 Split indices: Train [0:{train_end:,}), Val [{train_end:,}:{val_end:,}), Test [{val_end:,}:{num_lines:,})")
        
        # Assign lines to splits
        lines_by_split = {
            'train': all_lines[:train_end],
            'val': all_lines[train_end:val_end],
            'test': all_lines[val_end:]
        }
        
        # Process each split
        for split_name, lines in lines_by_split.items():
            print(f"\n🔄 Processing {split_name} set ({len(lines):,} lines)...")
            f_out = f_outs[split_name]
            
            for line_idx, line in tqdm(lines, desc=f"Processing {split_name}", unit="lines"):
                if not line:
                    # Boş satır için boş output
                    output = {
                        "tokens": [],
                        "morpho_types": [],
                        "semantic_categories": []
                    }
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                    processed_count += 1
                    continue
                
                try:
                    # Tokenize
                    token_ids = tokenizer.encode(line, out_type=int)
                    
                    # Decode token IDs to get token strings
                    tokens = []
                    for token_id in token_ids:
                        token = tokenizer.id_to_piece(token_id)
                        # Remove SentencePiece prefix (▁ for space)
                        if token.startswith('▁'):
                            token = token[1:]
                        if not token:
                            token = '<SPACE>'
                        tokens.append(token)
                    
                    # Get detailed morphological types and semantic categories for each token
                    morpho_types = []
                    semantic_categories = []
                    for token in tokens:
                        morpho_type = get_detailed_morpho_type(token, morpho_splitter)
                        semantic_category = get_semantic_category(token, morpho_splitter)
                        morpho_types.append(morpho_type)
                        semantic_categories.append(semantic_category)
                    
                    # Write JSONL line
                    output = {
                        "tokens": tokens,
                        "morpho_types": morpho_types,
                        "semantic_categories": semantic_categories
                    }
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"\n⚠️  Error processing {split_name} line {line_idx}: {e}")
                    # Error durumunda boş output yaz
                    output = {
                        "tokens": [],
                        "morpho_types": [],
                        "semantic_categories": []
                    }
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
    else:
        # Original behavior: single output file, streaming processing
        f_out = f_outs['all']
        with open(input_file, 'r', encoding='utf-8') as f_in:
            progress_bar = tqdm(total=total_lines, desc="Processing", unit="lines")
            
            for line_idx, line in enumerate(f_in):
                if max_lines and line_idx >= max_lines:
                    break
                
                line = line.strip()
                if not line:
                    # Boş satır için boş output
                    output = {
                        "tokens": [],
                        "morpho_types": [],
                        "semantic_categories": []
                    }
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                    processed_count += 1
                    progress_bar.update(1)
                    continue
                
                try:
                    # Tokenize
                    token_ids = tokenizer.encode(line, out_type=int)
                    
                    # Decode token IDs to get token strings
                    tokens = []
                    for token_id in token_ids:
                        token = tokenizer.id_to_piece(token_id)
                        # Remove SentencePiece prefix (▁ for space)
                        if token.startswith('▁'):
                            token = token[1:]
                        if not token:
                            token = '<SPACE>'
                        tokens.append(token)
                    
                    # Get detailed morphological types and semantic categories for each token
                    morpho_types = []
                    semantic_categories = []
                    for token in tokens:
                        morpho_type = get_detailed_morpho_type(token, morpho_splitter)
                        semantic_category = get_semantic_category(token, morpho_splitter)
                        morpho_types.append(morpho_type)
                        semantic_categories.append(semantic_category)
                    
                    # Write JSONL line
                    output = {
                        "tokens": tokens,
                        "morpho_types": morpho_types,
                        "semantic_categories": semantic_categories
                    }
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                    processed_count += 1
                
                except Exception as e:
                    error_count += 1
                    print(f"\n⚠️  Error processing line {line_idx}: {e}")
                    # Error durumunda boş output yaz
                    output = {
                        "tokens": [],
                        "morpho_types": [],
                        "semantic_categories": []
                    }
                    f_out.write(json.dumps(output, ensure_ascii=False) + '\n')
                
                progress_bar.update(1)
            
            progress_bar.close()
    
    # Close all output files
    for f_out in f_outs.values():
        f_out.close()
    
    print(f"\n✅ Preprocessing completed!")
    print(f"   Processed: {processed_count:,} lines")
    print(f"   Errors: {error_count:,} lines")
    if split_dataset:
        print(f"   Output files:")
        for split_name, file_path in output_files.items():
            print(f"     {split_name}: {file_path}")
    else:
        print(f"   Output: {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess corpus for TMA-1 training')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input corpus file (raw text, one line per sentence)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file (preprocessed with morpho types)')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to SentencePiece tokenizer model (.model file)')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum number of lines to process (default: all)')
    parser.add_argument('--use-java', action='store_true', default=False,
                       help='Use Zemberek Java (if available) for morphological analysis')
    parser.add_argument('--split-dataset', action='store_true', default=False,
                       help='Split corpus into train/val/test sets (80/10/10)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for shuffling (default: 42)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MorphoSplitter
    print(f"\n📝 Initializing MorphoSplitter...")
    morpho_splitter = MorphoSplitter(use_java=args.use_java)
    
    # Preprocess corpus
    success = preprocess_corpus(
        input_file=args.input,
        output_file=args.output,
        tokenizer_path=args.tokenizer,
        morpho_splitter=morpho_splitter,
        max_lines=args.max_lines,
        split_dataset=args.split_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    if success:
        print(f"\n✅ Preprocessing successful!")
        if args.split_dataset:
            base_path = args.output.replace('.jsonl', '') if args.output.endswith('.jsonl') else args.output
            train_file = f"{base_path}_train.jsonl"
            val_file = f"{base_path}_val.jsonl"
            test_file = f"{base_path}_test.jsonl"
            print(f"💡 Next step: Train TMA-1 model with:")
            print(f"   python train_tma1.py --corpus {train_file} --val-corpus {val_file} --tokenizer {args.tokenizer}")
            print(f"   Test set saved to: {test_file}")
        else:
            print(f"💡 Next step: Train TMA-1 model with:")
            print(f"   python train_tma1.py --corpus {args.output} --tokenizer {args.tokenizer}")
    else:
        print(f"\n❌ Preprocessing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
