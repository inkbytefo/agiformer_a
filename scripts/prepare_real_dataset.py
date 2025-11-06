# Developer: inkbytefo
# Modified: 2025-11-06

#!/usr/bin/env python3
"""
Turkish Text Corpus Preparation for AGIFORMER Phase 1
Downloads and prepares high-quality Turkish text corpus using HuggingFace Datasets
Supports FineWiki Turkish and other HuggingFace datasets
"""

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging
from tqdm import tqdm
import re

# Add parent directory to path to import agiformer
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import datasets library
try:
    from datasets import load_dataset, DatasetDict
    DATASETS_AVAILABLE = True
    logger.info("Datasets library available")
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("Datasets library not available. Install with: pip install datasets")


class TurkishTextCorpus:
    """
    Handles downloading, cleaning, and preparing Turkish text corpus
    for AGIFORMER Phase 1 experiments using HuggingFace Datasets
    """

    def __init__(self, output_dir: str, target_size_gb: float = 1.0):
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # Supported datasets from HuggingFace Hub
        self.dataset_configs = {
            'finewiki_turkish': {
                'name': 'FineWiki Turkish',
                'dataset_path': 'HuggingFaceFW/finewiki',
                'language': 'tr',
                'expected_size_gb': 2.5,
                'description': 'High-quality Turkish Wikipedia articles from FineWiki',
                'text_field': 'text',
                'requires_login': False
            },
            'synthetic_turkish': {
                'name': 'Synthetic Turkish Corpus',
                'dataset_path': None,
                'expected_size_gb': 1.0,
                'description': 'High-quality synthetic Turkish text for testing',
                'text_field': 'text',
                'requires_login': False
            },
            'oscar_turkish': {
                'name': 'OSCAR Turkish',
                'dataset_path': 'oscar-corpus/OSCAR_2201',
                'language': 'tr',
                'expected_size_gb': 3.0,
                'description': 'OSCAR corpus Turkish language subset',
                'text_field': 'text',
                'requires_login': False
            },
            'mc4_turkish': {
                'name': 'mC4 Turkish',
                'dataset_path': 'mc4',
                'language': 'tr',
                'expected_size_gb': 15.0,
                'description': 'Massive multilingual C4 corpus Turkish subset',
                'text_field': 'text',
                'requires_login': True
            }
        }
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.cleaned_dir = self.output_dir / "cleaned"
        self.final_file = self.output_dir / "turkish_corpus_phase1.txt"
        self.final_jsonl = self.output_dir / "turkish_corpus_phase1.jsonl"
        
        self.raw_dir.mkdir(exist_ok=True)
        self.cleaned_dir.mkdir(exist_ok=True)

    def generate_synthetic_turkish_text(self, num_lines: int = 100000) -> List[str]:
        """Generate high-quality synthetic Turkish text for testing"""
        logger.info(f"Generating {num_lines} lines of synthetic Turkish text...")
        
        # Turkish vocabulary and patterns
        turkish_words = {
            'nouns': ['ev', 'okul', 'kitap', 'masa', 'sandalye', 'araba', 'ağaç', 'gül', 'kedi', 'köpek', 'deniz', 'dağ', 'nehir', 'şehir', 'köy', 'park', 'bahçe', 'plaj', 'hava', 'gökyüzü', 'güneş', 'ay', 'yıldız', 'deniz', 'rüzgar', 'yağmur', 'kar', 'sis', 'fırtına'],
            'verbs': ['git', 'gel', 'yap', 'ol', 'ver', 'al', 'gör', 'dinle', 'okuma', 'yazma', 'konuşma', 'koş', 'dur', 'otur', 'kalk', 'aç', 'kapat', 'başla', 'bitir', 'öğren', 'öğret', 'anla', 'hiss', 'sev', 'kork', 'umut', 'hayal', 'düşün', 'hatırla'],
            'adjectives': ['büyük', 'küçük', 'iyi', 'kötü', 'güzel', 'çirkin', 'hızlı', 'yavaş', 'yüksek', 'alçak', 'yeni', 'eski', 'sıcak', 'soğuk', 'taze', 'bayat', 'kolay', 'zor', 'mutlu', 'üzgün', 'kızgın', 'heyecanlı', 'sakin', 'gürültülü', 'sessiz', 'parlak', 'karanlık'],
            'adverbs': ['çok', 'az', 'biraz', 'çokça', 'hiç', 'her zaman', 'asla', 'bugün', 'yarın', 'dün', 'şimdi', 'sonra', 'önce', 'burada', 'orada', 'şurada', 'uzak', 'yakın', 'yukarı', 'aşağı', 'içeri', 'dışarı', 'evet', 'hayır'],
            'pronouns': ['ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'bu', 'şu', 'o', 'bunlar', 'şunlar', 'onlar', 'kimin', 'neyin', 'neyi', 'kimi', 'hangi', 'nerede', 'ne zaman', 'nasıl', 'niçin']
        }
        
        # Sentence templates
        sentence_templates = [
            "{nouns} {verbs} {adverbs}.",
            "Bu {nouns} çok {adjectives}.",
            "Ben {verbs} {adverbs} çünkü {nouns} {adjectives}.",
            "{pronouns} {verbs} {nouns} {adverbs}.",
            "Her {nouns} {verbs} {adverbs}.",
            "{nouns} ve {nouns} {verbs} {adverbs}.",
            "Eğer {nouns} {verbs} ise, {pronouns} {verbs} {adverbs}.",
            "Bugün {nouns} {verbs} {adverbs} çünkü hava {adjectives}.",
            "{pronouns} {nouns} {verbs} {adverbs} ve {nouns} {adjectives}.",
            "Yarın {nouns} {verbs} {adverbs} olacak."
        ]
        
        import random
        random.seed(42)  # Reproducible results
        
        sentences = []
        for i in range(num_lines):
            # Select random template
            template = random.choice(sentence_templates)
            
            # Fill template with random words
            sentence = template
            for word_type in ['nouns', 'verbs', 'adjectives', 'adverbs', 'pronouns']:
                if word_type in template:
                    word = random.choice(turkish_words[word_type])
                    sentence = sentence.replace(word_type, word, 1)
            
            # Ensure proper sentence ending
            if not sentence.endswith(('.', '?', '!')):
                sentence += '.'
                
            sentences.append(sentence)
        
        return sentences

    def clean_turkish_text(self, text: str) -> str:
        """
        Advanced Turkish text cleaning
        """
        if not isinstance(text, str):
            return ""
            
        # Remove null bytes and control characters
        text = text.replace('\x00', '').replace('\r', '\n')
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\+90[0-9]{10}', '', text)
        text = re.sub(r'\([0-9]{3}\)[0-9]{3}-[0-9]{4}', '', text)
        
        # Clean repeated patterns (like movie subtitles)
        subtitle_patterns = [
            r'^\d+$',  # Line numbers
            r'^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}$',  # Timestamps
            r'^\d{2}:\d{2}:\d{2}$',  # Simple timestamps
            r'^[►❖⇨▂▃▅▆█]+.*$',  # Special characters
            r'^YAPIM YILI:.*$',
            r'^ÜLKE:.*$',
            r'^ÜYE YORUMU:.*$',
            r'izle,izle,',
            r'full izle,',
            r'ENGLISH SUBTITLES',
            r'IMDb Top 250',
        ]
        
        for pattern in subtitle_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Filter out lines that are too short or too long
        if len(text) < 20 or len(text) > 2000:
            return ""
        
        # Basic Turkish character validation
        turkish_chars = set('abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ')
        if len(text) > 0 and len([c for c in text if c in turkish_chars]) / len(text) < 0.3:
            return ""
        
        return text

    def load_huggingface_dataset(self, dataset_config: Dict) -> List[str]:
        """Load dataset from 🤗 Hub"""
        if not DATASETS_AVAILABLE:
            raise ImportError("HuggingFace Datasets library not available. Install with: pip install datasets")
        
        logger.info(f"Loading dataset: {dataset_config['name']}")
        logger.info(f"Path: {dataset_config['dataset_path']}")
        
        texts = []
        
        try:
            # Load dataset with language configuration if specified
            if 'language' in dataset_config:
                # For multilingual datasets like mc4, oscar
                dataset = load_dataset(
                    dataset_config['dataset_path'],
                    dataset_config['language'],
                    split='train',
                    trust_remote_code=True
                )
            else:
                # For single language datasets
                dataset = load_dataset(
                    dataset_config['dataset_path'],
                    split='train',
                    trust_remote_code=True
                )
            
            logger.info(f"Dataset loaded with {len(dataset)} examples")
            
            # Extract text content
            text_field = dataset_config['text_field']
            for example in tqdm(dataset, desc="Processing examples"):
                if text_field in example:
                    text = example[text_field]
                    cleaned = self.clean_turkish_text(text)
                    if cleaned:
                        texts.append(cleaned)
                
                # Check if we've reached target size
                current_size = sum(len(t) for t in texts)
                if current_size > self.target_size_bytes:
                    logger.info(f"Target size reached ({current_size / (1024*1024):.1f} MB)")
                    break
            
            logger.info(f"Extracted {len(texts)} clean text examples")
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_config['dataset_path']}: {e}")
            raise
        
        return texts

    def create_turkish_corpus(self, dataset_type: str = 'finewiki_turkish', 
                           output_format: str = 'txt') -> str:
        """Main function to create Turkish corpus"""
        logger.info(f"Creating Turkish corpus using {dataset_type}")
        logger.info(f"Target size: {self.target_size_gb} GB")
        logger.info(f"Output format: {output_format}")
        
        texts = []
        
        if dataset_type == 'synthetic_turkish':
            # Generate synthetic Turkish text
            sentences = self.generate_synthetic_turkish_text(200000)
            texts = sentences
            
        else:
            # Load dataset from HuggingFace Hub
            if dataset_type not in self.dataset_configs:
                raise ValueError(f"Unsupported dataset: {dataset_type}")
            
            dataset_config = self.dataset_configs[dataset_type]
            logger.info(f"Using dataset: {dataset_config['name']}")
            
            # Check if login is required
            if dataset_config.get('requires_login', False):
                logger.warning("This dataset requires HuggingFace authentication.")
                logger.warning("Please run: huggingface-cli login")
            
            texts = self.load_huggingface_dataset(dataset_config)
        
        # Write final corpus
        if output_format == 'jsonl':
            output_file = self.final_jsonl
            logger.info(f"Writing {len(texts)} lines to JSONL format...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    # Create JSONL entry with morphological analysis placeholder
                    entry = {
                        'text': text,
                        'tokens': text.split(),  # Simple tokenization for now
                        'morpho_types': [3] * len(text.split()),  # Default: other
                        'semantic_categories': [11] * len(text.split())  # Default: belirsiz
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            output_file = self.final_file
            logger.info(f"Writing {len(texts)} lines to text format...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + '\n')
        
        # Get final size
        final_size = output_file.stat().st_size
        logger.info(f"Final corpus size: {final_size / (1024*1024):.1f} MB ({final_size / (1024*1024*1024):.2f} GB)")
        
        # Create metadata file
        metadata = {
            'dataset_type': dataset_type,
            'dataset_name': self.dataset_configs[dataset_type]['name'],
            'dataset_path': self.dataset_configs[dataset_type].get('dataset_path'),
            'total_lines': len(texts),
            'total_characters': final_size,
            'total_size_mb': final_size / (1024*1024),
            'total_size_gb': final_size / (1024*1024*1024),
            'target_size_gb': self.target_size_gb,
            'output_format': output_format,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': f'Turkish text corpus for AGIFORMER Phase 1 experiments',
            'source': 'HuggingFace Datasets Hub'
        }
        
        metadata_file = self.output_dir / 'corpus_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Corpus preparation completed!")
        logger.info(f"Final corpus: {output_file}")
        logger.info(f"Metadata: {metadata_file}")
        
        return str(output_file)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prepare Turkish text corpus for AGIFORMER Phase 1 using HuggingFace Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use FineWiki Turkish dataset (recommended)
  python prepare_real_dataset.py --dataset finewiki_turkish --output data/turkish_corpus --size 2.0 --format jsonl
  
  # Use synthetic data for testing
  python prepare_real_dataset.py --dataset synthetic_turkish --output data/turkish_corpus --size 1.0
  
  # Use OSCAR Turkish
  python prepare_real_dataset.py --dataset oscar_turkish --output data/turkish_corpus --size 3.0 --format jsonl
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['finewiki_turkish', 'synthetic_turkish', 'oscar_turkish', 'mc4_turkish'],
        default='finewiki_turkish',
        help='Dataset type to use (default: finewiki_turkish)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/turkish_corpus',
        help='Output directory for corpus (default: data/turkish_corpus)'
    )
    
    parser.add_argument(
        '--size',
        type=float,
        default=2.0,
        help='Target corpus size in GB (default: 2.0)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'jsonl'],
        default='jsonl',
        help='Output format: txt or jsonl (default: jsonl)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing corpus'
    )
    
    args = parser.parse_args()
    
    if args.validate:
        corpus_file = Path(args.output) / f'turkish_corpus_phase1.{args.format}'
        if corpus_file.exists():
            size_mb = corpus_file.stat().st_size / (1024*1024)
            print(f"Corpus exists: {corpus_file}")
            print(f"Size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")
            
            if args.format == 'jsonl':
                # Count lines in JSONL
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for line in f)
                print(f"Lines: {lines:,}")
            else:
                # Count lines in text
                with open(corpus_file, 'r', encoding='utf-8') as f:
                    lines = sum(1 for line in f)
                print(f"Lines: {lines:,}")
        else:
            print(f"Corpus not found: {corpus_file}")
        return
    
    # Check dependencies
    if not DATASETS_AVAILABLE and args.dataset != 'synthetic_turkish':
        print("ERROR: Datasets library not available!")
        print("Install with: pip install datasets")
        print("Or use: --dataset synthetic_turkish")
        sys.exit(1)
    
    # Create corpus
    corpus_creator = TurkishTextCorpus(args.output, args.size)
    
    try:
        final_path = corpus_creator.create_turkish_corpus(args.dataset, args.format)
        
        print("\n" + "="*60)
        print("TURKISH CORPUS PREPARATION COMPLETED!")
        print("="*60)
        print(f"Output directory: {args.output}")
        print(f"Corpus file: {final_path}")
        print(f"Size: {args.size} GB")
        print(f"Dataset: {args.dataset}")
        print(f"Format: {args.format}")
        print("\nTo use this corpus in training:")
        if args.format == 'jsonl':
            print(f"python train.py data.data_path={final_path}")
        else:
            print(f"python train.py data.data_path={final_path}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Corpus preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()