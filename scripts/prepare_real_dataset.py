# Developer: inkbytefo
# Modified: 2025-11-06

#!/usr/bin/env python3
"""
Turkish Text Corpus Preparation for AGIFORMER Phase 1
Downloads and prepares a large-scale Turkish text corpus for benchmarking experiments
Supports OSCAR, C4, and other common language model training corpora
"""

import os
import sys
import json
import requests
import argparse
import hashlib
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import zlib
from urllib.parse import urlparse

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


class TurkishTextCorpus:
    """
    Handles downloading, cleaning, and preparing Turkish text corpus
    for AGIFORMER Phase 1 experiments
    """

    def __init__(self, output_dir: str, target_size_gb: float = 1.0):
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # Supported datasets
        self.dataset_urls = {
            'wikipedia_oscar_turkish': {
                'name': 'Wikipedia OSCAR Turkish',
                'urls': [
                    'https://huggingface.co/datasets/musabg/wikipedia-oscar-tr/resolve/main/data/ wikipedia-oscar-tr.1mb.jsonl'
                ],
                'expected_size_gb': 1.0,
                'description': 'Turkish Wikipedia corpus from OSCAR dataset'
            },
            'synthetic_turkish': {
                'name': 'Synthetic Turkish Corpus',
                'urls': [],
                'expected_size_gb': 1.0,
                'description': 'High-quality synthetic Turkish text for testing'
            },
            'oscar_turkish': {
                'name': 'OSCAR Turkish Subset',
                'urls': [
                    'https://huggingface.co/datasets/oscar-corpus/OSCAR_2201/resolve/main/splits/vi_mc4_ deduplicated.1GB.jsonl.gz',
                ],
                'expected_size_gb': 1.2,
                'description': 'OSCAR corpus Turkish language subset'
            }
        }
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.cleaned_dir = self.output_dir / "cleaned"
        self.final_file = self.output_dir / "turkish_corpus_phase1.txt"
        
        self.raw_dir.mkdir(exist_ok=True)
        self.cleaned_dir.mkdir(exist_ok=True)

    def download_with_progress(self, url: str, output_path: Path, desc: str = "") -> bool:
        """Download file with progress tracking and resume capability"""
        try:
            # Check if file already exists
            if output_path.exists():
                logger.info(f"File already exists: {output_path}")
                return True

            # Create temporary file for download
            temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
            
            # Get file size for resume capability
            resume_header = {}
            if temp_path.exists():
                resume_header['Range'] = f'bytes={temp_path.stat().st_size}-'

            logger.info(f"Downloading {url} to {output_path}")
            
            response = requests.get(url, headers=resume_header, stream=True, timeout=30)
            response.raise_for_status()
            
            # Get total size
            total_size = int(response.headers.get('content-length', 0))
            if resume_header:
                total_size += temp_path.stat().st_size
            
            # Download with progress
            mode = 'ab' if resume_header else 'wb'
            with open(temp_path, mode) as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                downloaded = temp_path.stat().st_size if temp_path.exists() else 0
                pbar.update(downloaded)
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))

            # Move to final location
            temp_path.rename(output_path)
            logger.info(f"Download completed: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False

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
        if len([c for c in text if c in turkish_chars]) / len(text) < 0.5:
            return ""
        
        return text

    def process_jsonl_file(self, file_path: Path) -> List[str]:
        """Process JSONL file and extract text content"""
        texts = []
        
        try:
            logger.info(f"Processing JSONL file: {file_path}")
            
            # Handle gzipped files
            if file_path.suffix == '.gz':
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'
            
            with opener(file_path, mode, encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0:
                        logger.info(f"Processed {line_num} lines...")
                    
                    try:
                        # Try to parse as JSON
                        data = json.loads(line)
                        
                        # Extract text based on common field names
                        text = ""
                        for field in ['text', 'content', 'sentence', 'document', 'article']:
                            if field in data:
                                text = str(data[field])
                                break
                        
                        # Fallback: use first string field
                        if not text:
                            for key, value in data.items():
                                if isinstance(value, str) and len(value) > 50:
                                    text = value
                                    break
                        
                        if text:
                            cleaned = self.clean_turkish_text(text)
                            if cleaned:
                                texts.append(cleaned)
                    
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        cleaned = self.clean_turkish_text(line)
                        if cleaned:
                            texts.append(cleaned)
                    
                    # Stop if we have enough text
                    current_size = sum(len(t) for t in texts)
                    if current_size > self.target_size_bytes:
                        logger.info(f"Target size reached ({current_size / (1024*1024):.1f} MB)")
                        break
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return texts

    def create_turkish_corpus(self, dataset_type: str = 'synthetic_turkish') -> str:
        """Main function to create Turkish corpus"""
        logger.info(f"Creating Turkish corpus using {dataset_type}")
        logger.info(f"Target size: {self.target_size_gb} GB")
        
        texts = []
        total_size = 0
        
        if dataset_type == 'synthetic_turkish':
            # Generate synthetic Turkish text
            sentences = self.generate_synthetic_turkish_text(200000)  # Generate lots of text
            texts = sentences
            
        else:
            # Download and process real dataset
            if dataset_type not in self.dataset_urls:
                raise ValueError(f"Unsupported dataset: {dataset_type}")
            
            dataset_info = self.dataset_urls[dataset_type]
            logger.info(f"Using dataset: {dataset_info['name']}")
            
            # Download files
            for i, url in enumerate(dataset_info['urls']):
                if total_size > self.target_size_bytes:
                    break
                
                file_name = Path(urlparse(url).path).name
                local_path = self.raw_dir / file_name
                
                if self.download_with_progress(url, local_path, f"Downloading {file_name}"):
                    # Process downloaded file
                    if local_path.suffix in ['.jsonl', '.jsonl.gz', '.json', '.json.gz']:
                        new_texts = self.process_jsonl_file(local_path)
                        texts.extend(new_texts)
                        total_size = sum(len(t) for t in texts)
                        
                        logger.info(f"Current corpus size: {total_size / (1024*1024):.1f} MB")
        
        # Write final corpus
        logger.info(f"Writing {len(texts)} lines to final corpus file...")
        with open(self.final_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Get final size
        final_size = self.final_file.stat().st_size
        logger.info(f"Final corpus size: {final_size / (1024*1024):.1f} MB ({final_size / (1024*1024*1024):.2f} GB)")
        
        # Create metadata file
        metadata = {
            'dataset_type': dataset_type,
            'total_lines': len(texts),
            'total_characters': final_size,
            'total_size_mb': final_size / (1024*1024),
            'total_size_gb': final_size / (1024*1024*1024),
            'target_size_gb': self.target_size_gb,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'description': f'Turkish text corpus for AGIFORMER Phase 1 experiments'
        }
        
        metadata_file = self.output_dir / 'corpus_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Corpus preparation completed!")
        logger.info(f"Final corpus: {self.final_file}")
        logger.info(f"Metadata: {metadata_file}")
        
        return str(self.final_file)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prepare Turkish text corpus for AGIFORMER Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_real_dataset.py --dataset synthetic_turkish --output data/turkish_corpus --size 1.0
  python prepare_real_dataset.py --dataset oscar_turkish --output data/turkish_corpus --size 2.0
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wikipedia_oscar_turkish', 'synthetic_turkish', 'oscar_turkish'],
        default='synthetic_turkish',
        help='Dataset type to use (default: synthetic_turkish)'
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
        default=1.0,
        help='Target corpus size in GB (default: 1.0)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing corpus'
    )
    
    args = parser.parse_args()
    
    if args.validate:
        corpus_path = Path(args.output) / 'turkish_corpus_phase1.txt'
        if corpus_path.exists():
            size_mb = corpus_path.stat().st_size / (1024*1024)
            print(f"✅ Corpus exists: {corpus_path}")
            print(f"📊 Size: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")
            
            # Count lines
            with open(corpus_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for line in f)
            print(f"📝 Lines: {lines:,}")
        else:
            print(f"❌ Corpus not found: {corpus_path}")
        return
    
    # Create corpus
    corpus_creator = TurkishTextCorpus(args.output, args.size)
    
    try:
        final_path = corpus_creator.create_turkish_corpus(args.dataset)
        
        print("\n" + "="*60)
        print("TURKISH CORPUS PREPARATION COMPLETED!")
        print("="*60)
        print(f"Output directory: {args.output}")
        print(f"Corpus file: {final_path}")
        print(f"Size: {args.size} GB")
        print(f"Dataset: {args.dataset}")
        print("\nTo use this corpus in training:")
        print(f"python train_phase1.py --config conf/model/agiformer-lite.yaml --data {final_path} --name agiformer-lite")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Corpus preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()