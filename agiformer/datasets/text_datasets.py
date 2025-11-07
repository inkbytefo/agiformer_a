# Developer: inkbytefo
# Modified: 2025-11-06

"""
Text Dataset Classes for AGIFORMER
Consolidated Turkish text dataset implementations
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sentencepiece import SentencePieceProcessor
import os
import json
from typing import Optional, Dict, Tuple, List, Union
import random
from pathlib import Path


class TurkishTextDataset(Dataset):
    """Dataset for Turkish text corpus with optional morphological preprocessing"""
    
    def __init__(
        self,
        corpus_file: str,
        tokenizer: Optional[SentencePieceProcessor] = None,
        max_seq_len: int = 512,
        buffer_size: int = 10000,
        is_jsonl: bool = False,
        vocab_size: int = 32000
    ):
        """
        Args:
            corpus_file: Path to corpus file (text or JSONL)
            tokenizer: SentencePiece tokenizer instance
            max_seq_len: Maximum sequence length
            buffer_size: Number of lines to buffer in memory
            is_jsonl: If True, corpus_file is JSONL with preprocessed morpho_types
            vocab_size: Vocabulary size for fallback character encoding
        """
        self.corpus_file = Path(corpus_file)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.is_jsonl = is_jsonl
        self.vocab_size = vocab_size
        
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        # Detect file format if not specified
        if not is_jsonl:
            self.is_jsonl = self.corpus_file.suffix == '.jsonl'
        
        # Count lines in file
        print(f"Counting lines in corpus: {corpus_file}")
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            self.num_lines = sum(1 for _ in f)
        
        print(f"Found {self.num_lines:,} lines")
        if self.is_jsonl:
            print(f"   Format: JSONL (with preprocessed morpho_types)")
        else:
            print(f"   Format: Text (morpho_types will be computed at runtime)")
        
        self._buffer = []
        self._buffer_start_idx = 0
    
    def _load_buffer(self, start_idx: int):
        """Load buffer of lines from file"""
        self._buffer = []
        self._buffer_start_idx = start_idx
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            # Skip to start position
            for _ in range(start_idx):
                try:
                    next(f)
                except StopIteration:
                    break
            
            # Load buffer
            for _ in range(self.buffer_size):
                line = f.readline()
                if not line:
                    break
                if self.is_jsonl:
                    # JSONL format: parse JSON
                    try:
                        data = json.loads(line.strip())
                        self._buffer.append(data)
                    except json.JSONDecodeError:
                        self._buffer.append({"tokens": [], "morpho_types": []})
                else:
                    # Text format: just store line
                    self._buffer.append(line.strip())
    
    def __len__(self) -> int:
        return self.num_lines
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single training example"""
        # Check if idx is in current buffer
        if idx < self._buffer_start_idx or idx >= self._buffer_start_idx + len(self._buffer):
            # Load new buffer
            buffer_start = max(0, idx - self.buffer_size // 2)
            self._load_buffer(buffer_start)
        
        # Get line from buffer
        local_idx = idx - self._buffer_start_idx
        if local_idx >= len(self._buffer):
            # Fallback: read directly
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                for _ in range(idx):
                    f.readline()
                line = f.readline().strip()
                if self.is_jsonl:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        data = {"tokens": [], "morpho_types": []}
                else:
                    data = line
        else:
            data = self._buffer[local_idx]
        
        # Process data based on format
        if self.is_jsonl:
            # JSONL format: data already has tokens, morpho_types, and semantic_categories
            tokens = data.get("tokens", [])
            morpho_types_list = data.get("morpho_types", [])
            semantic_categories_list = data.get("semantic_categories", [])
            
            # Encode tokens to IDs
            token_ids = []
            for i, token in enumerate(tokens):
                # Encode single token (SentencePiece handles tokenization)
                if self.tokenizer:
                    try:
                        # Try to encode as single token
                        ids = self.tokenizer.encode(token, out_type=int)
                        if ids:
                            # Clamp IDs to valid vocabulary range
                            valid_ids = []
                            for token_id in ids:
                                if 0 <= token_id < self.vocab_size:
                                    valid_ids.append(token_id)
                                else:
                                    # Use UNK for out-of-vocabulary tokens
                                    valid_ids.append(self.tokenizer.unk_id())
                            
                            token_ids.extend(valid_ids)
                            # If token maps to multiple IDs, extend morpho_types and semantic_categories accordingly
                            if len(valid_ids) > 1:
                                last_morpho = morpho_types_list[-1] if morpho_types_list else 3
                                last_semantic = semantic_categories_list[-1] if semantic_categories_list else 11
                                morpho_types_list.extend([last_morpho] * (len(valid_ids) - 1))
                                semantic_categories_list.extend([last_semantic] * (len(valid_ids) - 1))
                        else:
                            # Fallback: use UNK
                            unk_id = self.tokenizer.unk_id()
                            token_ids.append(unk_id if unk_id < self.vocab_size else 0)
                            morpho_types_list.append(3)  # Other
                            semantic_categories_list.append(11)  # belirsiz
                    except:
                        # Fallback: use UNK
                        unk_id = self.tokenizer.unk_id()
                        token_ids.append(unk_id if unk_id < self.vocab_size else 0)
                        morpho_types_list.append(3)  # Other
                        semantic_categories_list.append(11)  # belirsiz
                else:
                    # Fallback to character encoding
                    token_ids.extend([ord(c) % self.vocab_size for c in token])
                    morpho_types_list.extend([3] * len(token))
                    semantic_categories_list.extend([11] * len(token))
        else:
            # Text format: tokenize normally
            text = data if isinstance(data, str) else ""
            if self.tokenizer:
                raw_ids = self.tokenizer.encode(text, out_type=int)
                # Clamp IDs to valid vocabulary range
                token_ids = []
                for token_id in raw_ids:
                    if 0 <= token_id < self.vocab_size:
                        token_ids.append(token_id)
                    else:
                        # Use UNK for out-of-vocabulary tokens
                        unk_id = self.tokenizer.unk_id()
                        token_ids.append(unk_id if unk_id < self.vocab_size else 0)
            else:
                # Fallback to character encoding
                token_ids = [ord(c) % self.vocab_size for c in text[:self.max_seq_len]]
            morpho_types_list = [3] * len(token_ids)  # Default: other (will be ignored)
            semantic_categories_list = [11] * len(token_ids)  # Default: belirsiz (will be ignored)
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_len:
            # Randomly select a window
            start = random.randint(0, len(token_ids) - self.max_seq_len)
            token_ids = token_ids[start:start + self.max_seq_len]
            morpho_types_list = morpho_types_list[start:start + self.max_seq_len]
            semantic_categories_list = semantic_categories_list[start:start + self.max_seq_len]
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        input_morpho_types = morpho_types_list[:-1]
        target_morpho_types = morpho_types_list[1:]
        input_semantic_categories = semantic_categories_list[:-1]
        target_semantic_categories = semantic_categories_list[1:]

        # Pad to max_seq_len
        if self.tokenizer:
            pad_id = self.tokenizer.pad_id() if hasattr(self.tokenizer, 'pad_id') else self.tokenizer.unk_id()
        else:
            pad_id = 0
        pad_morpho_type = 0  # pad type
        pad_semantic_category = 0  # pad semantic category
        
        padding_len = self.max_seq_len - len(input_ids) - 1
        if padding_len > 0:
            input_ids = input_ids + [pad_id] * padding_len
            target_ids = target_ids + [pad_id] * padding_len
            input_morpho_types = input_morpho_types + [pad_morpho_type] * padding_len
            target_morpho_types = target_morpho_types + [pad_morpho_type] * padding_len
            input_semantic_categories = input_semantic_categories + [pad_semantic_category] * padding_len
            target_semantic_categories = target_semantic_categories + [pad_semantic_category] * padding_len
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(
                [1] * (len(input_ids) - padding_len) + [0] * padding_len,
                dtype=torch.long
            ) if padding_len > 0 else torch.ones(len(input_ids), dtype=torch.long)
        }
        
        # Add morpho_types and semantic_categories if available
        if self.is_jsonl:
            result['morpho_types'] = torch.tensor(input_morpho_types, dtype=torch.long)
            result['semantic_categories'] = torch.tensor(input_semantic_categories, dtype=torch.long)
        
        return result


class TextDataset(Dataset):
    """Flexible dataset supporting both .jsonl and .txt formats with morphological analysis"""
    
    def __init__(self, file_path: str, tokenizer, max_seq_len: int = 512, use_morpho: bool = True):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_morpho = use_morpho
        
        # Load and cache texts
        print(f"Loading corpus from: {self.file_path}")
        self.texts = []
        self.morpho_types = []
        
        if self.file_path.suffix == '.jsonl':
            self._load_jsonl()
        elif self.file_path.suffix == '.txt':
            self._load_txt()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        print(f"Loaded {len(self.texts)} texts from corpus")
    
    def _load_jsonl(self):
        """Load from .jsonl format"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            text = data['text']
                            self.texts.append(text)
                            # Extract morpho_types if available
                            if 'morpho_types' in data and self.use_morpho:
                                self.morpho_types.append(data['morpho_types'])
                            elif self.use_morpho:
                                self.morpho_types.append(None)
                        else:
                            print(f"Line {line_num} missing 'text' field, skipping")
                    except json.JSONDecodeError as e:
                        print(f"Line {line_num} not valid JSON, trying as text: {e}")
                        self.texts.append(line.strip())
                        if self.use_morpho:
                            self.morpho_types.append(None)
    
    def _load_txt(self):
        """Load from .txt format (one text per line)"""
        from agiformer.language.morpho_splitter import MorphoSplitter
        
        morpho_splitter = None
        if self.use_morpho:
            morpho_splitter = MorphoSplitter()
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    text = line.strip()
                    self.texts.append(text)
                    
                    if self.use_morpho:
                        # Generate morpho_types on-the-fly using MorphoSplitter
                        try:
                            words = text.split()
                            morpho_types = []
                            for word in words:
                                # Use MorphoSplitter to analyze each word
                                analysis = morpho_splitter.split_word(word)
                                # Get morphological types for each morpheme
                                for morfem in analysis['morfemler']:
                                    morpho_types.append(self._get_morpho_type(morfem['morfem'], morfem['tür']))
                            self.morpho_types.append(morpho_types)
                        except Exception as e:
                            print(f"Failed to analyze morphology for line {line_num}: {e}")
                            self.morpho_types.append(None)
    
    def _get_morpho_type(self, morpheme: str, morfem_tur: str) -> int:
        """Map morphological analysis to integer types for model input"""
        # Map Turkish morphological types to integer codes for model input
        type_mapping = {
            'kök': 0,           # stem/root
            'belirtme': 1,      # accusative
            'yönelme': 2,       # dative
            'bulunma': 3,       # locative
            'ayrılma': 4,       # ablative
            'ilgi': 5,          # genitive
            'iyelik_1tekil': 6, # 1st person singular
            'iyelik_2tekil': 7, # 2nd person singular
            'iyelik_3tekil': 8, # 3rd person singular
            'iyelik_1çoğul': 9, # 1st person plural
            'iyelik_2çoğul': 10, # 2nd person plural
            'iyelik_3çoğul': 11, # 3rd person plural
            'çoğul': 12,        # plural
            'şimdiki_zaman': 13, # progressive
            'geçmiş_zaman': 14,  # past tense
            'gelecek_zaman': 15, # future
            'şart': 16,         # conditional
            'emir': 17,         # imperative
            'mastar': 18,       # infinitive
            'olumsuz': 19,      # negative
            'ek': 20            # general suffix
        }
        
        return type_mapping.get(morfem_tur, 0)  # Default to stem (0) if unknown
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text using the tokenizer
        raw_ids = self.tokenizer.encode(text, out_type=int)
        
        # Clamp IDs to valid vocabulary range
        token_ids = []
        for token_id in raw_ids:
            if 0 <= token_id < self.tokenizer.vocab_size():
                token_ids.append(token_id)
            else:
                # Use UNK for out-of-vocabulary tokens
                unk_id = self.tokenizer.unk_id()
                token_ids.append(unk_id if unk_id < self.tokenizer.vocab_size() else 0)
        
        # Truncate if needed
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        
        # Create input-target pairs (causal language modeling)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        
        # Add padding
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_id()] * pad_len
            target_ids += [self.tokenizer.pad_id()] * pad_len
        
        # Create attention mask
        attention_mask = (torch.tensor(input_ids, dtype=torch.long) != self.tokenizer.pad_id()).long()
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': attention_mask,
            'text': text
        }
        
        # Add morpho_types if available and needed
        if self.use_morpho and idx < len(self.morpho_types) and self.morpho_types[idx] is not None:
            result['morpho_types'] = torch.tensor(self.morpho_types[idx][:self.max_seq_len-1], dtype=torch.long)
        
        return result


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration and testing"""
    
    def __init__(self, texts, max_seq_len=512, vocab_size=256):
        self.texts = texts
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Convert to character IDs
        char_ids = [ord(c) % self.vocab_size for c in text[:self.max_seq_len]]
        
        # Pad or truncate
        if len(char_ids) < self.max_seq_len:
            char_ids = char_ids + [0] * (self.max_seq_len - len(char_ids))
        else:
            char_ids = char_ids[:self.max_seq_len]
        
        # Input and target (shifted by 1)
        input_ids = torch.tensor(char_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(char_ids[1:], dtype=torch.long)
        
        return input_ids, target_ids


def create_dataloader(
    corpus_file: str,
    tokenizer: Optional[SentencePieceProcessor] = None,
    tokenizer_path: Optional[str] = None,
    batch_size: int = 8,
    max_seq_len: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    is_jsonl: Optional[bool] = None,
    vocab_size: int = 32000
) -> Tuple[DataLoader, Optional[SentencePieceProcessor]]:
    """
    Create DataLoader for training
    
    Args:
        corpus_file: Path to corpus file (text or JSONL)
        tokenizer: SentencePiece tokenizer instance (optional)
        tokenizer_path: Path to SentencePiece model file (optional)
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        is_jsonl: If True, corpus_file is JSONL with preprocessed morpho_types
        vocab_size: Vocabulary size for fallback character encoding
    
    Returns:
        Tuple of (DataLoader instance, tokenizer)
    """
    # Load tokenizer if not provided
    if tokenizer is None and tokenizer_path:
        tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
    
    # Detect file format if not specified
    if is_jsonl is None:
        is_jsonl = Path(corpus_file).suffix == '.jsonl'
    
    # Create dataset
    dataset = TurkishTextDataset(
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        is_jsonl=is_jsonl,
        vocab_size=vocab_size
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Drop last incomplete batch
    )
    
    return dataloader, tokenizer