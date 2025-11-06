# Developer: inkbytefo
# Modified: 2025-11-06

#!/usr/bin/env python3
"""
AGIFORMER Phase 1 Training Script
Specialized script for comparing AgglutinativeAttention vs Standard MultiHeadAttention
Designed for fair, reproducible benchmarking experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml
import json
import argparse
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from agiformer.utils import count_parameters, format_number, WarmupScheduler
from agiformer.language.tokenizer import MorphoPiece

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Enable anomaly detection for debugging gradient issues
torch.autograd.set_detect_anomaly(True)


class TextDataset(Dataset):
    """Flexible dataset supporting both .jsonl and .txt formats with morphological analysis"""
    
    def __init__(self, file_path: str, tokenizer, max_seq_len: int = 512, use_morpho: bool = True):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_morpho = use_morpho
        
        # Load and cache texts
        logger.info(f"Loading corpus from: {self.file_path}")
        self.texts = []
        self.morpho_types = []
        
        if self.file_path.suffix == '.jsonl':
            self._load_jsonl()
        elif self.file_path.suffix == '.txt':
            self._load_txt()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        logger.info(f"Loaded {len(self.texts)} texts from corpus")
    
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
                            logger.warning(f"Line {line_num} missing 'text' field, skipping")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num} not valid JSON, trying as text: {e}")
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
                            logger.warning(f"Failed to analyze morphology for line {line_num}: {e}")
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
        token_ids = self.tokenizer.encode(text, out_type=int)
        
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


# Legacy class for backward compatibility
class TurkishTextDataset(Dataset):
    """Dataset for Turkish text corpus with simple character-level encoding (legacy)"""
    
    def __init__(self, corpus_file: str, max_seq_len: int = 512, vocab_size: int = 32000):
        self.corpus_file = Path(corpus_file)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Load and cache texts
        logger.info(f"Loading corpus from: {self.corpus_file}")
        self.texts = []
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.texts.append(json.loads(line)['text'])
                    except (json.JSONDecodeError, KeyError):
                        # Fallback for plain text lines
                        self.texts.append(line.strip())
        
        logger.info(f"Loaded {len(self.texts)} texts from corpus")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Simple character-level encoding
        char_ids = [ord(c) % self.vocab_size for c in text[:self.max_seq_len]]
        
        # Pad or truncate to max_seq_len
        if len(char_ids) < self.max_seq_len:
            char_ids = char_ids + [0] * (self.max_seq_len - len(char_ids))
        else:
            char_ids = char_ids[:self.max_seq_len]
        
        # Create input-target pairs (causal language modeling)
        input_ids = torch.tensor(char_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(char_ids[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'text': text
        }


class Phase1Trainer:
    """Trainer for AGIFORMER Phase 1 experiments"""
    
    def __init__(
        self,
        model_config: Dict,
        experiment_name: str,
        output_dir: str = "outputs/phase1"
    ):
        self.model_config = model_config
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = MorphoPiece()
        self.tokenizer.vocab_size = model_config.get('vocab_size', 32000)
        
        # Training metrics storage
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_perplexities': [],
            'steps': [],
            'timestamps': []
        }
    
    def create_model(self) -> AGIFORMER:
        """Create AGIFORMER model with configuration"""
        model = AGIFORMER(
            tokenizer=self.tokenizer,
            d_model=self.model_config['d_model'],
            n_layers=self.model_config['n_layers'],
            n_heads=self.model_config['n_heads'],
            d_ff=self.model_config['d_ff'],
            n_experts=self.model_config['n_experts'],
            expert_types=self.model_config['expert_types'],
            memory_size=self.model_config.get('memory_size', 0),
            max_seq_len=self.model_config['max_seq_len'],
            dropout=self.model_config['dropout'],
            use_linear_attention=self.model_config.get('use_linear_attention', False),
            use_memory=self.model_config.get('use_memory', False),
            use_introspection=self.model_config.get('use_introspection', False),
            use_multimodal=self.model_config.get('use_multimodal', False),
            use_agglutinative_attention=self.model_config.get('use_agglutinative_attention', True),
            use_gradient_checkpointing=self.model_config.get('use_gradient_checkpointing', False)
        ).to(self.device)
        
        return model
    
    def train_epoch(self, model, dataloader, optimizer, criterion, epoch: int, current_step: int, max_steps: Optional[int] = None) -> Tuple[float, int]:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_steps and current_step >= max_steps:
                logger.info(f"Reached max_steps ({max_steps}), stopping training for this epoch.")
                break

            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad():
                logits, _ = model(input_ids=input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            if self.device.type == 'cuda':
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            current_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'step': current_step})
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Step: {current_step}, Loss: {loss.item():.4f}")
        
        processed_batches = batch_idx + 1 if 'batch_idx' in locals() and batch_idx is not None else 1
        return total_loss / processed_batches, current_step
    
    def validate(self, model, dataloader, criterion) -> Tuple[float, float]:
        """Validate model and return loss and perplexity"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                logits, _ = model(input_ids=input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, model, optimizer, epoch: int, step: int, metrics: Dict):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'model_config': self.model_config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_results(self, final_metrics: Dict):
        """Save final results and training summary"""
        results = {
            'experiment_name': self.experiment_name,
            'model_config': self.model_config,
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'completed_at': datetime.now().isoformat()
        }
        
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved: {results_path}")
        
        # Also save as markdown for easy reading
        md_path = self.output_dir / "training_summary.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.experiment_name} Training Results\n\n")
            f.write(f"**Completed:** {results['completed_at']}\n\n")
            f.write("## Model Configuration\n")
            for key, value in self.model_config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n## Final Metrics\n")
            for key, value in final_metrics.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n## Training Curves\n")
            f.write(f"- Best Validation Loss: {min(self.training_history['val_losses']):.4f}\n")
            f.write(f"- Best Validation Perplexity: {min(self.training_history['val_perplexities']):.4f}\n")
        
        logger.info(f"Training summary saved: {md_path}")
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        save_every: int = 5,
        val_every: int = 1,
        max_steps: Optional[int] = None
    ) -> Dict:
        """Main training function"""
        logger.info(f"Starting training: {self.experiment_name}")
        logger.info(f"Model config: {self.model_config}")
        
        # Create model
        model = self.create_model()
        
        # Log model info
        params = count_parameters(model)
        logger.info(f"Model Parameters: {format_number(params['total'])} total, {format_number(params['trainable'])} trainable")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        scheduler = WarmupScheduler(optimizer, warmup_steps=1000, d_model=self.model_config['d_model'])
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training loop
        best_val_loss = float('inf')
        best_val_perplexity = float('inf')
        global_step = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, global_step = self.train_epoch(model, train_loader, optimizer, criterion, epoch + 1, global_step, max_steps)
            
            # Check if max_steps reached
            if max_steps and global_step >= max_steps:
                logger.info(f"Max steps ({max_steps}) reached. Finalizing training.")
                # Perform a final validation
                val_loss, val_perplexity = self.validate(model, val_loader, criterion)
                logger.info(f"Final Val Loss: {val_loss:.4f}, Final Val Perplexity: {val_perplexity:.4f}")
                self.training_history['val_losses'].append(val_loss)
                self.training_history['val_perplexities'].append(val_perplexity)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_perplexity = val_perplexity
                break # Exit epoch loop

            # Validation
            if epoch % val_every == 0:
                val_loss, val_perplexity = self.validate(model, val_loader, criterion)
                
                # Update history
                self.training_history['train_losses'].append(train_loss)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['val_perplexities'].append(val_perplexity)
                self.training_history['steps'].append(global_step)
                self.training_history['timestamps'].append(time.time() - start_time)
                
                logger.info(f"Epoch {epoch + 1}/{num_epochs} completed:")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val Perplexity: {val_perplexity:.4f}")
                logger.info(f"  Time: {time.time() - epoch_start:.2f}s")
                
                # Save checkpoint if best
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_val_perplexity = val_perplexity
                
                if epoch % save_every == 0 or is_best:
                    self.save_checkpoint(model, optimizer, epoch, global_step, {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_perplexity': val_perplexity,
                        'best_val_loss': best_val_loss
                    })
            
            scheduler.step()
            # global_step is now updated inside train_epoch
        
        # Final metrics
        final_metrics = {
            'best_val_loss': best_val_loss,
            'best_val_perplexity': best_val_perplexity,
            'total_training_time': time.time() - start_time,
            'final_train_loss': train_loss
        }
        
        logger.info(f"Training completed! Best val loss: {best_val_loss:.4f}, Best perplexity: {best_val_perplexity:.4f}")
        
        # Save results
        self.save_results(final_metrics)
        
        return final_metrics


def load_config(config_path: str) -> Dict:
    """Load model configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(data_path: str, tokenizer, max_seq_len: int = 512, train_split: float = 0.9, use_morpho: bool = True) -> Tuple[Dataset, Dataset]:
    """Create train/val datasets from corpus file with flexible format support"""
    from torch.utils.data import random_split
    
    # Create full dataset with proper tokenizer and morphological analysis support
    full_dataset = TextDataset(data_path, tokenizer, max_seq_len=max_seq_len, use_morpho=use_morpho)
    
    # Split into train/val
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_dataset, val_dataset


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="AGIFORMER Phase 1 Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train AGIFORMER-Lite (with AgglutinativeAttention)
  python train_phase1.py --config conf/model/agiformer-lite.yaml --data data/turkish_corpus.txt --name agiformer-lite
  
  # Train Baseline (with standard MultiHeadAttention)
  python train_phase1.py --config conf/model/baseline.yaml --data data/turkish_corpus.txt --name baseline
  
  # Resume training
  python train_phase1.py --config conf/model/agiformer-lite.yaml --data data/turkish_corpus.txt --resume outputs/phase1/agiformer-lite/latest.pt
        """
    )
    
    parser.add_argument('--config', type=str, required=True, help='Model configuration file')
    parser.add_argument('--data', type=str, required=True, help='Turkish corpus file path')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--output', type=str, default='outputs/phase1', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--train_split', type=float, default=0.9, help='Train/validation split ratio')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    parser.add_argument('--dry_run', action='store_true', help='Test configuration without training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'max_seq_len': args.max_seq_len
    })
    
    logger.info(f"Configuration loaded: {config}")
    
    # Initialize tokenizer for dataset creation
    tokenizer = MorphoPiece()
    tokenizer.vocab_size = config.get('vocab_size', 32000)
    
    # Create datasets with flexible format support
    use_morpho = config.get('morphological_analysis', True)
    train_dataset, val_dataset = create_datasets(
        args.data, tokenizer, args.max_seq_len, args.train_split, use_morpho
    )
    
    if args.dry_run:
        logger.info("Dry run - testing configuration only")
        
        # Create trainer and model for testing
        trainer = Phase1Trainer(config, args.name, args.output)
        model = trainer.create_model()
        
        # Test forward pass
        sample_batch = next(iter(DataLoader(train_dataset, batch_size=2)))
        with torch.no_grad():
            sample_input = sample_batch['input_ids'][:1].to(trainer.device)
            try:
                logits, _ = model(input_ids=sample_input)
                logger.info(f"✅ Model test successful! Output shape: {logits.shape}")
                logger.info(f"   Config: {config.get('use_agglutinative_attention', 'N/A')} attention mechanism")
                return
            except Exception as e:
                logger.error(f"❌ Model test failed: {e}")
                return
    
    # Create trainer
    trainer = Phase1Trainer(config, args.name, args.output)
    
    # Train model
    try:
        final_metrics = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps
        )
        
        print("\n" + "="*60)
        print("🎉 PHASE 1 TRAINING COMPLETED!")
        print("="*60)
        print(f"Experiment: {args.name}")
        print(f"Best validation loss: {final_metrics['best_val_loss']:.4f}")
        print(f"Best validation perplexity: {final_metrics['best_val_perplexity']:.4f}")
        print(f"Training time: {final_metrics['total_training_time']:.2f}s")
        print(f"Output directory: {trainer.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()