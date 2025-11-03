"""
Professional Training Script for AGIFORMER
Enhanced with Checkpointing, W&B Logging, and Multimodal Support
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agiformer import AGIFORMER
from agiformer.utils import count_parameters, format_number, WarmupScheduler

# Import CC12M dataset
from scripts.prepare_cc12m import CC12MDataset

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class CheckpointManager:
    """Professional checkpoint management for AGIFORMER training"""
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_val_loss = float('inf')
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': {}  # Will be filled if needed
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best if needed
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved to {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint"""
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return self.load_checkpoint(latest_path)
        return None
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[self.keep_last_n:]:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")


class MetricsLogger:
    """Professional metrics logging with W&B integration"""
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config
            )
            self.logger = wandb
        else:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
    
    def log_training_metrics(self, step: int, loss: float, model_info: Dict[str, Any]):
        """Log training metrics"""
        metrics = {
            'Training/loss': loss,
            'Training/step': step
        }
        
        # Extract MoE metrics
        if 'blocks' in model_info:
            for i, block in enumerate(model_info['blocks']):
                if 'moe' in block and 'router_info' in block['moe']:
                    router_info = block['moe']['router_info']
                    
                    # Expert usage
                    if 'expert_usage' in router_info:
                        for j, usage in enumerate(router_info['expert_usage']):
                            metrics[f'Training/expert_usage_{i}_{j}'] = usage
                    
                    # Load balancing loss
                    if 'load_balancing_loss' in router_info:
                        metrics[f'Training/load_balancing_loss_{i}'] = router_info['load_balancing_loss']
                    
                    # Router confidence
                    if 'avg_router_confidence' in router_info:
                        metrics[f'Training/router_confidence_{i}'] = router_info['avg_router_confidence']
        
        # Extract memory metrics
        if 'memory' in model_info:
            memory_info = model_info['memory']
            if 'step_count' in memory_info:
                metrics['Training/memory_step_count'] = memory_info['step_count']
        
        # Extract introspection metrics
        if 'blocks' in model_info:
            for i, block in enumerate(model_info['blocks']):
                if 'introspection' in block and 'final_confidence' in block['introspection']:
                    metrics[f'Training/introspection_confidence_{i}'] = block['introspection']['final_confidence']
        
        # Multimodal metrics
        if 'multimodal' in model_info and model_info['multimodal']:
            metrics['Training/multimodal_active'] = 1
            if 'modalities' in model_info:
                for modality in model_info['modalities']:
                    metrics[f'Training/modality_{modality}'] = 1
        
        if self.use_wandb:
            self.logger.log(metrics)
        else:
            self.logger.info(f"Step {step}: {metrics}")
    
    def log_validation_metrics(self, step: int, val_loss: float, val_metrics: Dict[str, float]):
        """Log validation metrics"""
        metrics = {
            'Validation/loss': val_loss,
            'Validation/step': step,
            **val_metrics
        }
        
        if self.use_wandb:
            self.logger.log(metrics)
        else:
            self.logger.info(f"Validation Step {step}: {metrics}")
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters"""
        if self.use_wandb:
            self.logger.config.update(config)
        else:
            self.logger.info(f"Hyperparameters: {json.dumps(config, indent=2)}")
    
    def finish(self):
        """Finish logging"""
        if self.use_wandb:
            self.logger.finish()


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration"""
    
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


def prepare_batch_data(batch, device, is_multimodal):
    """Batch verisini model için hazırla - tek yerden yönet"""
    if is_multimodal:
        # Multimodal batch (sözlük yapısı)
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        model_inputs = {'text': input_ids, 'image': images}
    else:
        # Text-only batch (tuple yapısı)
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        model_inputs = {'text': input_ids}
    
    return model_inputs, target_ids


def validate_epoch(model, dataloader, criterion, device, use_amp=False, is_multimodal=False):
    """Validate for one epoch - refactor edilmiş ve standardize edilmiş"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Batch verisini hazırla
            model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)
            
            # Model çağrısı - standardize edilmiş
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    logits, info = model(**model_inputs)
                    
                    # Reshape for loss
                    logits = logits.view(-1, logits.size(-1))
                    target_ids = target_ids.view(-1)
                    
                    loss = criterion(logits, target_ids)
            else:
                logits, info = model(**model_inputs)
                
                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                target_ids = target_ids.view(-1)
                
                loss = criterion(logits, target_ids)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False, 
                metrics_logger=None, step_offset=0, is_multimodal=False):
    """Train for one epoch - refactor edilmiş ve standardize edilmiş"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, batch in enumerate(dataloader):
        # Batch verisini hazırla
        model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)
        
        optimizer.zero_grad()
        
        # Model çağrısı - standardize edilmiş
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits, info = model(**model_inputs)
                
                # Reshape for loss
                logits = logits.view(-1, logits.size(-1))
                target_ids = target_ids.view(-1)
                
                loss = criterion(logits, target_ids)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, info = model(**model_inputs)
            
            # Reshape for loss
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = criterion(logits, target_ids)
            
            # Add MoE load balancing loss
            total_loss_batch = loss
            for block_info in info.get('blocks', []):
                if 'moe' in block_info and 'load_balancing_loss' in block_info['moe']['router_info']:
                    lb_loss = block_info['moe']['router_info']['load_balancing_loss']
                    total_loss_batch = total_loss_batch + lb_loss
            
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        current_step = step_offset + batch_idx
        
        # Log metrics
        if metrics_logger and batch_idx % 10 == 0:  # Log every 10 batches
            metrics_logger.log_training_metrics(current_step, loss.item(), info)
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches, current_step + 1


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Professional Training Script for AGIFORMER")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default="agiformer_experiment",
                       help="Name of the experiment")
    parser.add_argument("--wandb_project", type=str, default="agiformer",
                       help="W&B project name")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_interval", type=int, default=500,
                       help="Evaluate every N steps")
    
    # Data
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Training data split ratio")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data file")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Path to CC12M dataset directory")
    
    # Other
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable W&B logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    return parser.parse_args()


def create_dataset(data_dir: Optional[str], data_path: Optional[str], model_config: Dict[str, Any], train_split: float = 0.9):
    """Create training and validation datasets"""
    
    # Check for CC12M dataset first
    if data_dir and Path(data_dir).exists():
        print(f"Loading CC12M dataset from {data_dir}")
        
        # Check if it's a valid CC12M dataset
        train_metadata = Path(data_dir) / "metadata_train.json"
        val_metadata = Path(data_dir) / "metadata_val.json"
        
        if train_metadata.exists() and val_metadata.exists():
            print("Found CC12M dataset structure")
            
            # Load train and validation datasets
            train_dataset = CC12MDataset(
                data_path=data_dir,
                split="train",
                max_samples=None,
                image_size=(224, 224),
                max_text_len=model_config['max_seq_len'],
                vocab_size=model_config['vocab_size']
            )
            
            val_dataset = CC12MDataset(
                data_path=data_dir,
                split="val",
                max_samples=None,
                image_size=(224, 224),
                max_text_len=model_config['max_seq_len'],
                vocab_size=model_config['vocab_size']
            )
            
            return train_dataset, val_dataset, True  # True for multimodal
        else:
            print(f"CC12M metadata files not found in {data_dir}")
            print(f"Expected: {train_metadata}, {val_metadata}")
    
    # Fallback to text-only dataset
    print("Using fallback text-only dataset")
    dummy_texts = [
        "This is a sample text for training.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Deep learning models require large datasets.",
        "Natural language processing is evolving rapidly.",
        "Transformers revolutionized AI research.",
        "Attention mechanisms are powerful.",
        "AGIFORMER represents the future of AI.",
    ] * 1000  # Repeat for more data
    
    dataset = SimpleTextDataset(
        dummy_texts,
        max_seq_len=model_config['max_seq_len'],
        vocab_size=model_config['vocab_size']
    )
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, False  # False for text-only


def main():
    # DEĞİŞİKLİK: parse_known_args() kullanarak bilinmeyen argümanları yakala
    parser = argparse.ArgumentParser(description="Professional Training Script for AGIFORMER")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--experiment_name", type=str, default="agiformer_experiment",
                       help="Name of the experiment")
    parser.add_argument("--wandb_project", type=str, default="agiformer",
                       help="W&B project name")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_interval", type=int, default=500,
                       help="Evaluate every N steps")
    
    # Data
    parser.add_argument("--train_split", type=float, default=0.9,
                       help="Training data split ratio")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data file")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Path to CC12M dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    
    # Other
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable W&B logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Enable W&B logging")
    
    # Komut satırı argümanlarını iki aşamada parse et:
    # 1. Bilinen argümanları al
    # 2. Geri kalanları (dinamik olanları) ayrıca al
    args, unknown_args = parser.parse_known_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Dinamik argümanları (unknown_args) parse et ve config'i güncelle
    # Örn: --model.n_experts 1 -> config['model']['n_experts'] = 1
    for i in range(0, len(unknown_args), 2):
        if i + 1 >= len(unknown_args):
            continue
            
        arg_name = unknown_args[i]
        arg_value = unknown_args[i+1]
        
        if not arg_name.startswith('--'):
            continue
        
        # Değeri doğru tipe dönüştürmeye çalış (int, float, bool, str)
        try:
            if '.' in arg_value:
                arg_value = float(arg_value)
            else:
                arg_value = int(arg_value)
        except ValueError:
            if arg_value.lower() == 'true':
                arg_value = True
            elif arg_value.lower() == 'false':
                arg_value = False
            # Aksi halde string olarak kalır
        
        # '.' ile ayrılmış anahtarları takip ederek config sözlüğünü güncelle
        # Örn: "model.n_experts" -> keys = ["model", "n_experts"]
        keys = arg_name.strip('-').split('.')
        d = config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = arg_value

    print("Updated config with command-line arguments:", json.dumps(config, indent=2))
    
    model_config = config['model']
    train_config = config['training']
    
    # Override config with command line arguments
    if args.epochs:
        train_config['max_steps'] = args.epochs
    if args.batch_size:
        train_config['batch_size'] = args.batch_size
    if args.learning_rate:
        train_config['learning_rate'] = args.learning_rate
    
    # Setup
    device = torch.device(train_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir="checkpoints",
        keep_last_n=5
    )
    
    # Create metrics logger
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    metrics_logger = None
    if use_wandb:
        metrics_logger = MetricsLogger(
            project_name=args.wandb_project,
            experiment_name=args.experiment_name,
            config=config
        )
        metrics_logger.log_hyperparameters(config)
    
    # Create model
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        n_experts=model_config['n_experts'],
        expert_types=model_config['expert_types'],
        memory_size=model_config['memory_size'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=model_config['use_memory'],
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    ).to(device)
    
    # Print model info
    params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total: {format_number(params['total'])}")
    print(f"  Trainable: {format_number(params['trainable'])}")
    print(f"  Non-trainable: {format_number(params['non_trainable'])}")
    print(f"  Multimodal: {model_config['use_multimodal']}")
    
    # Create datasets
    train_dataset, val_dataset, is_multimodal = create_dataset(
        data_dir=args.data_dir,
        data_path=args.data_path,
        model_config=model_config,
        train_split=args.train_split
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Multimodal: {is_multimodal}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['learning_rate'],
        betas=(train_config['adam_beta1'], train_config['adam_beta2']),
        eps=train_config['adam_epsilon'],
        weight_decay=train_config['weight_decay']
    )
    
    # Scheduler
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=train_config['warmup_steps'],
        d_model=model_config['d_model']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        print(f"Resumed training from step {start_step}")
    
    # Training loop
    num_epochs = train_config.get('max_steps', 10)
    use_amp = train_config.get('use_amp', False)
    best_val_loss = float('inf')
    
    print(f"\n🔥 AGIFORMER Training Started!")
    print(f"📊 Dataset type: {'Multimodal' if is_multimodal else 'Text-only'}")
    print(f"🚀 Device: {device}")
    print(f"📦 Batch size: {train_config['batch_size']}")
    print(f"🧠 Learning rate: {train_config['learning_rate']}")
    print(f"⚡ Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
    
    # Debug: Test first batch
    print(f"🔍 Testing first batch...")
    try:
        first_batch = next(iter(train_dataloader))
        if is_multimodal:
            print(f"   First batch shapes: input_ids={first_batch['input_ids'].shape}, images={first_batch['image'].shape}")
        else:
            print(f"   First batch shapes: input_ids={first_batch[0].shape}, targets={first_batch[1].shape}")
        print("✅ First batch test successful!")
    except Exception as e:
        print(f"❌ First batch test failed: {e}")
        raise
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    
    try:
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            avg_loss, end_step = train_epoch(
                model, train_dataloader, optimizer, criterion, device, 
                use_amp, metrics_logger, start_step, is_multimodal
            )
            
            # Validation
            val_loss = validate_epoch(model, val_dataloader, criterion, device, use_amp, is_multimodal)
            
            # Update scheduler
            scheduler.step()
            
            # Log validation metrics
            if metrics_logger:
                metrics_logger.log_validation_metrics(end_step, val_loss, {
                    'Validation/perplexity': torch.exp(torch.tensor(val_loss)).item()
                })
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            checkpoint_manager.save_checkpoint(
                step=end_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={'train_loss': avg_loss, 'val_loss': val_loss},
                is_best=is_best
            )
            
            print(f"Epoch {epoch + 1} completed:")
            print(f"  Training Loss: {avg_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            start_step = end_step
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if metrics_logger:
            metrics_logger.finish()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
