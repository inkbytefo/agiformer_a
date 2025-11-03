"""
Professional Training Script for AGIFORMER
Features: Checkpointing, W&B logging, Resume capability, Validation, Early Stopping
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import sys
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from agiformer.utils import count_parameters, format_number, WarmupScheduler


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


def train_step(
    model, 
    batch, 
    optimizer, 
    criterion, 
    device, 
    use_amp=False,
    scaler=None
) -> Tuple[torch.Tensor, Dict]:
    """Train for one step"""
    input_ids, target_ids = batch
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    optimizer.zero_grad()
    
    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            logits, info = model(text=input_ids)
            
            # Reshape for loss
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = criterion(logits, target_ids)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, info = model(text=input_ids)
        
        # Reshape for loss
        logits = logits.view(-1, logits.size(-1))
        target_ids = target_ids.view(-1)
        
        loss = criterion(logits, target_ids)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    return loss, info


def validate(model, dataloader, criterion, device) -> Tuple[float, Dict]:
    """Validation step"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_info = []
    
    with torch.no_grad():
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, info = model(text=input_ids)
            
            # Reshape for loss
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = criterion(logits, target_ids)
            total_loss += loss.item()
            num_batches += 1
            all_info.append(info)
    
    # Aggregate info
    avg_loss = total_loss / num_batches
    
    # Extract metrics from info
    metrics = {
        'val_loss': avg_loss,
        'val_perplexity': torch.exp(torch.tensor(avg_loss)).item()
    }
    
    # Average MoE metrics if available
    if all_info and 'blocks' in all_info[0]:
        moe_losses = []
        for info in all_info:
            for block_info in info['blocks']:
                if 'moe' in block_info and 'router_info' in block_info['moe']:
                    if 'load_balancing_loss' in block_info['moe']['router_info']:
                        moe_losses.append(block_info['moe']['router_info']['load_balancing_loss'].item())
        
        if moe_losses:
            metrics['val_moe_loss'] = sum(moe_losses) / len(moe_losses)
    
    return avg_loss, metrics


def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    epoch: int, 
    step: int, 
    best_loss: float,
    save_path: Path,
    config: Dict,
    extra_info: Dict = None
):
    """Save training checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'best_loss': best_loss,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'extra_info': extra_info or {}
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: Path, 
    model, 
    optimizer=None, 
    scheduler=None
) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'best_loss': checkpoint['best_loss'],
        'config': checkpoint['config'],
        'extra_info': checkpoint.get('extra_info', {})
    }


def setup_wandb(config: Dict, resume_id: Optional[str] = None) -> Optional[object]:
    """Setup Weights & Biases logging"""
    try:
        import wandb
        
        run_name = f"agiformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project="agiformer",
            name=run_name,
            config=config,
            resume=resume_id,
            settings=wandb.Settings(_disable_stats=True)
        )
        
        print("✅ W&B initialized successfully")
        return wandb
    except ImportError:
        print("⚠️  W&B not available. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"⚠️  W&B initialization failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train AGIFORMER model")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Config file path")
    parser.add_argument("--data", type=str, default="dummy", help="Data type: dummy, cc12m")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--val_every", type=int, default=100, help="Validate every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_resume_id", type=str, default=None, help="W&B resume ID")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    config['training'].update({
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'save_every': args.save_every,
        'val_every': args.val_every,
        'early_stopping_patience': args.early_stopping_patience,
        'max_steps': args.max_steps,
        'use_amp': args.use_amp
    })
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup W&B
    wandb = None
    if args.wandb:
        wandb = setup_wandb(config, args.wandb_resume_id)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create model
    model_config = config['model']
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
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(Path(args.resume), model)
        start_epoch = checkpoint_info['epoch']
        start_step = checkpoint_info['step']
        best_loss = checkpoint_info['best_loss']
        print(f"Resumed from epoch {start_epoch}, step {start_step}")
    
    # Create dataset (for now use dummy data)
    if args.data == "dummy":
        dummy_texts = [
            "This is a sample text for training.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating.",
            "Artificial intelligence is transforming our world.",
            "Deep learning models require large datasets.",
            "Neural networks can learn complex patterns.",
            "Training deep models requires careful optimization.",
            "Gradient descent helps models learn from data.",
            "Backpropagation computes gradients efficiently.",
            "Transformers revolutionized natural language processing."
        ] * 100  # Repeat for more data
    else:
        raise NotImplementedError(f"Data type {args.data} not implemented yet")
    
    # Split into train/val
    train_texts = dummy_texts[:-len(dummy_texts)//10]
    val_texts = dummy_texts[-len(dummy_texts)//10:]
    
    train_dataset = SimpleTextDataset(
        train_texts,
        max_seq_len=model_config['max_seq_len'],
        vocab_size=model_config['vocab_size']
    )
    
    val_dataset = SimpleTextDataset(
        val_texts,
        max_seq_len=model_config['max_seq_len'],
        vocab_size=model_config['vocab_size']
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_epsilon'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=config['training']['warmup_steps'],
        d_model=model_config['d_model']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training variables
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    patience_counter = 0
    
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Save every: {args.save_every} steps")
    print(f"  Validate every: {args.val_every} steps")
    
    # Training loop
    model.train()
    total_steps = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            step = start_step + total_steps + 1
            
            # Training step
            loss, info = train_step(model, batch, optimizer, criterion, device, args.use_amp, scaler)
            
            # Log training metrics
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")
                
                # W&B logging
                if wandb:
                    log_dict = {
                        'train_loss': loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'step': step
                    }
                    
                    # Add MoE metrics if available
                    if 'blocks' in info:
                        for i, block_info in enumerate(info['blocks']):
                            if 'moe' in block_info and 'router_info' in block_info['moe']:
                                if 'load_balancing_loss' in block_info['moe']['router_info']:
                                    log_dict[f'block_{i}_moe_loss'] = block_info['moe']['router_info']['load_balancing_loss'].item()
                    
                    wandb.log(log_dict)
            
            # Validation
            if step % args.val_every == 0:
                val_loss, val_metrics = validate(model, val_dataloader, criterion, device)
                print(f"Step {step} - Validation Loss: {val_loss:.4f}")
                
                # W&B validation logging
                if wandb:
                    wandb.log({**val_metrics, 'step': step})
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    best_model_path = save_dir / "best_model.pt"
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, step, best_loss,
                        best_model_path, config, {'val_loss': val_loss}
                    )
                    print(f"New best model saved! Loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.early_stopping_patience:
                        print(f"Early stopping triggered after {patience_counter} patience steps")
                        break
            
            # Save checkpoint
            if step % args.save_every == 0:
                checkpoint_path = save_dir / f"checkpoint_step_{step}.pt"
                save_checkpoint(
                    model, optimizer, scheduler, epoch, step, best_loss,
                    checkpoint_path, config
                )
            
            total_steps += 1
            
            # Max steps check
            if args.max_steps and step >= args.max_steps:
                print(f"Reached max steps: {args.max_steps}")
                break
        
        scheduler.step()
        
        if patience_counter >= args.early_stopping_patience:
            print("Training stopped early due to lack of improvement")
            break
    
    # Final save
    final_path = save_dir / "final_model.pt"
    save_checkpoint(
        model, optimizer, scheduler, epoch, total_steps, best_loss,
        final_path, config
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Final model saved: {final_path}")
    
    # Close W&B
    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
