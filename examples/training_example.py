"""
Training Example for AGIFORMER
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import sys

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


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
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
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    train_config = config['training']
    
    # Device
    device = torch.device(train_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Create dummy dataset
    dummy_texts = [
        "This is a sample text for training.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
    ] * 1000  # Repeat for more data
    
    dataset = SimpleTextDataset(
        dummy_texts,
        max_seq_len=model_config['max_seq_len'],
        vocab_size=model_config['vocab_size']
    )
    
    # Use 0 workers to avoid multiprocessing issues
    dataloader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
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
    
    # Training loop
    num_epochs = 3  # Reduced for quick testing
    use_amp = train_config.get('use_amp', False)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, use_amp)
        scheduler.step()
        
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("\nTraining completed!")
    
    # Save model
    model_path = Path(__file__).parent.parent / "checkpoints" / "agiformer_model.pt"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
