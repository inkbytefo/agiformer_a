"""
Quick Test for AGIFORMER Kıvılcım Phase
Fast test to verify loss decreases
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER


def quick_test():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create small model for quick test
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=384,  # Smaller model
        n_layers=4,   # Fewer layers
        n_heads=6,    # Fewer heads
        d_ff=1536,    # Smaller FFN
        n_experts=model_config['n_experts'],  # Use config value (4 experts)
        expert_types=model_config['expert_types'],
        memory_size=model_config['memory_size'],
        max_seq_len=128,  # Shorter sequences
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=model_config['use_memory'],
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    )
    
    # Create dummy data
    batch_size = 4
    seq_len = 64
    
    # Simple text data
    dummy_text = "Hello world! This is a test."
    char_ids = [ord(c) % model_config['vocab_size'] for c in dummy_text * (seq_len // len(dummy_text) + 1)]
    char_ids = char_ids[:seq_len]  # Truncate to seq_len
    
    input_ids = torch.tensor([char_ids[:-1]], dtype=torch.long)  # Remove last char for input
    target_ids = torch.tensor([char_ids[1:]], dtype=torch.long)  # Remove first char for target
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits, info = model(text=input_ids)
        print(f"Logits shape: {logits.shape}")
        print(f"Model info: {info}")
    
    # Test loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    initial_loss = criterion(logits_flat, target_flat)
    print(f"Initial loss: {initial_loss.item():.4f}")
    
    # Test training step
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Forward pass
    logits, info = model(text=input_ids)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    loss = criterion(logits_flat, target_flat)
    print(f"Training loss: {loss.item():.4f}")
    
    # Add MoE load balancing loss
    total_loss = loss
    for block_info in info.get('blocks', []):
        if 'moe' in block_info and 'load_balancing_loss' in block_info['moe']['router_info']:
            lb_loss = block_info['moe']['router_info']['load_balancing_loss']
            total_loss = total_loss + lb_loss
            print(f"Load balancing loss: {lb_loss.item():.6f}")
            print(f"Total loss (with MoE): {total_loss.item():.4f}")
    
    # Backward pass with total loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Test second forward pass
    model.eval()
    with torch.no_grad():
        logits, info = model(text=input_ids)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_ids.view(-1)
        
        new_loss = criterion(logits_flat, target_flat)
        print(f"Loss after one step: {new_loss.item():.4f}")
        
        if new_loss.item() < initial_loss.item():
            print("SUCCESS: Loss decreased! The model is learning.")
        else:
            print("❌ Loss did not decrease. Model may need more training or adjustment.")
    
    print("\n🎉 Quick test completed!")
    return new_loss.item() < initial_loss.item()


if __name__ == "__main__":
    success = quick_test()
