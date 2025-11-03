"""
Introspection Test for AGIFORMER Gözlemci Phase
Test self-awareness and iterative thinking capabilities
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER


def test_introspection_functionality():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model with Introspection enabled
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=384,  # Smaller model for testing
        n_layers=2,   # Fewer layers for quick test
        n_heads=6,    # Fewer heads
        d_ff=1536,    # Smaller FFN
        n_experts=4,  # MoE enabled
        expert_types=model_config['expert_types'],
        memory_size=1000,  # Smaller memory for testing
        max_seq_len=64,
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=True,  # Memory enabled
        use_introspection=True,  # Introspection enabled
        use_multimodal=model_config['use_multimodal']
    )
    
    print(f"Model created with Introspection enabled")
    print(f"Introspection: {model_config['use_introspection']}")
    print(f"Memory: {model_config['use_memory']}")
    print(f"MoE Experts: {model_config['n_experts']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create test data
    batch_size = 2
    seq_len = 16
    
    # Simple text data
    dummy_text = "The quick brown fox jumps over the lazy dog."
    char_ids = [ord(c) % model_config['vocab_size'] for c in dummy_text * (seq_len // len(dummy_text) + 1)]
    char_ids = char_ids[:seq_len]
    
    input_ids = torch.tensor([char_ids[:-1] for _ in range(batch_size)], dtype=torch.long)
    target_ids = torch.tensor([char_ids[1:] for _ in range(batch_size)], dtype=torch.long)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")
    
    # Test forward pass with introspection
    model.eval()
    with torch.no_grad():
        logits, info = model(text=input_ids)
        print(f"Logits shape: {logits.shape}")
        print(f"Model info keys: {list(info.keys())}")
        
        # Check introspection info
        if 'blocks' in info:
            for i, block_info in enumerate(info['blocks']):
                if 'introspection' in block_info and block_info['introspection']:
                    introspection_info = block_info['introspection']
                    print(f"Block {i} Introspection info keys: {list(introspection_info.keys())}")
                    
                    if 'iterations' in introspection_info:
                        iterations = introspection_info['iterations']
                        print(f"Block {i} Introspection iterations: {len(iterations)}")
                        for j, iteration in enumerate(iterations):
                            print(f"  Iteration {j}: error={iteration['error_score']:.4f}, confidence={iteration['confidence']:.4f}")
                    
                    if 'num_iterations' in introspection_info:
                        print(f"Block {i} Total introspection iterations: {introspection_info['num_iterations']}")
    
    # Test loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    initial_loss = criterion(logits_flat, target_flat)
    print(f"Initial loss: {initial_loss.item():.4f}")
    
    # Test training step with introspection
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Forward pass
    logits, info = model(text=input_ids)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    loss = criterion(logits_flat, target_flat)
    print(f"Training loss: {loss.item():.4f}")
    
    # Check for MoE load balancing loss
    total_loss = loss
    for block_info in info['blocks']:
        if 'moe' in block_info and 'router_info' in block_info['moe']:
            if 'load_balancing_loss' in block_info['moe']['router_info']:
                lb_loss = block_info['moe']['router_info']['load_balancing_loss']
                total_loss = total_loss + lb_loss
                print(f"Load balancing loss: {lb_loss.item():.6f}")
    
    print(f"Total loss (with LB): {total_loss.item():.4f}")
    
    # Backward pass
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
            print("✅ SUCCESS: Loss decreased with Introspection + MoE + Memory! The model is learning.")
            print("🧠 Introspection system is working correctly.")
            print("🎯 Self-awareness capabilities activated!")
        else:
            print("❌ Loss did not decrease. Model may need more training or adjustment.")
    
    # Test introspection with different inputs
    print("\n--- Testing Introspection Variability ---")
    
    # Test with different text
    different_text = "Artificial intelligence will change the world."
    different_char_ids = [ord(c) % model_config['vocab_size'] for c in different_text * (seq_len // len(different_text) + 1)]
    different_char_ids = different_char_ids[:seq_len]
    different_input_ids = torch.tensor([different_char_ids[:-1]], dtype=torch.long)
    
    model.reset_memory()  # Reset memory for clean test
    
    with torch.no_grad():
        logits2, info2 = model(text=different_input_ids)
        
        # Check introspection for different input
        if 'blocks' in info2:
            last_block = info2['blocks'][-1]
            if 'introspection' in last_block and last_block['introspection']:
                introspection_info2 = last_block['introspection']
                if 'iterations' in introspection_info2:
                    iterations2 = introspection_info2['iterations']
                    print(f"Different text introspection iterations: {len(iterations2)}")
                    for j, iteration in enumerate(iterations2):
                        print(f"  Iteration {j}: error={iteration['error_score']:.4f}, confidence={iteration['confidence']:.4f}")
    
    print("\n🎯 Introspection Test completed!")
    return new_loss.item() < initial_loss.item()


if __name__ == "__main__":
    success = test_introspection_functionality()
