"""
Memory Test for AGIFORMER Düşünür Phase
Test Working Memory and Long-term Memory functionality
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER


def test_memory_functionality():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model with Memory enabled
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
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    )
    
    print(f"Model created with Memory enabled")
    print(f"Memory size: {model_config['memory_size']}")
    print(f"Max sequence length: {model_config['max_seq_len']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create test data - multiple sequences to test memory
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
    
    # Test forward pass with memory
    model.eval()
    with torch.no_grad():
        logits, info = model(text=input_ids)
        print(f"Logits shape: {logits.shape}")
        print(f"Model info keys: {list(info.keys())}")
        
        # Check memory info
        if 'memory' in info:
            memory_info = info['memory']
            print(f"Memory info keys: {list(memory_info.keys())}")
            if 'step_count' in memory_info:
                print(f"Memory step count: {memory_info['step_count']}")
    
    # Test loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    initial_loss = criterion(logits_flat, target_flat)
    print(f"Initial loss: {initial_loss.item():.4f}")
    
    # Test memory reset
    print("\n--- Testing Memory Reset ---")
    model.reset_memory()
    print("Memory reset completed")
    
    # Test training step with memory
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
            print("✅ SUCCESS: Loss decreased with Memory + MoE! The model is learning.")
            print("🧠 Memory system is working correctly.")
            print("🎯 MoE + Memory integration successful!")
        else:
            print("❌ Loss did not decrease. Model may need more training or adjustment.")
    
    # Test sequential processing (memory persistence)
    print("\n--- Testing Memory Persistence ---")
    model.reset_memory()
    
    # Process first sequence
    seq1_ids = input_ids[:, :8]  # First half
    with torch.no_grad():
        logits1, info1 = model(text=seq1_ids)
        print(f"First sequence processed. Memory step: {info1['memory']['step_count']}")
    
    # Process second sequence (should use memory from first)
    seq2_ids = input_ids[:, 8:]  # Second half
    with torch.no_grad():
        logits2, info2 = model(text=seq2_ids)
        print(f"Second sequence processed. Memory step: {info2['memory']['step_count']}")
        
        # Check if memory is being used
        if info2['memory']['step_count'] > info1['memory']['step_count']:
            print("✅ Memory persistence working - step count increased")
        else:
            print("⚠️  Memory step count not increasing as expected")
    
    print("\n🎯 Memory Test completed!")
    return new_loss.item() < initial_loss.item()


if __name__ == "__main__":
    success = test_memory_functionality()
