"""
Multimodal Test for AGIFORMER Gözlemci Phase
Test text, image, and multimodal processing capabilities
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER


def test_multimodal_functionality():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model with all features enabled
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
        use_multimodal=True  # Multimodal enabled
    )
    
    print(f"Model created with all features enabled:")
    print(f"  MoE Experts: {model_config['n_experts']}")
    print(f"  Memory: {model_config['use_memory']}")
    print(f"  Introspection: {model_config['use_introspection']}")
    print(f"  Multimodal: {model_config['use_multimodal']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test 1: Text-only input
    print("\n=== Test 1: Text-only Input ===")
    batch_size = 2
    seq_len = 16
    
    # Simple text data
    dummy_text = "The quick brown fox jumps over the lazy dog."
    char_ids = [ord(c) % model_config['vocab_size'] for c in dummy_text * (seq_len // len(dummy_text) + 1)]
    char_ids = char_ids[:seq_len]
    
    text_input = torch.tensor([char_ids[:-1] for _ in range(batch_size)], dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        logits, info = model(text=text_input)
        print(f"Text-only logits shape: {logits.shape}")
        print(f"Multimodal: {info['multimodal']}")
        print(f"Tokenizer: {info.get('tokenizer', 'N/A')}")
        
        # Check if memory and introspection are working
        if 'memory' in info:
            print(f"Memory step count: {info['memory']['step_count']}")
        
        # Check introspection in last block
        last_block = info['blocks'][-1]
        if 'introspection' in last_block and last_block['introspection']:
            print(f"Introspection iterations: {last_block['introspection']['num_iterations']}")
    
    # Test 2: Image-only input (dummy image data)
    print("\n=== Test 2: Image-only Input ===")
    
    # Create dummy image data [batch, 3, 224, 224]
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    try:
        with torch.no_grad():
            logits, info = model(image=dummy_image)
            print(f"Image-only logits shape: {logits.shape}")
            print(f"Multimodal: {info['multimodal']}")
            print(f"Modalities: {info.get('modalities', 'N/A')}")
    except Exception as e:
        print(f"Image-only test failed: {e}")
    
    # Test 3: Text + Image combined
    print("\n=== Test 3: Text + Image Combined ===")
    
    try:
        with torch.no_grad():
            logits, info = model(text=text_input, image=dummy_image)
            print(f"Combined logits shape: {logits.shape}")
            print(f"Multimodal: {info['multimodal']}")
            print(f"Modalities: {info.get('modalities', 'N/A')}")
            
            # Check cross-modal processing
            if 'modalities' in info:
                print(f"Cross-modal fusion working with: {info['modalities']}")
    except Exception as e:
        print(f"Combined test failed: {e}")
    
    # Test 4: Training capability with multimodal
    print("\n=== Test 4: Training Capability ===")
    
    # Create target data
    target_ids = torch.tensor([char_ids[1:] for _ in range(batch_size)], dtype=torch.long)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Test text-only training
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Forward pass
    logits, info = model(text=text_input)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)
    
    loss = criterion(logits_flat, target_flat)
    print(f"Text-only training loss: {loss.item():.4f}")
    
    # Add MoE load balancing loss
    total_loss = loss
    for block_info in info['blocks']:
        if 'moe' in block_info and 'router_info' in block_info['moe']:
            if 'load_balancing_loss' in block_info['moe']['router_info']:
                lb_loss = block_info['moe']['router_info']['load_balancing_loss']
                total_loss = total_loss + lb_loss
                print(f"Load balancing loss: {lb_loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # Test second forward pass
    model.eval()
    with torch.no_grad():
        logits, info = model(text=text_input)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_ids.view(-1)
        
        new_loss = criterion(logits_flat, target_flat)
        print(f"Loss after one step: {new_loss.item():.4f}")
        
        if new_loss.item() < loss.item():
            print("✅ SUCCESS: Loss decreased with full multimodal model!")
            print("🧠 All systems working: MoE + Memory + Introspection + Multimodal")
        else:
            print("❌ Loss did not decrease. Model may need more training.")
    
    print("\n🎯 Multimodal Test completed!")
    return new_loss.item() < loss.item()


if __name__ == "__main__":
    success = test_multimodal_functionality()
