"""
Create a small test model for production setup demonstration
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER


def create_test_model():
    """Create a small test model and save checkpoint"""
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create small test model
    model = AGIFORMER(
        vocab_size=256,
        d_model=128,  # Small model
        n_layers=2,   # Few layers
        n_heads=4,    # Few heads
        d_ff=512,     # Small FFN
        n_experts=2,  # Few experts
        expert_types=["language", "logic"],
        memory_size=100,  # Small memory
        max_seq_len=32,   # Short sequences
        dropout=0.1,
        use_linear_attention=False,
        use_memory=True,
        use_introspection=True,
        use_multimodal=True
    )
    
    # Create dummy optimizer and scheduler for checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    
    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': 0,
        'step': 0,
        'metrics': {'loss': 5.0},
        'config': config,
        'timestamp': '2024-11-02T17:30:00'
    }
    
    # Save checkpoint
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / "test_model_v01.pt"
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Test model created: {checkpoint_path}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return str(checkpoint_path)


if __name__ == "__main__":
    checkpoint_path = create_test_model()
    print(f"Test model ready for production setup: {checkpoint_path}")
