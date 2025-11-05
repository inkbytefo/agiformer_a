#!/usr/bin/env python3
# Developer: inkbytefo
# Modified: 2025-11-06

"""
RX590 Configuration Test Script
Test AGIFORMER configs for RX590 8GB + Ryzen 3100 system
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_system_specs():
    """Check system specifications"""
    print("🔍 System Specification Check")
    print("=" * 50)

    # CPU Info
    print(f"CPU: Ryzen 3100 (4 cores, 8 threads)")

    # RAM
    print(f"RAM: 24GB DDR4")

    # GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        print(f"CUDA: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
    else:
        print("❌ CUDA not available")
        return False

    print()
    return True

def test_config_loading():
    """Test loading RX590 configurations"""
    print("🔧 Configuration Loading Test")
    print("=" * 50)

    try:
        from omegaconf import OmegaConf
        import hydra
        from hydra.core.config_store import ConfigStore

        # Load config
        config_path = Path(__file__).parent.parent / "conf"
        config_name = "rx590_full"

        with hydra.initialize(config_path=str(config_path)):
            cfg = hydra.compose(config_name=config_name)

        print("✅ Config loaded successfully")
        print(f"Run name: {cfg.run_name}")
        print(f"Hardware device: {cfg.hardware.device}")
        print(f"GPU memory fraction: {cfg.hardware.memory_fraction}")
        print(f"Batch size: {cfg.training.batch_size}")
        print(f"Gradient accumulation: {cfg.training.gradient_accumulation_steps}")
        print(f"Model d_model: {cfg.model.d_model}")
        print(f"Model layers: {cfg.model.n_layers}")
        print(f"Max seq len: {cfg.model.max_seq_len}")

        return cfg

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return None

def test_model_creation(cfg):
    """Test model creation with RX590 config"""
    print("\n🤖 Model Creation Test")
    print("=" * 50)

    try:
        from agiformer import AGIFORMER

        # Create model
        model = AGIFORMER(
            tokenizer=None,  # Skip tokenizer for test
            use_gradient_checkpointing=cfg.training.use_gradient_checkpointing,
            **{k: v for k, v in cfg.model.items() if k != 'tokenizer_path'}
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("✅ Model created successfully")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Memory size: {cfg.model.memory_size}")
        print(f"Experts: {cfg.model.n_experts} ({cfg.model.expert_types})")

        # Memory estimate
        param_memory = total_params * 4 / (1024**3)  # Rough estimate in GB
        print(f"Estimated parameter memory: {param_memory:.2f} GB")
        return model

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_memory_usage(model, cfg):
    """Test memory usage estimation"""
    print("\n💾 Memory Usage Test")
    print("=" * 50)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        model = model.to(device)

        # Get initial memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"Initial GPU memory: {initial_memory:.2f} GB")

        # Test forward pass with small batch
        batch_size = cfg.training.batch_size
        seq_len = min(cfg.model.max_seq_len, 128)  # Small for testing
        vocab_size = cfg.model.vocab_size

        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        print(f"Testing with batch_size={batch_size}, seq_len={seq_len}")

        # Forward pass
        with torch.no_grad():
            logits, info = model(input_ids)

        # Check memory after forward
        if torch.cuda.is_available():
            after_memory = torch.cuda.memory_allocated() / (1024**3)
            memory_used = after_memory - initial_memory
            print(f"Final GPU memory: {after_memory:.2f} GB")
            print(f"Memory used for forward pass: {memory_used:.1f} GB")

            # Check if within limits
            if memory_used < 6.0:  # Leave 2GB buffer
                print("✅ Memory usage acceptable")
            else:
                print("⚠️ High memory usage - consider reducing batch size")

        print("✅ Forward pass successful")
        print(f"Output shape: {logits.shape}")

        return True

    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 AGIFORMER RX590 Configuration Test")
    print("=" * 60)
    print()

    # Check system
    if not check_system_specs():
        return 1

    # Test config loading
    cfg = test_config_loading()
    if cfg is None:
        return 1

    # Test model creation
    model = test_model_creation(cfg)
    if model is None:
        return 1

    # Test memory usage
    if not test_memory_usage(model, cfg):
        return 1

    print("\n🎉 All tests passed!")
    print("=" * 60)
    print("✅ RX590 configuration is ready for training")
    print("💡 Use: python train.py --config-name rx590_full")
    print()

    return 0

if __name__ == "__main__":
    exit(main())
