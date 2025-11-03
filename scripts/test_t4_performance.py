#!/usr/bin/env python3
"""
T4 GPU Performance Test Script for AGIFORMER
Tests model loading, memory usage, and training speed
"""

import torch
import time
import psutil
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from agiformer.utils import count_parameters, format_number


def get_gpu_info():
    """Get GPU information"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
        "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
    }


def test_model_loading(config_path: str):
    """Test model loading performance"""
    print(f"🔄 Testing model loading with config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # GPU info before loading
    gpu_before = get_gpu_info()
    print(f"📊 GPU before loading: {gpu_before['name'] if gpu_before['available'] else 'N/A'}")
    if gpu_before['available']:
        print(f"   Memory: {gpu_before['memory_allocated_gb']:.2f}GB / {gpu_before['total_memory_gb']:.2f}GB")
    
    # Time model creation
    start_time = time.time()
    
    try:
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
        )
        
        creation_time = time.time() - start_time
        
        # Move to GPU
        gpu_start = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        gpu_time = time.time() - gpu_start
        
        # GPU info after loading
        gpu_after = get_gpu_info()
        
        # Model parameters
        params = count_parameters(model)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Creation time: {creation_time:.2f}s")
        print(f"   GPU transfer time: {gpu_time:.2f}s")
        print(f"   Total parameters: {format_number(params['total'])}")
        print(f"   Trainable parameters: {format_number(params['trainable'])}")
        
        if gpu_after['available']:
            print(f"   Memory after loading: {gpu_after['memory_allocated_gb']:.2f}GB / {gpu_after['total_memory_gb']:.2f}GB")
            print(f"   Memory increase: {gpu_after['memory_allocated_gb'] - gpu_before['memory_allocated_gb']:.2f}GB")
        
        return model, True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, False


def test_forward_pass(model, config_path: str):
    """Test forward pass performance"""
    print(f"\n🚀 Testing forward pass performance...")
    
    # Load config for batch size
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    train_config = config['training']
    
    device = next(model.parameters()).device
    batch_size = train_config['batch_size']
    seq_len = model_config['max_seq_len']
    
    # Create dummy input
    input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)).to(device)
    
    # GPU info before forward
    gpu_before = get_gpu_info()
    
    # Time forward pass
    model.eval()
    start_time = time.time()
    
    try:
        with torch.no_grad():
            logits, info = model(text=input_ids)
        
        forward_time = time.time() - start_time
        
        # GPU info after forward
        gpu_after = get_gpu_info()
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Output shape: {logits.shape}")
        print(f"   Forward time: {forward_time:.3f}s")
        print(f"   Tokens/sec: {(batch_size * seq_len) / forward_time:.0f}")
        
        if gpu_after['available']:
            print(f"   Peak memory: {gpu_after['memory_allocated_gb']:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False


def test_training_step(model, config_path: str):
    """Test training step performance"""
    print(f"\n🏋️ Testing training step performance...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    train_config = config['training']
    
    device = next(model.parameters()).device
    batch_size = train_config['batch_size']
    seq_len = model_config['max_seq_len']
    
    # Create dummy data
    input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)).to(device)
    target_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len)).to(device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    
    # GPU info before training
    gpu_before = get_gpu_info()
    
    # Time training step
    start_time = time.time()
    
    try:
        optimizer.zero_grad()
        
        # Forward pass
        logits, info = model(text=input_ids)
        
        # Calculate loss
        logits = logits.view(-1, logits.size(-1))
        target_ids = target_ids.view(-1)
        loss = criterion(logits, target_ids)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        training_time = time.time() - start_time
        
        # GPU info after training
        gpu_after = get_gpu_info()
        
        print(f"✅ Training step successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Training time: {training_time:.3f}s")
        print(f"   Tokens/sec: {(batch_size * seq_len) / training_time:.0f}")
        
        if gpu_after['available']:
            print(f"   Memory after training: {gpu_after['memory_allocated_gb']:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False


def main():
    """Main performance test function"""
    print("🔥 AGIFORMER T4 GPU Performance Test")
    print("=" * 50)
    
    # System info
    print(f"🖥️  System Info:")
    print(f"   CPU: {psutil.cpu_count()} cores")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    
    gpu_info = get_gpu_info()
    if gpu_info['available']:
        print(f"   GPU: {gpu_info['name']}")
        print(f"   GPU Memory: {gpu_info['total_memory_gb']:.1f}GB")
    else:
        print(f"   GPU: Not available")
        return
    
    print("\n" + "=" * 50)
    
    # Test configurations
    configs = [
        "configs/base_config.yaml",
        "configs/t4_optimized_config.yaml"
    ]
    
    for config_path in configs:
        if not Path(config_path).exists():
            print(f"⚠️  Config not found: {config_path}")
            continue
        
        print(f"\n📋 Testing configuration: {config_path}")
        print("-" * 40)
        
        # Test model loading
        model, success = test_model_loading(config_path)
        if not success:
            continue
        
        # Test forward pass
        if not test_forward_pass(model, config_path):
            continue
        
        # Test training step
        if not test_training_step(model, config_path):
            continue
        
        print(f"\n✅ All tests passed for {config_path}!")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    print(f"\n🎯 Performance test completed!")
    print(f"💡 Recommendations:")
    if gpu_info['available']:
        if gpu_info['total_memory_gb'] < 8:
            print(f"   - Use t4_optimized_config.yaml for better performance")
            print(f"   - Consider reducing batch size further")
        elif gpu_info['total_memory_gb'] >= 16:
            print(f"   - You can use base_config.yaml for better quality")
            print(f"   - Consider increasing batch size for faster training")


if __name__ == "__main__":
    main()
