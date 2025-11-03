#!/usr/bin/env python3
"""
Test script to verify the n_heads fix for both configurations
"""

import torch
import yaml
from agiformer import AGIFORMER

def test_configuration(config_path, config_name):
    """Test a specific configuration"""
    print(f"\n=== Testing {config_name} Configuration ===")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    print(f"d_model: {model_config['d_model']}")
    print(f"n_heads: {model_config['n_heads']}")
    print(f"d_model % n_heads = {model_config['d_model'] % model_config['n_heads']}")
    
    try:
        # Create model
        model = AGIFORMER(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            d_ff=model_config['d_ff'],
            n_experts=model_config['n_experts'],
            use_multimodal=model_config['use_multimodal'],
            dropout=model_config['dropout']
        )
        
        print(f"✅ Model created successfully!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with text input
        batch_size, seq_len = 2, 32
        text_input = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        
        with torch.no_grad():
            output, info = model(text=text_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Model info keys: {list(info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test both configurations"""
    print("Testing AGIFORMER n_heads fix...")
    
    configs = [
        ("configs/base_config.yaml", "Base"),
        ("configs/colab_config.yaml", "Colab")
    ]
    
    results = []
    for config_path, config_name in configs:
        success = test_configuration(config_path, config_name)
        results.append((config_name, success))
    
    print("\n" + "="*50)
    print("SUMMARY:")
    for config_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {config_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All configurations passed! The fix is working correctly.")
    else:
        print("\n⚠️  Some configurations failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
