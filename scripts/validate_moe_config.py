## Developer: inkbytefo
## Modified: 2025-11-07
"""
Quick validation script to test MoE configuration fixes
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer.experts.moe import MixtureOfExperts, ExpertRouter
from agiformer.model import AGIFORMERBlock

def test_moe_bounds_checking():
    """Test that MoE doesn't cause out-of-bounds errors"""
    print("Testing MoE bounds checking...")
    
    # Test case 1: Normal case (4 experts, k=2)
    batch_size, seq_len, d_model = 2, 10, 256
    n_experts, k = 4, 2
    
    # Create test data
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    
    # Test ExpertRouter
    router = ExpertRouter(d_model, n_experts, k)
    expert_weights, expert_indices, router_info = router(hidden_states)
    
    print(f"✓ ExpertRouter test passed")
    print(f"  - expert_indices shape: {expert_indices.shape}")
    print(f"  - expert_indices max: {expert_indices.max().item()}")
    print(f"  - expert_indices min: {expert_indices.min().item()}")
    
    # Verify indices are within bounds
    assert expert_indices.max() < n_experts, f"Index out of bounds: max={expert_indices.max()}, n_experts={n_experts}"
    assert expert_indices.min() >= 0, f"Negative index: min={expert_indices.min()}"
    
    # Test MixtureOfExperts
    moe = MixtureOfExperts(d_model, n_experts, k=k)
    output, moe_info = moe(hidden_states)
    
    print(f"✓ MixtureOfExperts test passed")
    print(f"  - output shape: {output.shape}")
    print(f"  - valid_indices_ratio: {moe_info['valid_indices_ratio']}")
    
    # Test edge case 2: Single expert (1 expert, k=2 but should be clamped to 1)
    print("\nTesting edge case: single expert...")
    n_experts, k = 1, 2
    moe_single = MixtureOfExperts(d_model, n_experts, k=k)
    
    # The k should be automatically clamped
    assert moe_single.k == 1, f"k should be clamped to 1, got {moe_single.k}"
    
    output_single, info_single = moe_single(hidden_states)
    print(f"✓ Single expert case handled correctly")
    print(f"  - effective k: {moe_single.k}")

def test_agiformer_block():
    """Test AGIFORMER block with different expert configurations"""
    print("\nTesting AGIFORMER block...")
    
    d_model, n_heads, d_ff = 256, 4, 1024
    
    # Test with 4 experts
    n_experts = 4
    expert_types = ["language", "logic", "spatial", "causal"]
    
    block = AGIFORMERBlock(
        d_model, n_heads, d_ff, n_experts, expert_types, 
        use_agglutinative_attention=True
    )
    
    # Test forward pass
    x = torch.randn(2, 10, d_model)
    output, info = block(x)
    
    print(f"✓ AGIFORMER block test passed")
    print(f"  - output shape: {output.shape}")
    print(f"  - n_experts: {block.n_experts}")
    print(f"  - k: {block.k}")
    print(f"  - routing_bias shape: {info['moe']['router_info']['expert_usage']}")

if __name__ == "__main__":
    print("=== Testing MoE Configuration Fixes ===\n")
    
    try:
        test_moe_bounds_checking()
        test_agiformer_block()
        
        print("\n🎉 All tests passed! The CUDA error should be fixed.")
        print("\nKey fixes applied:")
        print("1. Added bounds checking in MoE gather operations")
        print("2. Ensured k <= n_experts in all components")
        print("3. Added proper clamping for expert indices")
        print("4. Fixed routing bias tensor sizing")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)