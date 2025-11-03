"""
Tests for AGIFORMER model
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from agiformer.core import MultimodalPerceptionCore, MorfoSemanticTokenizer, UnifiedMemoryBackbone
from agiformer.experts import MixtureOfExperts


def test_model_creation():
    """Test model creation"""
    model = AGIFORMER(
        vocab_size=256,
        d_model=128,  # Smaller for testing
        n_layers=2,
        n_heads=4,
        d_ff=512,
        n_experts=2
    )
    
    assert model is not None
    assert model.vocab_size == 256
    assert model.d_model == 128


def test_forward_pass():
    """Test forward pass"""
    model = AGIFORMER(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        n_experts=2,
        max_seq_len=128
    )
    
    batch_size = 2
    seq_len = 32
    
    # Text input
    text_input = torch.randint(0, 256, (batch_size, seq_len))
    
    logits, info = model(text=text_input)
    
    assert logits.shape == (batch_size, seq_len, 256)
    assert isinstance(info, dict)


def test_multimodal():
    """Test multimodal input"""
    model = AGIFORMER(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        use_multimodal=True
    )
    
    batch_size = 2
    seq_len = 32
    
    text_input = torch.randint(0, 256, (batch_size, seq_len))
    image_input = torch.randn(batch_size, 3, 224, 224)
    
    logits, info = model(text=text_input, image=image_input)
    
    assert logits.shape[0] == batch_size
    assert info.get('multimodal', False) == True


def test_memory_system():
    """Test memory system"""
    memory = UnifiedMemoryBackbone(
        d_model=128,
        memory_size=100,
        max_segment_len=64
    )
    
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 128)
    
    output, info = memory(x)
    
    assert output.shape == x.shape
    assert 'step_count' in info


def test_moe():
    """Test Mixture of Experts"""
    moe = MixtureOfExperts(
        d_model=128,
        n_experts=4,
        k=2
    )
    
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 128)
    
    output, info = moe(x)
    
    assert output.shape == x.shape
    assert 'router_info' in info


def test_tokenizer():
    """Test morfo-semantic tokenizer"""
    tokenizer = MorfoSemanticTokenizer(
        vocab_size=256,
        d_model=128
    )
    
    batch_size = 2
    seq_len = 32
    char_ids = torch.randint(0, 256, (batch_size, seq_len))
    
    output = tokenizer(char_ids)
    
    assert output.shape == (batch_size, seq_len, 128)


def test_generation():
    """Test text generation"""
    model = AGIFORMER(
        vocab_size=256,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_seq_len=128
    )
    
    model.eval()
    
    prompt = torch.randint(0, 256, (1, 10))
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=5)
    
    assert generated.shape[1] == 15  # 10 + 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

