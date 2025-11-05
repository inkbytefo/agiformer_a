# Developer: inkbytefo
# Modified: 2025-11-05

"""
Pytest fixtures for AGIFORMER testing
Provides common test utilities and data setup
"""

import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER


@pytest.fixture(scope="session")
def config():
    """Load base configuration for testing"""
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope="session")
def model_config(config):
    """Extract model configuration"""
    return config['model']


@pytest.fixture(scope="session")
def training_config(config):
    """Extract training configuration"""
    return config['training']


@pytest.fixture
def dummy_text_data(model_config):
    """Create dummy text data for testing"""
    batch_size = 2
    seq_len = 32

    # Simple text data
    dummy_text = "The quick brown fox jumps over the lazy dog."
    char_ids = [ord(c) % model_config['vocab_size'] for c in dummy_text * (seq_len // len(dummy_text) + 1)]
    char_ids = char_ids[:seq_len]

    input_ids = torch.tensor([char_ids[:-1] for _ in range(batch_size)], dtype=torch.long)
    target_ids = torch.tensor([char_ids[1:] for _ in range(batch_size)], dtype=torch.long)

    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'batch_size': batch_size,
        'seq_len': seq_len
    }


@pytest.fixture
def dummy_multimodal_data(dummy_text_data):
    """Create dummy multimodal data for testing"""
    batch_size = dummy_text_data['batch_size']
    # Create dummy images (3 channels, 224x224)
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    return {
        **dummy_text_data,
        'images': dummy_images
    }


@pytest.fixture
def base_model(model_config):
    """Create base AGIFORMER model for testing"""
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=384,  # Smaller model for testing
        n_layers=2,   # Fewer layers for quick test
        n_heads=6,    # Fewer heads
        d_ff=1536,    # Smaller FFN
        n_experts=4,  # MoE enabled
        expert_types=model_config['expert_types'],
        memory_size=model_config['memory_size'],
        max_seq_len=128,
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=model_config['use_memory'],
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    )
    return model


@pytest.fixture
def moe_model(model_config):
    """Create AGIFORMER model with MoE focus"""
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=384,
        n_layers=2,
        n_heads=6,
        d_ff=1536,
        n_experts=4,  # MoE enabled
        expert_types=model_config['expert_types'],
        memory_size=1000,
        max_seq_len=64,
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=True,
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    )
    return model


@pytest.fixture
def memory_model(model_config):
    """Create AGIFORMER model with Memory focus"""
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=384,
        n_layers=2,
        n_heads=6,
        d_ff=1536,
        n_experts=4,
        expert_types=model_config['expert_types'],
        memory_size=1000,  # Smaller memory for testing
        max_seq_len=64,
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=True,  # Memory enabled
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    )
    return model


@pytest.fixture
def introspection_model(model_config):
    """Create AGIFORMER model with Introspection focus"""
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=384,
        n_layers=2,
        n_heads=6,
        d_ff=1536,
        n_experts=4,
        expert_types=model_config['expert_types'],
        memory_size=1000,
        max_seq_len=64,
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=True,
        use_introspection=True,  # Introspection enabled
        use_multimodal=model_config['use_multimodal']
    )
    return model


@pytest.fixture
def optimizer(base_model):
    """Create optimizer for testing"""
    return torch.optim.AdamW(base_model.parameters(), lr=0.001)


@pytest.fixture
def criterion():
    """Create loss criterion for testing"""
    return nn.CrossEntropyLoss(ignore_index=0)


@pytest.fixture
def device():
    """Get available device for testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model_on_device(base_model, device):
    """Move model to appropriate device"""
    return base_model.to(device)


@pytest.fixture
def data_on_device(dummy_text_data, device):
    """Move data to appropriate device"""
    return {
        'input_ids': dummy_text_data['input_ids'].to(device),
        'target_ids': dummy_text_data['target_ids'].to(device),
        'batch_size': dummy_text_data['batch_size'],
        'seq_len': dummy_text_data['seq_len']
    }


def run_training_step(model, input_ids, target_ids, optimizer, criterion, device):
    """Common training step logic for tests"""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    logits, info = model(text=input_ids)
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target_ids.view(-1)

    loss = criterion(logits_flat, target_flat)

    # Add MoE load balancing loss if present
    total_loss = loss
    for block_info in info['blocks']:
        if 'moe' in block_info and 'router_info' in block_info['moe']:
            if 'load_balancing_loss' in block_info['moe']['router_info']:
                lb_loss = block_info['moe']['router_info']['load_balancing_loss']
                total_loss = total_loss + lb_loss

    # Backward pass
    total_loss.backward()
    optimizer.step()

    return loss.item(), total_loss.item(), info


def run_inference_step(model, input_ids, device):
    """Common inference step logic for tests"""
    model.eval()
    with torch.no_grad():
        logits, info = model(text=input_ids)
    return logits, info


def assert_loss_decreases(initial_loss, final_loss, test_name=""):
    """Assert that loss decreased and provide informative message"""
    assert final_loss < initial_loss, f"{test_name} Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"✅ SUCCESS: {test_name} Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")


def count_model_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total_params, 'trainable': trainable_params}
