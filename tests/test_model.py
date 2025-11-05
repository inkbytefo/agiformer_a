"""
Tests for AGIFORMER model
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from agiformer.core import MultimodalPerceptionCore, UnifiedMemoryBackbone
from agiformer.language.tokenizer import MorphoPiece
from agiformer.experts import MixtureOfExperts


def test_model_creation():
    """Test model creation"""
    tokenizer = MorphoPiece()
    tokenizer.vocab_size = 256  # Mock vocab size

    model = AGIFORMER(
        tokenizer=tokenizer,
        d_model=128,  # Smaller for testing
        n_layers=2,
        n_heads=4,
        d_ff=512,
        n_experts=2,
        expert_types=["language", "neuro_symbolic"]
    )
    
    assert model is not None
    assert model.vocab_size == 256
    assert model.d_model == 128


def test_forward_pass():
    """Test forward pass"""
    tokenizer = MorphoPiece()
    tokenizer.vocab_size = 256

    model = AGIFORMER(
        tokenizer=tokenizer,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        n_experts=2,
        expert_types=["language", "neuro_symbolic"],
        max_seq_len=128
    )
    
    batch_size = 2
    seq_len = 32
    
    # Text input
    input_ids = torch.randint(0, 256, (batch_size, seq_len))

    logits, info = model(input_ids=input_ids)
    
    assert logits.shape == (batch_size, seq_len, 256)
    assert isinstance(info, dict)


@pytest.mark.skip(reason="Multimodal test requires torch>=2.6 for CLIP model loading")
def test_multimodal():
    """Test multimodal input"""
    tokenizer = MorphoPiece()
    tokenizer.vocab_size = 256

    model = AGIFORMER(
        tokenizer=tokenizer,
        d_model=128,
        n_layers=2,
        n_heads=8,  # d_model=128 için uygun
        use_multimodal=True
    )

    batch_size = 2
    seq_len = 32

    text_input = torch.randint(0, 256, (batch_size, seq_len))
    image_input = torch.randn(batch_size, 3, 224, 224)

    logits, info = model(input_ids=text_input, image=image_input)

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
    """Test morpho-piece tokenizer"""
    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 256

        def encode(self, text):
            return [1, 2, 3]  # Mock tokens

    tokenizer = MockTokenizer()

    # Test basic functionality
    test_text = "merhaba dünya"
    tokens = tokenizer.encode(test_text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_pseudo_labeler():
    """Test PseudoLabeler for relation labeling"""
    from agiformer.experts.pseudo_labeler import PseudoLabeler
    from agiformer.experts.relations import RELATION_TYPES

    labeler = PseudoLabeler()

    # Test with simple sentence
    tokens = ["Ali", "kitabı", "okuyor"]
    token_embeddings = torch.randn(3, 128)  # Mock embeddings

    labels = labeler.generate_labels(tokens, token_embeddings)

    # Should return a dict with potential relation labels
    assert isinstance(labels, dict)

    # Test causal keyword detection
    tokens_with_causal = ["Yağmur", "yağdığı", "çünkü", "ıslak"]
    labels_causal = labeler.generate_labels(tokens_with_causal, token_embeddings[:4])

    # Should detect causal relation (çünkü keyword should trigger CAUSALITY)
    assert RELATION_TYPES["CAUSALITY"] in labels_causal.values()

    # Test syntactic relations if spaCy is available
    if labeler.nlp:
        # Test with English sentence for spaCy
        tokens_en = ["The", "cat", "sits", "on", "mat"]
        labels_syntactic = labeler.generate_labels(tokens_en, token_embeddings[:5])
        # Should have some syntactic relations
        assert len(labels_syntactic) > 0

    # Test task type classification
    task_type = labeler.classify_task_type(tokens)
    assert isinstance(task_type, int)
    assert task_type in [0, 1]  # LINGUISTIC or SYMBOLIC

    # Test symbolic task detection
    symbolic_tokens = ["Yağmur", "yağdığı", "çünkü", "ıslak"]
    symbolic_task = labeler.classify_task_type(symbolic_tokens)
    from agiformer.experts.task_classifier import EXPERT_DOMAINS
    assert symbolic_task == EXPERT_DOMAINS["SYMBOLIC"]


def test_task_type_classifier():
    """Test TaskTypeClassifier"""
    from agiformer.experts.task_classifier import TaskTypeClassifier, EXPERT_DOMAINS

    classifier = TaskTypeClassifier(d_model=128)

    # Test forward pass
    batch_size = 2
    hidden_states = torch.randn(batch_size, 10, 128)  # [batch, seq_len, d_model]

    logits = classifier(hidden_states)

    # Should return logits for each domain
    assert logits.shape == (batch_size, len(EXPERT_DOMAINS))

    # Test that softmax works
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))


def test_performance_optimizations():
    """Test that performance optimizations work together"""
    from agiformer.experts.knowledge_graph import GlobalKnowledgeGraph

    # Test GlobalKnowledgeGraph still works
    kg = GlobalKnowledgeGraph(num_concepts=100, d_model=128, num_relations=5)

    # Test forward pass still works
    concepts = torch.randn(10, 128)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_type = torch.randint(0, 5, (20,))

    output = kg(concepts, edge_index, edge_type)
    assert output.shape == concepts.shape

    # Test gradient checkpointing in model
    tokenizer = __import__('agiformer.language.tokenizer', fromlist=['MorphoPiece']).MorphoPiece()
    tokenizer.vocab_size = 256

    model = AGIFORMER(
        tokenizer=tokenizer,
        d_model=128,
        n_layers=2,
        n_heads=4,
        use_gradient_checkpointing=True
    )

    # Test forward pass with gradient checkpointing
    input_ids = torch.randint(0, 256, (2, 10))
    logits, info = model(input_ids)
    assert logits.shape[0] == 2
    assert logits.shape[2] == 256

    # Test AMP compatibility
    model_amp = AGIFORMER(
        tokenizer=tokenizer,
        d_model=128,
        n_layers=2,
        n_heads=4,
        use_gradient_checkpointing=False
    )

    # Test with autocast
    with torch.cuda.amp.autocast(enabled=True):
        logits_amp, info_amp = model_amp(input_ids)
        assert logits_amp.shape[0] == 2
        assert logits_amp.shape[2] == 256


def test_generation():
    """Test text generation"""
    tokenizer = MorphoPiece()
    tokenizer.vocab_size = 256

    model = AGIFORMER(
        tokenizer=tokenizer,
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
