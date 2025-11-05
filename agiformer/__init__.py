"""
AGIFORMER: Towards Artificial General Intelligence
A revolutionary Transformer architecture designed for AGI
"""

from .model import AGIFORMER
from .core.multimodal_perception import MultimodalPerceptionCore
from .core.morfo_semantic_tokenizer import MorfoSemanticTokenizer
from .core.memory_backbone import UnifiedMemoryBackbone
from .experts.moe import MixtureOfExperts

# Data and quality control
from . import datasets
from . import data_quality

__version__ = "0.1.0"
__all__ = [
    "AGIFORMER",
    "MultimodalPerceptionCore",
    "MorfoSemanticTokenizer",
    "UnifiedMemoryBackbone",
    "MixtureOfExperts",
    "datasets",
    "data_quality",
]
