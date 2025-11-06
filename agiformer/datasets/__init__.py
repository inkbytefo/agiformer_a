# Developer: inkbytefo
# Modified: 2025-11-06

"""
AGIFORMER Datasets Package
Provides multimodal and text dataset implementations and utilities
"""

from .base_dataset import (
    BaseMultimodalDataset,
    SyntheticDatasetGenerator,
    validate_dataset,
    download_image
)

from .cc_datasets import (
    CC3MDataset,
    CC12MDataset,
    create_synthetic_cc3m_dataset,
    create_synthetic_cc12m_dataset,
    download_cc3m_subset,  # Backward compatibility
    download_cc12m_subset   # Backward compatibility
)

from .text_datasets import (
    TurkishTextDataset,
    TextDataset,
    SimpleTextDataset,
    create_dataloader
)

__all__ = [
    # Base classes
    'BaseMultimodalDataset',
    'SyntheticDatasetGenerator',
    
    # Multimodal Dataset implementations
    'CC3MDataset',
    'CC12MDataset',
    
    # Text Dataset implementations
    'TurkishTextDataset',
    'TextDataset',
    'SimpleTextDataset',
    
    # Utility functions
    'validate_dataset',
    'download_image',
    'create_synthetic_cc3m_dataset',
    'create_synthetic_cc12m_dataset',
    'create_dataloader',
    
    # Backward compatibility
    'download_cc3m_subset',
    'download_cc12m_subset'
]
