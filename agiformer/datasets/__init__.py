# Developer: inkbytefo
# Modified: 2025-11-05

"""
AGIFORMER Datasets Package
Provides multimodal dataset implementations and utilities
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

__all__ = [
    # Base classes
    'BaseMultimodalDataset',
    'SyntheticDatasetGenerator',

    # Dataset implementations
    'CC3MDataset',
    'CC12MDataset',

    # Utility functions
    'validate_dataset',
    'download_image',
    'create_synthetic_cc3m_dataset',
    'create_synthetic_cc12m_dataset',

    # Backward compatibility
    'download_cc3m_subset',
    'download_cc12m_subset'
]
