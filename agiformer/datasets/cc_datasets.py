# Developer: inkbytefo
# Modified: 2025-11-05

"""
CC3M and CC12M Dataset Implementations for AGIFORMER
Concrete implementations of multimodal datasets
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from .base_dataset import BaseMultimodalDataset, SyntheticDatasetGenerator, validate_dataset


class CC3MDataset(BaseMultimodalDataset):
    """
    Conceptual Captions 3M Dataset for AGIFORMER
    Handles image-text pairs with proper preprocessing
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_text_len: int = 77,
        vocab_size: int = 256
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            max_samples=max_samples,
            image_size=image_size,
            max_text_len=max_text_len,
            vocab_size=vocab_size,
            dataset_name="CC3M"
        )

    def __getitem__(self, idx) -> Dict:
        sample = self.metadata[idx]

        # Load and preprocess image
        image_path = self.data_path / "images" / sample['image_file']
        image = self._load_image(image_path)

        # Process text
        text = sample['caption']
        input_ids, target_ids = self._process_text(text)

        return {
            'image': image,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'caption': text,
            'image_path': str(image_path)
        }


class CC12MDataset(BaseMultimodalDataset):
    """
    Conceptual Captions 12M Dataset for AGIFORMER
    Handles image-text pairs with proper preprocessing
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_text_len: int = 77,
        vocab_size: int = 256
    ):
        super().__init__(
            data_path=data_path,
            split=split,
            max_samples=max_samples,
            image_size=image_size,
            max_text_len=max_text_len,
            vocab_size=vocab_size,
            dataset_name="CC12M"
        )

    def __getitem__(self, idx) -> Dict:
        sample = self.metadata[idx]

        # Load and preprocess image
        image_path = self.data_path / "images" / sample['image_file']
        image = self._load_image(image_path)

        # Process text
        text = sample['caption']
        input_ids, target_ids = self._process_text(text)

        return {
            'image': image,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'caption': text,
            'image_path': str(image_path)
        }


def create_synthetic_cc3m_dataset(
    output_dir: str,
    num_samples: int = 10000,
) -> Path:
    """
    Create a synthetic CC3M-like dataset for testing and development
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic CC3M dataset with {num_samples} samples...")

    # Generate synthetic data
    generator = SyntheticDatasetGenerator("CC3M")
    metadata = generator.generate_metadata(num_samples, "image")

    # Save metadata
    metadata_file = output_path / "metadata_train.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create a small subset for validation
    val_data = metadata[:1000]
    val_metadata_file = output_path / "metadata_val.json"
    with open(val_metadata_file, 'w') as f:
        json.dump(val_data, f, indent=2)

    # Generate synthetic images
    generator.generate_images(output_path, metadata, max_images=1000)

    print(f"Dataset prepared at: {output_path}")
    return output_path


def create_synthetic_cc12m_dataset(
    output_dir: str,
    num_samples: int = 10000,
) -> Path:
    """
    Create a synthetic CC12M-like dataset for testing and development
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic CC12M dataset with {num_samples} samples...")

    # Generate synthetic data with more diverse captions
    generator = SyntheticDatasetGenerator("CC12M")

    # CC12M has more diverse captions, so let's generate more varied ones
    metadata = []
    for i in range(num_samples):
        caption = generator.generate_caption()
        metadata.append({
            'image_file': f"cc12m_image_{i:06d}.jpg",
            'caption': caption,
            'url': f"https://example.com/cc12m_{i}.jpg"  # Placeholder URL
        })

    # Save metadata
    metadata_file = output_path / "metadata_train.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create validation split (10%)
    val_split = int(num_samples * 0.1)
    val_data = metadata[:val_split]
    val_metadata_file = output_path / "metadata_val.json"
    with open(val_metadata_file, 'w') as f:
        json.dump(val_data, f, indent=2)

    # Generate synthetic images
    generator.generate_images(output_path, metadata, max_images=2000)

    print(f"Dataset prepared at: {output_path}")
    return output_path


# Backward compatibility - keep the old functions for existing scripts
def download_cc3m_subset(output_dir: str, num_samples: int = 10000, max_workers: int = 8):
    """Backward compatibility wrapper"""
    return create_synthetic_cc3m_dataset(output_dir, num_samples)


def download_cc12m_subset(output_dir: str, num_samples: int = 10000, max_workers: int = 8):
    """Backward compatibility wrapper"""
    return create_synthetic_cc12m_dataset(output_dir, num_samples)
