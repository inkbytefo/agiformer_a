# Developer: inkbytefo
# Modified: 2025-11-05

"""
Base Dataset Classes for AGIFORMER
Provides common functionality for multimodal datasets
"""

import os
import json
import requests
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm


class BaseMultimodalDataset(Dataset):
    """
    Base class for multimodal datasets (image-text pairs)
    Provides common functionality for CC3M, CC12M, and other datasets
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_text_len: int = 77,
        vocab_size: int = 256,
        dataset_name: str = "base"
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.dataset_name = dataset_name

        # Load metadata
        metadata_file = self.data_path / f"metadata_{split}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Filter samples if max_samples specified
        if max_samples:
            self.metadata = self.metadata[:max_samples]

        print(f"Loaded {len(self.metadata)} samples for {split} split from {dataset_name}")

    def __len__(self):
        return len(self.metadata)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess image with error handling"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size)
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, *self.image_size)

    def _process_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process text into input and target tensors"""
        # Convert text to character IDs (AGIFORMER's tokenizer)
        char_ids = [ord(c) % self.vocab_size for c in text[:self.max_text_len]]

        # Pad or truncate
        if len(char_ids) < self.max_text_len:
            char_ids = char_ids + [0] * (self.max_text_len - len(char_ids))
        else:
            char_ids = char_ids[:self.max_text_len]

        # Input and target (shifted by 1 for language modeling)
        input_ids = torch.tensor(char_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(char_ids[1:], dtype=torch.long)

        return input_ids, target_ids

    def __getitem__(self, idx) -> Dict:
        """Get a single sample - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement __getitem__")


class SyntheticDatasetGenerator:
    """
    Generator for synthetic datasets for testing and development
    """

    def __init__(self, dataset_name: str = "synthetic"):
        self.dataset_name = dataset_name
        self.caption_templates = [
            "A beautiful {color} {object} in {location}",
            "The {object} is {adjective} and {adjective}",
            "People {action} with {object} in {location}",
            "A {size} {object} {action} {preposition} the {location}",
            "The {color} {object} {action} during {time}",
            "{object} and {object} in {location}",
            "A person {action} a {color} {object}",
            "The {location} is filled with {adjective} {object}",
            "{object} {action} in {weather} weather",
            "A {adjective} {object} sits {preposition} a {object}"
        ]

        self.templates_data = {
            'colors': ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"],
            'objects': ["car", "house", "tree", "person", "dog", "cat", "bird", "flower", "building", "road"],
            'locations': ["park", "city", "forest", "beach", "mountain", "street", "garden", "office"],
            'adjectives': ["beautiful", "large", "small", "bright", "dark", "modern", "old", "new"],
            'actions': ["standing", "sitting", "walking", "running", "playing", "working", "sleeping", "eating"],
            'prepositions': ["on", "in", "under", "near", "above", "below", "beside", "behind"],
            'sizes': ["big", "small", "huge", "tiny", "medium", "large", "little"],
            'times': ["morning", "afternoon", "evening", "night", "daytime", "sunset", "sunrise"],
            'weather': ["sunny", "rainy", "cloudy", "snowy", "windy", "clear", "foggy"]
        }

    def generate_caption(self) -> str:
        """Generate a random caption using templates"""
        import random
        template = random.choice(self.caption_templates)

        # Fill template with random choices
        caption = template.format(**{
            key: random.choice(values)
            for key, values in self.templates_data.items()
        })

        return caption

    def generate_metadata(self, num_samples: int, image_prefix: str = "image") -> List[Dict]:
        """Generate synthetic metadata"""
        metadata = []
        for i in range(num_samples):
            caption = self.generate_caption()
            metadata.append({
                'image_file': f"{image_prefix}_{i:06d}.jpg",
                'caption': caption,
                'url': f"https://example.com/{image_prefix}_{i}.jpg"  # Placeholder URL
            })
        return metadata

    def generate_images(self, output_dir: Path, metadata: List[Dict], max_images: int = 2000):
        """Generate synthetic images for testing"""
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        print("Creating synthetic images...")
        for i in tqdm(range(min(len(metadata), max_images))):
            # Create more diverse synthetic images
            seed = i + 42  # Consistent seed
            torch.manual_seed(seed)

            # Generate random colors
            r = torch.randint(0, 256, (1,)).item()
            g = torch.randint(0, 256, (1,)).item()
            b = torch.randint(0, 256, (1,)).item()

            # Create gradient image
            image = Image.new('RGB', (224, 224), (r, g, b))

            # Add some patterns based on index
            if i % 3 == 0:  # Striped pattern
                for x in range(0, 224, 20):
                    for y in range(224):
                        if (x + y) % 40 < 20:
                            image.putpixel((x, y), (255-r, 255-g, 255-b))
            elif i % 3 == 1:  # Circular pattern
                center_x, center_y = 112, 112
                for x in range(224):
                    for y in range(224):
                        dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
                        if dist < 50:
                            image.putpixel((x, y), ((r+128)%256, (g+128)%256, (b+128)%256))
            # else: solid color (already created)

            image_path = images_dir / metadata[i]['image_file']
            image.save(image_path, "JPEG")


def download_image(url: str, image_path: Path, max_retries: int = 3) -> bool:
    """Download a single image with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Save image
            with open(image_path, 'wb') as f:
                f.write(response.content)

            # Verify image can be loaded
            Image.open(image_path).convert('RGB')
            return True

        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    return False


def validate_dataset(data_path: str) -> Dict:
    """Validate the prepared dataset"""
    data_path = Path(data_path)

    validation_report = {
        'total_samples': 0,
        'valid_images': 0,
        'invalid_images': 0,
        'missing_images': 0,
        'avg_caption_length': 0,
        'caption_lengths': []
    }

    # Load metadata
    metadata_file = data_path / "metadata_train.json"
    if not metadata_file.exists():
        validation_report['error'] = "Metadata file not found"
        return validation_report

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    validation_report['total_samples'] = len(metadata)

    images_dir = data_path / "images"
    if not images_dir.exists():
        validation_report['error'] = "Images directory not found"
        return validation_report

    print("Validating dataset...")
    for sample in tqdm(metadata):
        image_file = sample['image_file']
        image_path = images_dir / image_file

        # Check if image exists
        if not image_path.exists():
            validation_report['missing_images'] += 1
            continue

        try:
            # Try to load and validate image
            image = Image.open(image_path)
            if image.size != (224, 224):
                # Resize if needed
                image = image.resize((224, 224))
                image.save(image_path, "JPEG")
            validation_report['valid_images'] += 1
        except Exception as e:
            validation_report['invalid_images'] += 1
            print(f"Invalid image: {image_file} - {e}")

        # Track caption statistics
        caption = sample['caption']
        validation_report['caption_lengths'].append(len(caption))

    # Calculate averages
    if validation_report['caption_lengths']:
        validation_report['avg_caption_length'] = sum(validation_report['caption_lengths']) / len(validation_report['caption_lengths'])

    return validation_report
