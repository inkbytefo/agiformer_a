"""
CC12M Dataset Preparation for AGIFORMER
Downloads and prepares Conceptual Captions 12M dataset for multimodal training
"""

import os
import json
import requests
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm


class CC12MDataset(Dataset):
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
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        
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
        
        print(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx) -> Dict:
        sample = self.metadata[idx]
        
        # Load and preprocess image
        image_path = self.data_path / "images" / sample['image_file']
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.image_size)
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, *self.image_size)
        
        # Process text
        text = sample['caption']
        
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
        
        return {
            'image': image,
            'input_ids': input_ids,
            'target_ids': target_ids,
            'caption': text,
            'image_path': str(image_path)
        }


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


def download_cc12m_subset(
    output_dir: str,
    num_samples: int = 10000,
    max_workers: int = 8
):
    """
    Download a subset of CC12M dataset
    For demonstration purposes, we'll use a smaller synthetic dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating CC12M subset with {num_samples} samples...")
    
    # Create synthetic data for demonstration
    # In real implementation, this would download from actual CC12M URLs
    synthetic_data = []
    
    # Diverse caption templates for more realistic synthetic data
    caption_templates = [
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
    
    colors = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
    objects = ["car", "house", "tree", "person", "dog", "cat", "bird", "flower", "building", "road"]
    locations = ["park", "city", "forest", "beach", "mountain", "street", "garden", "office"]
    adjectives = ["beautiful", "large", "small", "bright", "dark", "modern", "old", "new"]
    actions = ["standing", "sitting", "walking", "running", "playing", "working", "sleeping", "eating"]
    prepositions = ["on", "in", "under", "near", "above", "below", "beside", "behind"]
    sizes = ["big", "small", "huge", "tiny", "medium", "large", "little"]
    times = ["morning", "afternoon", "evening", "night", "daytime", "sunset", "sunrise"]
    weather = ["sunny", "rainy", "cloudy", "snowy", "windy", "clear", "foggy"]
    
    import random
    
    for i in range(num_samples):
        template = random.choice(caption_templates)
        caption = template.format(
            color=random.choice(colors),
            object=random.choice(objects),
            location=random.choice(locations),
            adjective=random.choice(adjectives),
            action=random.choice(actions),
            preposition=random.choice(prepositions),
            size=random.choice(sizes),
            time=random.choice(times),
            weather=random.choice(weather)
        )
        
        synthetic_data.append({
            'image_file': f"cc12m_image_{i:06d}.jpg",
            'caption': caption,
            'url': f"https://example.com/cc12m_{i}.jpg"  # Placeholder URL
        })
    
    # Save metadata
    metadata_file = output_path / "metadata_train.json"
    with open(metadata_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    # Create a small subset for validation (10%)
    val_split = int(num_samples * 0.1)
    val_data = synthetic_data[:val_split]
    val_metadata_file = output_path / "metadata_val.json"
    with open(val_metadata_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Create synthetic images (for demonstration)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("Creating synthetic images...")
    for i in tqdm(range(min(num_samples, 2000))):  # Create only first 2000 images for demo
        # Create more diverse synthetic images
        seed = i + 42  # Consistent seed
        torch.manual_seed(seed)
        
        # Generate random colors
        r = torch.randint(0, 256, (1,)).item()
        g = torch.randint(0, 256, (1,)).item()
        b = torch.randint(0, 256, (1,)).item()
        
        # Create gradient image
        image = Image.new('RGB', (224, 224), (r, g, b))
        
        # Add some patterns
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
        
        image.save(images_dir / f"cc12m_image_{i:06d}.jpg", "JPEG")
    
    print(f"Dataset prepared at: {output_path}")
    return output_path


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


def main():
    """Main function to prepare CC12M dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare CC12M dataset for AGIFORMER")
    parser.add_argument("--output_dir", type=str, default="data/cc12m", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--validate", action="store_true", help="Validate existing dataset")
    
    args = parser.parse_args()
    
    if args.validate:
        print("Validating existing dataset...")
        report = validate_dataset(args.output_dir)
        print("\nValidation Report:")
        print(f"Total samples: {report['total_samples']}")
        print(f"Valid images: {report['valid_images']}")
        print(f"Invalid images: {report['invalid_images']}")
        print(f"Missing images: {report['missing_images']}")
        print(f"Average caption length: {report['avg_caption_length']:.2f}")
        return
    
    # Download and prepare dataset
    print("Preparing CC12M dataset for AGIFORMER...")
    dataset_path = download_cc12m_subset(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Validate dataset
    print("\nValidating prepared dataset...")
    report = validate_dataset(args.output_dir)
    
    print("\n" + "="*50)
    print("CC12M DATASET PREPARATION COMPLETED")
    print("="*50)
    print(f"Dataset location: {dataset_path}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Valid images: {report['valid_images']}")
    print(f"Average caption length: {report['avg_caption_length']:.2f}")
    print("\nTo use this dataset in training:")
    print(f"dataset = CC12MDataset('{args.output_dir}', split='train')")
    print("="*50)


if __name__ == "__main__":
    main()
