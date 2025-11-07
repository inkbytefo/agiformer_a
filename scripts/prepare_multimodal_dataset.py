# Developer: inkbytefo
# Modified: 2025-11-07

"""
Multimodal Dataset Preparation for AGIFORMER
Downloads and prepares Conceptual Captions 3M dataset for multimodal training
This script handles image-text pairs for multimodal model training
"""

import sys
from pathlib import Path

# Add parent directory to path to import agiformer
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer.datasets import (
    CC3MDataset,
    create_synthetic_cc3m_dataset,
    validate_dataset
)


def main():
    """Main function to prepare multimodal CC3M dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare multimodal CC3M dataset for AGIFORMER")
    parser.add_argument("--output_dir", type=str, default="data/multimodal_cc3m", help="Output directory for multimodal dataset")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to download/generate")
    parser.add_argument("--validate", action='store_true', help="Validate existing dataset")
    
    args = parser.parse_args()
    
    if args.validate:
        print("Validating existing multimodal dataset...")
        report = validate_dataset(args.output_dir)
        print("\nValidation Report:")
        print(f"Total samples: {report['total_samples']}")
        print(f"Valid images: {report['valid_images']}")
        print(f"Invalid images: {report['invalid_images']}")
        print(f"Missing images: {report['missing_images']}")
        print(f"Average caption length: {report['avg_caption_length']:.2f}")
        return
    
    # Download and prepare multimodal dataset
    print("Preparing multimodal CC3M dataset for AGIFORMER...")
    print("This dataset contains image-text pairs for multimodal training")
    dataset_path = create_synthetic_cc3m_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    # Validate dataset
    print("\nValidating prepared multimodal dataset...")
    report = validate_dataset(args.output_dir)
    
    print("\n" + "="*60)
    print("MULTIMODAL DATASET PREPARATION COMPLETED")
    print("="*60)
    print(f"Dataset location: {dataset_path}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Valid images: {report['valid_images']}")
    print(f"Average caption length: {report['avg_caption_length']:.2f}")
    print("\nDataset Type: MULTIMODAL (Image-Text Pairs)")
    print("Purpose: Training AGIFORMER multimodal components")
    print("\nTo use this dataset in training:")
    print(f"dataset = CC3MDataset('{args.output_dir}', split='train')")
    print("sample = dataset[0]  # Returns dict with 'image', 'caption', etc.")
    print("="*60)


if __name__ == "__main__":
    main()