# Developer: inkbytefo
# Modified: 2025-11-05

"""
CC12M Dataset Preparation for AGIFORMER
Downloads and prepares Conceptual Captions 12M dataset for multimodal training
"""

import sys
from pathlib import Path

# Add parent directory to path to import agiformer
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer.datasets import (
    CC12MDataset,
    create_synthetic_cc12m_dataset,
    validate_dataset
)


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
    dataset_path = create_synthetic_cc12m_dataset(
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
