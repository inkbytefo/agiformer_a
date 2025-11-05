#!/usr/bin/env python3
# Developer: inkbytefo
# Modified: 2025-11-05

"""
Data Quality Analysis Script for AGIFORMER
Demonstrates comprehensive data quality control system
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer.data_quality import run_quality_analysis


def main():
    """Main function for data quality analysis"""
    parser = argparse.ArgumentParser(description="Analyze data quality for AGIFORMER datasets")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to dataset directory to analyze"
    )
    parser.add_argument(
        "--dataset_name",
        default="dataset",
        help="Name of the dataset for reporting"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save quality report (default: dataset_path)"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save quality report to file"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return 1

    print(f"🔍 Analyzing data quality for: {args.dataset_name}")
    print(f"📁 Dataset path: {dataset_path}")

    try:
        # Run comprehensive quality analysis
        controller = run_quality_analysis(
            str(dataset_path),
            args.dataset_name,
            save_report=not args.no_save
        )

        # Additional custom output if needed
        recommendations = controller.get_recommendations()
        if recommendations:
            print(f"\n💡 Found {len(recommendations)} recommendations for data improvement")

        print("✅ Data quality analysis completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Data quality analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
