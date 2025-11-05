# Developer: inkbytefo
# Modified: 2025-11-05

"""
Data Quality Control System for AGIFORMER
Provides comprehensive validation, quality metrics, and outlier detection
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict, Counter
import statistics
import re
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataQualityController:
    """
    Comprehensive data quality control for multimodal datasets
    """

    def __init__(self, dataset_path: str, dataset_name: str = "dataset"):
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset_name
        self.metadata = {}
        self.quality_report = {}

        # Quality thresholds
        self.thresholds = {
            'min_caption_length': 5,
            'max_caption_length': 100,
            'min_image_size': 32,
            'max_aspect_ratio': 5.0,
            'min_unique_words': 3,
            'max_repetition_ratio': 0.5,
            'brightness_range': (10, 245),  # Avoid too dark/bright images
        }

    def load_metadata(self) -> bool:
        """Load metadata files"""
        try:
            for split in ['train', 'val', 'test']:
                metadata_file = self.dataset_path / f"metadata_{split}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.metadata[split] = json.load(f)
                    logger.info(f"Loaded {len(self.metadata[split])} samples for {split} split")
                else:
                    logger.warning(f"Metadata file not found: {metadata_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return False

    def analyze_caption_quality(self, captions: List[str]) -> Dict[str, Any]:
        """Analyze caption quality metrics"""
        quality_metrics = {
            'total_captions': len(captions),
            'lengths': [],
            'unique_words': [],
            'vocab_size': set(),
            'repetition_ratios': [],
            'issues': defaultdict(int)
        }

        for caption in captions:
            # Basic length analysis
            length = len(caption.split())
            quality_metrics['lengths'].append(length)

            # Unique words
            words = caption.lower().split()
            unique_words = len(set(words))
            quality_metrics['unique_words'].append(unique_words)
            quality_metrics['vocab_size'].update(words)

            # Repetition ratio (repeated words / total words)
            if words:
                word_counts = Counter(words)
                repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
                repetition_ratio = repeated_words / len(words)
                quality_metrics['repetition_ratios'].append(repetition_ratio)

            # Quality checks
            if length < self.thresholds['min_caption_length']:
                quality_metrics['issues']['too_short'] += 1
            if length > self.thresholds['max_caption_length']:
                quality_metrics['issues']['too_long'] += 1
            if unique_words < self.thresholds['min_unique_words']:
                quality_metrics['issues']['low_uniqueness'] += 1
            if repetition_ratio > self.thresholds['max_repetition_ratio']:
                quality_metrics['issues']['high_repetition'] += 1

            # Check for common issues
            if re.search(r'\b(?:http|www|\.com|\.jpg|\.png)\b', caption):
                quality_metrics['issues']['contains_urls'] += 1
            if re.search(r'[^\w\s.,!?-]', caption):  # Non-standard punctuation
                quality_metrics['issues']['strange_chars'] += 1

        # Summary statistics
        quality_metrics['avg_length'] = statistics.mean(quality_metrics['lengths']) if quality_metrics['lengths'] else 0
        quality_metrics['median_length'] = statistics.median(quality_metrics['lengths']) if quality_metrics['lengths'] else 0
        quality_metrics['length_std'] = statistics.stdev(quality_metrics['lengths']) if len(quality_metrics['lengths']) > 1 else 0

        quality_metrics['vocab_size'] = len(quality_metrics['vocab_size'])
        quality_metrics['avg_unique_words'] = statistics.mean(quality_metrics['unique_words']) if quality_metrics['unique_words'] else 0
        quality_metrics['avg_repetition_ratio'] = statistics.mean(quality_metrics['repetition_ratios']) if quality_metrics['repetition_ratios'] else 0

        return quality_metrics

    def analyze_image_quality(self, image_paths: List[Path], max_samples: int = 1000) -> Dict[str, Any]:
        """Analyze image quality metrics"""
        quality_metrics = {
            'total_images': len(image_paths),
            'analyzed_images': min(len(image_paths), max_samples),
            'sizes': [],
            'aspect_ratios': [],
            'brightness_values': [],
            'issues': defaultdict(int)
        }

        # Sample images for analysis
        sample_paths = image_paths[:max_samples]

        for image_path in tqdm(sample_paths, desc="Analyzing images"):
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')

                    # Basic size analysis
                    width, height = img.size
                    quality_metrics['sizes'].append((width, height))

                    aspect_ratio = max(width, height) / min(width, height)
                    quality_metrics['aspect_ratios'].append(aspect_ratio)

                    # Brightness analysis
                    img_array = np.array(img)
                    brightness = np.mean(img_array)
                    quality_metrics['brightness_values'].append(brightness)

                    # Quality checks
                    if width < self.thresholds['min_image_size'] or height < self.thresholds['min_image_size']:
                        quality_metrics['issues']['too_small'] += 1
                    if aspect_ratio > self.thresholds['max_aspect_ratio']:
                        quality_metrics['issues']['extreme_aspect_ratio'] += 1
                    if not (self.thresholds['brightness_range'][0] <= brightness <= self.thresholds['brightness_range'][1]):
                        quality_metrics['issues']['brightness_issue'] += 1

            except Exception as e:
                quality_metrics['issues']['corrupted'] += 1
                logger.warning(f"Failed to analyze image {image_path}: {e}")

        # Summary statistics
        if quality_metrics['sizes']:
            widths, heights = zip(*quality_metrics['sizes'])
            quality_metrics['avg_width'] = statistics.mean(widths)
            quality_metrics['avg_height'] = statistics.mean(heights)
            quality_metrics['median_width'] = statistics.median(widths)
            quality_metrics['median_height'] = statistics.median(heights)

        if quality_metrics['aspect_ratios']:
            quality_metrics['avg_aspect_ratio'] = statistics.mean(quality_metrics['aspect_ratios'])
            quality_metrics['median_aspect_ratio'] = statistics.median(quality_metrics['aspect_ratios'])

        if quality_metrics['brightness_values']:
            quality_metrics['avg_brightness'] = statistics.mean(quality_metrics['brightness_values'])
            quality_metrics['brightness_std'] = statistics.stdev(quality_metrics['brightness_values']) if len(quality_metrics['brightness_values']) > 1 else 0

        return quality_metrics

    def detect_outliers(self, data: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[int]:
        """Detect outliers in numerical data"""
        if len(data) < 4:
            return []

        if method == 'iqr':
            # IQR method
            sorted_data = sorted(data)
            q1 = np.percentile(sorted_data, 25)
            q3 = np.percentile(sorted_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            outliers = [i for i, val in enumerate(data) if val < lower_bound or val > upper_bound]

        elif method == 'zscore':
            # Z-score method
            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data)
            if std_val == 0:
                return []

            z_scores = [(val - mean_val) / std_val for val in data]
            outliers = [i for i, z in enumerate(z_scores) if abs(z) > threshold]

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        return outliers

    def find_duplicate_captions(self, captions: List[str]) -> Dict[str, List[int]]:
        """Find duplicate captions and their indices"""
        caption_to_indices = defaultdict(list)

        for i, caption in enumerate(captions):
            # Normalize caption for comparison
            normalized = caption.lower().strip()
            normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
            caption_to_indices[normalized].append(i)

        # Filter to only duplicates
        duplicates = {caption: indices for caption, indices in caption_to_indices.items() if len(indices) > 1}

        return duplicates

    def analyze_caption_image_alignment(self, metadata: List[Dict], max_samples: int = 1000) -> Dict[str, Any]:
        """Analyze alignment between captions and images (basic semantic check)"""
        alignment_metrics = {
            'total_pairs': len(metadata),
            'analyzed_pairs': min(len(metadata), max_samples),
            'potential_mismatches': 0,
            'alignment_score': 0.0
        }

        # This is a simplified analysis - real implementation would use CLIP or similar
        # For now, we'll do basic keyword matching

        sample_metadata = metadata[:max_samples]

        for item in tqdm(sample_metadata, desc="Analyzing caption-image alignment"):
            caption = item['caption'].lower()
            image_file = item['image_file']

            # Extract potential keywords from caption
            words = re.findall(r'\b\w+\b', caption)
            content_words = [w for w in words if len(w) > 2 and w not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'had', 'by', 'hot', 'but', 'about', 'were', 'this', 'that'}]

            # Check if image filename contains any content words (very basic)
            image_name_lower = image_file.lower()
            matches = sum(1 for word in content_words if word in image_name_lower)

            if matches == 0:
                alignment_metrics['potential_mismatches'] += 1

        # Calculate alignment score
        if alignment_metrics['analyzed_pairs'] > 0:
            alignment_metrics['alignment_score'] = 1.0 - (alignment_metrics['potential_mismatches'] / alignment_metrics['analyzed_pairs'])

        return alignment_metrics

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        logger.info(f"Generating quality report for {self.dataset_name}")

        self.quality_report = {
            'dataset_name': self.dataset_name,
            'dataset_path': str(self.dataset_path),
            'splits': {},
            'overall': {}
        }

        total_samples = 0
        all_captions = []
        all_image_paths = []

        # Analyze each split
        for split, metadata in self.metadata.items():
            logger.info(f"Analyzing {split} split with {len(metadata)} samples")

            split_report = {
                'num_samples': len(metadata),
                'caption_quality': {},
                'image_quality': {},
                'outliers': {},
                'duplicates': {},
                'alignment': {}
            }

            # Extract captions and image paths
            captions = [item['caption'] for item in metadata]
            image_paths = [self.dataset_path / "images" / item['image_file'] for item in metadata]

            all_captions.extend(captions)
            all_image_paths.extend(image_paths)

            # Caption quality analysis
            split_report['caption_quality'] = self.analyze_caption_quality(captions)

            # Image quality analysis (sample for performance)
            existing_images = [p for p in image_paths if p.exists()]
            split_report['image_quality'] = self.analyze_image_quality(existing_images, max_samples=500)

            # Outlier detection
            if split_report['caption_quality']['lengths']:
                length_outliers = self.detect_outliers(split_report['caption_quality']['lengths'])
                split_report['outliers']['caption_length'] = len(length_outliers)

            # Duplicate detection
            duplicates = self.find_duplicate_captions(captions)
            split_report['duplicates']['num_duplicate_groups'] = len(duplicates)
            split_report['duplicates']['total_duplicates'] = sum(len(indices) - 1 for indices in duplicates.values())

            # Caption-image alignment (basic)
            split_report['alignment'] = self.analyze_caption_image_alignment(metadata, max_samples=500)

            self.quality_report['splits'][split] = split_report
            total_samples += len(metadata)

        # Overall statistics
        self.quality_report['overall'] = {
            'total_samples': total_samples,
            'total_splits': len(self.metadata),
            'caption_quality': self.analyze_caption_quality(all_captions),
            'duplicate_analysis': {
                'total_duplicate_groups': len(self.find_duplicate_captions(all_captions)),
            }
        }

        return self.quality_report

    def save_report(self, output_path: Optional[str] = None) -> str:
        """Save quality report to JSON file"""
        if output_path is None:
            output_path = self.dataset_path / f"{self.dataset_name}_quality_report.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.quality_report, f, indent=2, ensure_ascii=False)

        logger.info(f"Quality report saved to: {output_path}")
        return str(output_path)

    def print_summary(self) -> None:
        """Print human-readable summary of quality report"""
        if not self.quality_report:
            logger.warning("No quality report available. Run generate_quality_report() first.")
            return

        print("\n" + "="*80)
        print(f"📊 DATA QUALITY REPORT: {self.dataset_name.upper()}")
        print("="*80)

        overall = self.quality_report['overall']
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"📈 Total Samples: {overall['total_samples']:,}")
        print(f"📂 Splits: {self.quality_report['overall']['total_splits']}")

        print(f"\n📝 CAPTION QUALITY:")
        cq = overall['caption_quality']
        print(f"   • Average length: {cq['avg_length']:.1f} words")
        print(f"   • Median length: {cq['median_length']:.1f} words")
        print(f"   • Vocabulary size: {cq['vocab_size']:,} unique words")
        print(f"   • Average unique words per caption: {cq['avg_unique_words']:.1f}")

        if cq['issues']:
            print(f"   • Quality issues found:")
            for issue, count in cq['issues'].items():
                print(f"     - {issue.replace('_', ' ').title()}: {count:,}")

        print(f"\n🖼️  IMAGE QUALITY:")
        for split_name, split_data in self.quality_report['splits'].items():
            iq = split_data['image_quality']
            if iq['analyzed_images'] > 0:
                print(f"   • {split_name.upper()} split ({iq['analyzed_images']} analyzed):")
                print(f"     - Average size: {iq['avg_width']:.0f}×{iq['avg_height']:.0f}")
                print(f"     - Average brightness: {iq['avg_brightness']:.1f}")
                if iq['issues']:
                    print(f"     - Issues: {dict(iq['issues'])}")

        print(f"\n🔍 DATA INTEGRITY:")
        total_duplicates = sum(split['duplicates']['total_duplicates'] for split in self.quality_report['splits'].values())
        print(f"   • Total duplicate captions: {total_duplicates:,}")

        for split_name, split_data in self.quality_report['splits'].items():
            alignment = split_data['alignment']
            if alignment['analyzed_pairs'] > 0:
                score = alignment['alignment_score'] * 100
                print(f"   • {split_name.upper()} caption-image alignment: {score:.1f}%")

        print("\n" + "="*80)

    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on quality report"""
        if not self.quality_report:
            return ["Run quality analysis first"]

        recommendations = []

        overall = self.quality_report['overall']
        cq = overall['caption_quality']

        # Caption quality recommendations
        if cq['avg_length'] < 8:
            recommendations.append("Captions are quite short. Consider using more descriptive captions.")
        if cq['avg_unique_words'] < 5:
            recommendations.append("Low vocabulary diversity. Consider more varied caption generation.")

        if cq['issues'].get('too_short', 0) > overall['total_samples'] * 0.01:
            recommendations.append(f"Remove {cq['issues']['too_short']} captions that are too short (< {self.thresholds['min_caption_length']} words)")

        if cq['issues'].get('contains_urls', 0) > 0:
            recommendations.append(f"Remove {cq['issues']['contains_urls']} captions containing URLs")

        # Duplicate recommendations
        total_duplicates = sum(split['duplicates']['total_duplicates'] for split in self.quality_report['splits'].values())
        if total_duplicates > overall['total_samples'] * 0.05:
            recommendations.append(f"High duplicate ratio ({total_duplicates/overall['total_samples']:.1%}). Consider deduplication.")

        # Image quality recommendations
        for split_name, split_data in self.quality_report['splits'].items():
            iq = split_data['image_quality']
            if iq['issues'].get('corrupted', 0) > iq['analyzed_images'] * 0.01:
                recommendations.append(f"{split_name.upper()}: Remove corrupted images ({iq['issues']['corrupted']})")

        return recommendations


def run_quality_analysis(dataset_path: str, dataset_name: str = "dataset", save_report: bool = True) -> DataQualityController:
    """Convenience function to run full quality analysis"""
    controller = DataQualityController(dataset_path, dataset_name)

    if not controller.load_metadata():
        raise RuntimeError("Failed to load dataset metadata")

    logger.info("Starting comprehensive quality analysis...")
    controller.generate_quality_report()

    if save_report:
        controller.save_report()

    controller.print_summary()

    recommendations = controller.get_recommendations()
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   • {rec}")

    return controller


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AGIFORMER Data Quality Analysis")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset directory")
    parser.add_argument("--dataset_name", default="dataset", help="Name of the dataset")
    parser.add_argument("--output", help="Output path for quality report")

    args = parser.parse_args()

    controller = run_quality_analysis(args.dataset_path, args.dataset_name)

    if args.output:
        controller.save_report(args.output)
