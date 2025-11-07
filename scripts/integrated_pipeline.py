# Developer: inkbytefo
# Modified: 2025-11-07

#!/usr/bin/env python3
"""
Integrated AGIFORMER Training Pipeline
Coordinates dataset preparation and tokenizer training in proper sequence
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AGIFORMERPipeline:
    """
    Integrated pipeline for AGIFORMER dataset preparation and tokenizer training
    """
    
    def __init__(self, base_output_dir: str = "data"):
        self.base_output_dir = Path(base_output_dir)
        self.turkish_corpus_dir = self.base_output_dir / "turkish_corpus"
        self.multimodal_dir = self.base_output_dir / "multimodal_cc3m"
        self.tokenizer_dir = Path("tokenizer")
        
        # Ensure directories exist
        self.base_output_dir.mkdir(exist_ok=True)
        self.turkish_corpus_dir.mkdir(exist_ok=True)
        self.multimodal_dir.mkdir(exist_ok=True)
        self.tokenizer_dir.mkdir(exist_ok=True)
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status"""
        print(f"\n🔄 {description}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Success!")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print("Error output:", e.stderr)
            return False
    
    def prepare_turkish_corpus(self, dataset_type: str = "synthetic_turkish", 
                              size_gb: float = 1.0, format_type: str = "jsonl") -> bool:
        """Step 1: Prepare Turkish text corpus"""
        cmd = [
            sys.executable, "scripts/prepare_real_dataset.py",
            "--dataset", dataset_type,
            "--output", str(self.turkish_corpus_dir),
            "--size", str(size_gb),
            "--format", format_type
        ]
        
        return self.run_command(cmd, f"Preparing Turkish corpus ({dataset_type})")
    
    def prepare_multimodal_dataset(self, num_samples: int = 10000) -> bool:
        """Step 2: Prepare multimodal CC3M dataset"""
        cmd = [
            sys.executable, "scripts/prepare_multimodal_dataset.py",
            "--output_dir", str(self.multimodal_dir),
            "--num_samples", str(num_samples)
        ]
        
        return self.run_command(cmd, f"Preparing multimodal dataset ({num_samples} samples)")
    
    def train_tokenizer(self, vocab_size: int = 32000, model_type: str = "unigram") -> bool:
        """Step 3: Train MorphoPiece tokenizer on prepared corpus"""
        cmd = [
            sys.executable, "scripts/train_tokenizer.py",
            "--train",
            "--output", str(self.tokenizer_dir / "morphopiece"),
            "--vocab-size", str(vocab_size),
            "--model-type", model_type,
            "--character-coverage", "1.0"
        ]
        
        return self.run_command(cmd, f"Training MorphoPiece tokenizer (vocab_size={vocab_size})")
    
    def validate_outputs(self) -> Dict[str, bool]:
        """Validate that all expected outputs exist"""
        results = {}
        
        # Check Turkish corpus
        corpus_txt = self.turkish_corpus_dir / "turkish_corpus_phase1.txt"
        corpus_jsonl = self.turkish_corpus_dir / "turkish_corpus_phase1.jsonl"
        results["turkish_corpus_txt"] = corpus_txt.exists()
        results["turkish_corpus_jsonl"] = corpus_jsonl.exists()
        
        # Check multimodal dataset
        metadata_train = self.multimodal_dir / "metadata_train.json"
        metadata_val = self.multimodal_dir / "metadata_val.json"
        images_dir = self.multimodal_dir / "images"
        results["multimodal_metadata_train"] = metadata_train.exists()
        results["multimodal_metadata_val"] = metadata_val.exists()
        results["multimodal_images_dir"] = images_dir.exists()
        
        # Check tokenizer outputs
        tokenizer_model = self.tokenizer_dir / "morphopiece.model"
        tokenizer_vocab = self.tokenizer_dir / "morphopiece.vocab"
        tokenizer_json = self.tokenizer_dir / "morphopiece_vocab.json"
        results["tokenizer_model"] = tokenizer_model.exists()
        results["tokenizer_vocab"] = tokenizer_vocab.exists()
        results["tokenizer_json"] = tokenizer_json.exists()
        
        return results
    
    def run_full_pipeline(self, 
                         turkish_dataset: str = "synthetic_turkish",
                         turkish_size_gb: float = 1.0,
                         turkish_format: str = "jsonl",
                         multimodal_samples: int = 10000,
                         tokenizer_vocab_size: int = 32000,
                         skip_turkish: bool = False,
                         skip_multimodal: bool = False,
                         skip_tokenizer: bool = False) -> bool:
        """Run the complete pipeline"""
        
        print("\n" + "="*80)
        print("🚀 AGIFORMER INTEGRATED TRAINING PIPELINE")
        print("="*80)
        
        start_time = time.time()
        success = True
        
        # Step 1: Turkish Corpus Preparation
        if not skip_turkish:
            if not self.prepare_turkish_corpus(turkish_dataset, turkish_size_gb, turkish_format):
                print("❌ Turkish corpus preparation failed!")
                success = False
        else:
            print("⏭️  Skipping Turkish corpus preparation")
        
        # Step 2: Multimodal Dataset Preparation
        if not skip_multimodal and success:
            if not self.prepare_multimodal_dataset(multimodal_samples):
                print("❌ Multimodal dataset preparation failed!")
                success = False
        elif skip_multimodal:
            print("⏭️  Skipping multimodal dataset preparation")
        
        # Step 3: Tokenizer Training
        if not skip_tokenizer and success:
            if not self.train_tokenizer(tokenizer_vocab_size):
                print("❌ Tokenizer training failed!")
                success = False
        elif skip_tokenizer:
            print("⏭️  Skipping tokenizer training")
        
        # Validation
        print("\n" + "="*60)
        print("🔍 VALIDATING OUTPUTS")
        print("="*60)
        
        validation_results = self.validate_outputs()
        
        for output_name, exists in validation_results.items():
            status = "✅" if exists else "❌"
            print(f"{status} {output_name}")
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        
        if success:
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("\n📁 Generated files:")
            print(f"   Turkish corpus: {self.turkish_corpus_dir}")
            print(f"   Multimodal dataset: {self.multimodal_dir}")
            print(f"   Tokenizer: {self.tokenizer_dir}")
            print("\n🔧 Usage examples:")
            print(f"   # Turkish corpus")
            print(f"   from agiformer.datasets.text_datasets import TurkishTextDataset")
            print(f"   dataset = TurkishTextDataset('{self.turkish_corpus_dir}/turkish_corpus_phase1.jsonl')")
            print(f"   ")
            print(f"   # Multimodal dataset")
            print(f"   from agiformer.datasets import CC3MDataset")
            print(f"   dataset = CC3MDataset('{self.multimodal_dir}', split='train')")
            print(f"   ")
            print(f"   # Tokenizer")
            print(f"   from agiformer.language.tokenizer import MorphoPiece")
            print(f"   tokenizer = MorphoPiece('{self.tokenizer_dir}/morphopiece.model')")
        else:
            print("\n❌ PIPELINE FAILED!")
            print("Check the error messages above for details.")
        
        print("="*80)
        return success


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="AGIFORMER Integrated Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with defaults
  python integrated_pipeline.py
  
  # Run with real Turkish dataset
  python integrated_pipeline.py --turkish-dataset finewiki_turkish --turkish-size 2.0
  
  # Skip certain steps
  python integrated_pipeline.py --skip-multimodal --skip-tokenizer
  
  # Custom configuration
  python integrated_pipeline.py --turkish-size 3.0 --multimodal-samples 50000 --tokenizer-vocab 64000
        """
    )
    
    # Turkish corpus options
    parser.add_argument("--turkish-dataset", type=str, 
                       choices=["synthetic_turkish", "finewiki_turkish", "oscar_turkish", "mc4_turkish"],
                       default="synthetic_turkish",
                       help="Turkish dataset type (default: synthetic_turkish)")
    parser.add_argument("--turkish-size", type=float, default=1.0,
                       help="Turkish corpus size in GB (default: 1.0)")
    parser.add_argument("--turkish-format", type=str, choices=["txt", "jsonl"], default="jsonl",
                       help="Turkish corpus output format (default: jsonl)")
    
    # Multimodal dataset options
    parser.add_argument("--multimodal-samples", type=int, default=10000,
                       help="Number of multimodal samples (default: 10000)")
    
    # Tokenizer options
    parser.add_argument("--tokenizer-vocab", type=int, default=32000,
                       help="Tokenizer vocabulary size (default: 32000)")
    parser.add_argument("--tokenizer-model", type=str, choices=["unigram", "bpe", "char", "word"],
                       default="unigram", help="Tokenizer model type (default: unigram)")
    
    # Skip options
    parser.add_argument("--skip-turkish", action="store_true",
                       help="Skip Turkish corpus preparation")
    parser.add_argument("--skip-multimodal", action="store_true",
                       help="Skip multimodal dataset preparation")
    parser.add_argument("--skip-tokenizer", action="store_true",
                       help="Skip tokenizer training")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Base output directory (default: data)")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = AGIFORMERPipeline(args.output_dir)
    
    success = pipeline.run_full_pipeline(
        turkish_dataset=args.turkish_dataset,
        turkish_size_gb=args.turkish_size,
        turkish_format=args.turkish_format,
        multimodal_samples=args.multimodal_samples,
        tokenizer_vocab_size=args.tokenizer_vocab,
        skip_turkish=args.skip_turkish,
        skip_multimodal=args.skip_multimodal,
        skip_tokenizer=args.skip_tokenizer
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()