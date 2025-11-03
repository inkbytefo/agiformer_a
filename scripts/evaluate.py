"""
Evaluation and Benchmarking Script for AGIFORMER v0.1
Comprehensive evaluation including CLIP Score, expert utilization, and multimodal capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import json
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from scripts.prepare_dataset import CC3MDataset
from agiformer.utils import count_parameters, format_number


class CLIPScoreEvaluator:
    """
    Simplified CLIP-style evaluation for text-image similarity
    This is a basic implementation for demonstration purposes
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def compute_text_image_similarity(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between text and image embeddings
        """
        # Normalize embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(text_embeddings, image_embeddings.transpose(-2, -1))
        return similarity
    
    def evaluate_batch(self, batch: Dict) -> Dict:
        """Evaluate a single batch"""
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            images = batch['image'].to(self.device)
            captions = batch['caption']
            
            # Get multimodal embeddings
            logits, model_info = self.model(text=input_ids, image=images)
            
            # For CLIP-style evaluation, we would extract text and image embeddings
            # This is a simplified version since AGIFORMER doesn't have separate CLIP encoders
            text_embeddings = logits.mean(dim=1)  # Average pooling over sequence
            image_embeddings = images.flatten(1)  # Flatten images
            
            # Compute similarity
            similarity = self.compute_text_image_similarity(text_embeddings, image_embeddings)
            
            # For simplicity, we'll use the diagonal as positive pairs
            batch_size = similarity.size(0)
            positive_similarity = similarity[range(batch_size), range(batch_size)]
            
            # Compute CLIP Score (simplified)
            clip_score = positive_similarity.mean().item()
            
            return {
                'clip_score': clip_score,
                'similarity_matrix': similarity.cpu().numpy(),
                'model_info': model_info
            }


class AGIFORMERBenchmark:
    """Comprehensive benchmarking suite for AGIFORMER"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        self.results = {}
    
    def benchmark_model_info(self) -> Dict:
        """Benchmark basic model information"""
        params = count_parameters(self.model)
        
        model_info = {
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'non_trainable_parameters': params['non_trainable'],
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024),
        }
        
        return model_info
    
    def benchmark_inference_speed(self, dataloader: DataLoader, num_batches: int = 10) -> Dict:
        """Benchmark inference speed"""
        import time
        
        # Warm up
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                images = batch['image'].to(self.device) if 'image' in batch else None
                _ = self.model(text=input_ids, image=images) if images is not None else self.model(text=input_ids)
        
        # Benchmark
        times = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            images = batch['image'].to(self.device) if 'image' in batch else None
            
            start_time = time.time()
            with torch.no_grad():
                if images is not None:
                    _ = self.model(text=input_ids, image=images)
                else:
                    _ = self.model(text=input_ids)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'throughput_samples_per_sec': 1.0 / avg_time,
            'num_batches_tested': len(times)
        }
    
    def benchmark_expert_utilization(self, dataloader: DataLoader, num_batches: int = 5) -> Dict:
        """Benchmark MoE expert utilization"""
        expert_usage = {i: [] for i in range(4)}  # Assuming 4 experts
        total_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            images = batch['image'].to(self.device) if 'image' in batch else None
            
            with torch.no_grad():
                if images is not None:
                    _, model_info = self.model(text=input_ids, image=images)
                else:
                    _, model_info = self.model(text=input_ids)
            
            # Extract expert usage from model info
            for block_info in model_info.get('blocks', []):
                if 'moe' in block_info and 'router_info' in block_info['moe']:
                    router_info = block_info['moe']['router_info']
                    if 'expert_usage' in router_info:
                        usage = router_info['expert_usage']
                        if isinstance(usage, torch.Tensor):
                            usage = usage.cpu().numpy()
                        
                        for expert_idx in range(4):
                            if expert_idx < len(usage):
                                expert_usage[expert_idx].append(usage[expert_idx])
            
            total_batches += 1
        
        # Compute statistics
        expert_stats = {}
        for expert_idx in range(4):
            if expert_usage[expert_idx]:
                expert_stats[f'expert_{expert_idx}'] = {
                    'mean_usage': np.mean(expert_usage[expert_idx]),
                    'std_usage': np.std(expert_usage[expert_idx]),
                    'min_usage': np.min(expert_usage[expert_idx]),
                    'max_usage': np.max(expert_usage[expert_idx])
                }
        
        return {
            'expert_utilization': expert_stats,
            'total_batches_analyzed': total_batches
        }
    
    def benchmark_introspection(self, dataloader: DataLoader, num_batches: int = 5) -> Dict:
        """Benchmark introspection capabilities"""
        introspection_data = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            images = batch['image'].to(self.device) if 'image' in batch else None
            
            with torch.no_grad():
                if images is not None:
                    _, model_info = self.model(text=input_ids, image=images)
                else:
                    _, model_info = self.model(text=input_ids)
            
            # Extract introspection data
            for block_info in model_info.get('blocks', []):
                if 'introspection' in block_info and block_info['introspection']:
                    introspection_info = block_info['introspection']
                    introspection_data.append({
                        'num_iterations': introspection_info.get('num_iterations', 0),
                        'final_confidence': introspection_info.get('final_confidence', 0.0),
                        'final_error': introspection_info.get('final_error', 0.0)
                    })
        
        if not introspection_data:
            return {'error': 'No introspection data found'}
        
        # Compute statistics
        num_iterations = [d['num_iterations'] for d in introspection_data]
        confidences = [d['final_confidence'] for d in introspection_data]
        errors = [d['final_error'] for d in introspection_data]
        
        return {
            'introspection_stats': {
                'avg_iterations': np.mean(num_iterations),
                'std_iterations': np.std(num_iterations),
                'avg_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences),
                'avg_error': np.mean(errors),
                'std_error': np.std(errors),
                'total_introspection_calls': len(introspection_data)
            }
        }
    
    def benchmark_memory_usage(self, dataloader: DataLoader, num_batches: int = 5) -> Dict:
        """Benchmark memory system usage"""
        memory_stats = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            images = batch['image'].to(self.device) if 'image' in batch else None
            
            with torch.no_grad():
                if images is not None:
                    _, model_info = self.model(text=input_ids, image=images)
                else:
                    _, model_info = self.model(text=input_ids)
            
            # Extract memory usage
            if 'memory' in model_info:
                memory_info = model_info['memory']
                memory_stats.append({
                    'step_count': memory_info.get('step_count', 0),
                    'working_memory_size': memory_info.get('working_memory_size', 0),
                    'longterm_memory_size': memory_info.get('longterm_memory_size', 0)
                })
        
        if not memory_stats:
            return {'error': 'No memory data found'}
        
        # Compute statistics
        step_counts = [d['step_count'] for d in memory_stats]
        working_sizes = [d['working_memory_size'] for d in memory_stats]
        longterm_sizes = [d['longterm_memory_size'] for d in memory_stats]
        
        return {
            'memory_stats': {
                'avg_step_count': np.mean(step_counts),
                'std_step_count': np.std(step_counts),
                'avg_working_memory_size': np.mean(working_sizes),
                'avg_longterm_memory_size': np.mean(longterm_sizes),
                'total_memory_calls': len(memory_stats)
            }
        }


def evaluate_model(
    model_path: str,
    data_dir: str,
    config_path: str = "configs/base_config.yaml",
    output_dir: str = "evaluation_results",
    batch_size: int = 8,
    num_eval_batches: int = 20
) -> Dict:
    """Comprehensive model evaluation"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    # Create model
    model = AGIFORMER(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        d_ff=model_config['d_ff'],
        n_experts=model_config['n_experts'],
        expert_types=model_config['expert_types'],
        memory_size=model_config['memory_size'],
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config['dropout'],
        use_linear_attention=model_config['use_linear_attention'],
        use_memory=model_config['use_memory'],
        use_introspection=model_config['use_introspection'],
        use_multimodal=model_config['use_multimodal']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📂 Loaded model from {model_path}")
    
    # Create evaluation dataset
    eval_dataset = CC3MDataset(
        data_path=data_dir,
        split="val",
        max_samples=min(num_eval_batches * batch_size, 200)
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize evaluators
    benchmark = AGIFORMERBenchmark(model, device)
    clip_evaluator = CLIPScoreEvaluator(model, device)
    
    print(f"🔍 Starting comprehensive evaluation...")
    
    # Run benchmarks
    results = {}
    
    print("📊 Benchmarking model info...")
    results['model_info'] = benchmark.benchmark_model_info()
    
    print("⚡ Benchmarking inference speed...")
    results['inference_speed'] = benchmark.benchmark_inference_speed(eval_dataloader)
    
    print("🧠 Benchmarking expert utilization...")
    results['expert_utilization'] = benchmark.benchmark_expert_utilization(eval_dataloader)
    
    print("🤔 Benchmarking introspection...")
    results['introspection'] = benchmark.benchmark_introspection(eval_dataloader)
    
    print("💾 Benchmarking memory usage...")
    results['memory_usage'] = benchmark.benchmark_memory_usage(eval_dataloader)
    
    print("🎯 Computing CLIP scores...")
    clip_scores = []
    for batch_idx, batch in enumerate(eval_dataloader):
        if batch_idx >= min(num_eval_batches, 10):  # Limit CLIP evaluation
            break
        
        batch_results = clip_evaluator.evaluate_batch(batch)
        clip_scores.append(batch_results['clip_score'])
    
    results['clip_score'] = {
        'average_clip_score': np.mean(clip_scores),
        'std_clip_score': np.std(clip_scores),
        'num_batches_evaluated': len(clip_scores)
    }
    
    # Save results
    results_file = output_path / f"evaluation_results_{Path(model_path).stem}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 AGIFORMER v0.1 EVALUATION RESULTS")
    print("="*60)
    
    print(f"📊 Model Parameters: {format_number(results['model_info']['total_parameters'])}")
    print(f"⚡ Inference Speed: {results['inference_speed']['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"🎯 CLIP Score: {results['clip_score']['average_clip_score']:.4f} ± {results['clip_score']['std_clip_score']:.4f}")
    
    if 'expert_utilization' in results and 'expert_utilization' in results['expert_utilization']:
        print("🧠 Expert Utilization:")
        for expert_name, stats in results['expert_utilization']['expert_utilization'].items():
            print(f"  {expert_name}: {stats['mean_usage']:.3f} ± {stats['std_usage']:.3f}")
    
    if 'introspection' in results and 'introspection_stats' in results['introspection']:
        introspect_stats = results['introspection']['introspection_stats']
        print(f"🤔 Introspection: {introspect_stats['avg_iterations']:.1f} avg iterations, "
              f"{introspect_stats['avg_confidence']:.3f} avg confidence")
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    print("="*60)
    
    return results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate AGIFORMER model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/cc3m", help="Dataset directory")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Config file path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to evaluate")
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_eval_batches=args.num_batches
    )


if __name__ == "__main__":
    main()
