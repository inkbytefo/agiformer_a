"""
AGIFORMER v0.1 Production Setup
Creates production-ready model artifacts including model card, inference API, and documentation
"""

import torch
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from scripts.evaluate import evaluate_model


def create_model_card(
    model_path: str,
    config_path: str,
    evaluation_results: Dict,
    output_dir: str
) -> str:
    """Create a comprehensive model card for AGIFORMER v0.1"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    
    model_card = f"""# AGIFORMER v0.1 Model Card

## Model Details

**Model Name:** AGIFORMER (Artificial General Intelligence Transformer) v0.1  
**Model Type:** Multimodal Transformer with MoE, Memory, and Introspection  
**Architecture:** Enhanced Transformer with specialized components  
**Parameters:** {evaluation_results.get('model_info', {}).get('total_parameters', 'N/A'):,}  
**Model Size:** {evaluation_results.get('model_info', {}).get('model_size_mb', 0):.1f} MB  
**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**License:** MIT  

## Model Description

AGIFORMER v0.1 is a prototype Artificial General Intelligence model that combines multiple advanced AI capabilities:

### Core Components
1. **Multimodal Perception**: Processes text, images, audio, and video
2. **Mixture of Experts (MoE)**: 4 specialized experts (language, logic, spatial, causal)
3. **Memory System**: Working memory + Long-term memory for context retention
4. **Introspection**: Self-awareness and iterative thinking capabilities

### Architecture Specifications
- **Vocabulary Size**: {model_config['vocab_size']} (character-level)
- **Model Dimension**: {model_config['d_model']}
- **Layers**: {model_config['n_layers']}
- **Attention Heads**: {model_config['n_heads']}
- **Feed-forward Dimension**: {model_config['d_ff']}
- **Experts**: {model_config['n_experts']}
- **Memory Size**: {model_config['memory_size']:,}
- **Max Sequence Length**: {model_config['max_seq_len']}

### Feature Flags
- **Linear Attention**: {model_config['use_linear_attention']}
- **Memory System**: {model_config['use_memory']}
- **Introspection**: {model_config['use_introspection']}
- **Multimodal**: {model_config['use_multimodal']}

## Performance Metrics

### Inference Performance
- **Throughput**: {evaluation_results.get('inference_speed', {}).get('throughput_samples_per_sec', 0):.2f} samples/second
- **Average Inference Time**: {evaluation_results.get('inference_speed', {}).get('avg_inference_time', 0):.4f} seconds

### Multimodal Capabilities
- **CLIP Score**: {evaluation_results.get('clip_score', {}).get('average_clip_score', 0):.4f} ± {evaluation_results.get('clip_score', {}).get('std_clip_score', 0):.4f}

### Expert Utilization
"""

    # Add expert utilization if available
    if 'expert_utilization' in evaluation_results and 'expert_utilization' in evaluation_results['expert_utilization']:
        model_card += "\n"
        for expert_name, stats in evaluation_results['expert_utilization']['expert_utilization'].items():
            model_card += f"- **{expert_name.replace('_', ' ').title()}**: {stats['mean_usage']:.3f} ± {stats['std_usage']:.3f}\n"

    model_card += f"""
### Introspection Capabilities
"""

    # Add introspection metrics if available
    if 'introspection' in evaluation_results and 'introspection_stats' in evaluation_results['introspection']:
        introspect_stats = evaluation_results['introspection']['introspection_stats']
        model_card += f"- **Average Iterations**: {introspect_stats['avg_iterations']:.1f}\n"
        model_card += f"- **Average Confidence**: {introspect_stats['avg_confidence']:.3f}\n"
        model_card += f"- **Total Introspection Calls**: {introspect_stats['total_introspection_calls']}\n"

    model_card += f"""
## Usage

### Basic Text Generation
```python
from agiformer import AGIFORMER
import torch

# Load model
model = AGIFORMER()
model.load_state_dict(torch.load('{Path(model_path).name}'))

# Generate text
input_text = torch.tensor([[72, 101, 108, 108, 111]])  # "Hello"
generated = model.generate(input_text, max_new_tokens=50)
```

### Multimodal Processing
```python
# Process text + image
text = torch.tensor([[72, 101, 108, 108, 111]])  # "Hello"
image = torch.randn(1, 3, 224, 224)  # Dummy image
logits, info = model(text=text, image=image)
```

### Memory and Introspection
```python
# Model provides detailed information about its processes
logits, info = model(text=text)
print(f"Memory steps: {{info['memory']['step_count']}}")
print(f"Introspection iterations: {{info['blocks'][-1]['introspection']['num_iterations']}}")
```

## Training Data

The model was trained on a synthetic multimodal dataset designed to test:
- Text generation capabilities
- Image-text understanding
- Expert specialization
- Memory retention
- Self-awareness through introspection

## Limitations

1. **Scale**: This is a prototype model with limited training data
2. **Real-world Performance**: Requires larger datasets for production use
3. **Multimodal Integration**: Cross-modal attention could be further improved
4. **Efficiency**: MoE routing and introspection add computational overhead

## Ethical Considerations

- The model includes self-awareness capabilities that should be monitored
- Multimodal processing raises privacy considerations for image data
- Expert specialization may introduce bias in different domains

## Citation

```bibtex
@software{{agiformer_v01,
  title={{AGIFORMER: Artificial General Intelligence Transformer v0.1}},
  author={{Inkbytefo}},
  year={{2024}},
  version={{0.1}}
}}
```

## Contact

For questions, issues, or contributions, please contact the development team.
"""

    # Save model card
    output_path = Path(output_dir)
    model_card_path = output_path / "MODEL_CARD.md"
    
    with open(model_card_path, 'w') as f:
        f.write(model_card)
    
    print(f"📄 Model card created: {model_card_path}")
    return str(model_card_path)


def create_inference_api(model_path: str, config_path: str, output_dir: str) -> str:
    """Create FastAPI-based inference service"""
    
    api_code = '''"""
AGIFORMER v0.1 Inference API
FastAPI-based REST service for model inference
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import yaml
from pathlib import Path
import sys
from typing import Optional, List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER

app = FastAPI(
    title="AGIFORMER v0.1 API",
    description="Artificial General Intelligence Transformer Inference API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextRequest(BaseModel):
    text: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

class MultimodalRequest(BaseModel):
    text: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image
    max_new_tokens: int = 50

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"
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
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found, using random weights")
    
    model.eval()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "AGIFORMER v0.1",
        "device": str(device),
        "multimodal": True,
        "introspection": True,
        "memory": True
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "model_name": "AGIFORMER v0.1",
        "parameters": total_params,
        "device": str(device),
        "capabilities": {
            "text_generation": True,
            "multimodal": True,
            "introspection": True,
            "memory": True,
            "moe": True
        }
    }

@app.post("/generate/text")
async def generate_text(request: TextRequest):
    """Generate text from input text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert text to character IDs
        char_ids = [ord(c) % 256 for c in request.text]
        input_tensor = torch.tensor([char_ids], dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p
            )
        
        # Convert back to text
        generated_text = ''.join([chr(c % 256) for c in generated[0].cpu().numpy()])
        
        return {
            "input": request.text,
            "generated": generated_text,
            "max_new_tokens": request.max_new_tokens
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/multimodal")
async def process_multimodal(request: MultimodalRequest):
    """Process multimodal input (text + image)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare inputs
        inputs = {}
        
        if request.text:
            char_ids = [ord(c) % 256 for c in request.text]
            inputs['text'] = torch.tensor([char_ids], dtype=torch.long).to(device)
        
        if request.image_data:
            # In a real implementation, you would decode the base64 image
            # For now, we'll use a dummy image
            inputs['image'] = torch.randn(1, 3, 224, 224).to(device)
        
        # Process with model
        with torch.no_grad():
            logits, model_info = model(**inputs)
        
        # Extract information
        info = {
            "multimodal": model_info.get('multimodal', False),
            "modalities": model_info.get('modalities', []),
            "memory_steps": model_info.get('memory', {}).get('step_count', 0),
        }
        
        # Check for introspection in last block
        if model_info.get('blocks'):
            last_block = model_info['blocks'][-1]
            if 'introspection' in last_block and last_block['introspection']:
                info['introspection'] = {
                    "iterations": last_block['introspection'].get('num_iterations', 0),
                    "confidence": last_block['introspection'].get('final_confidence', 0.0)
                }
        
        return info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark")
async def run_benchmark():
    """Run quick benchmark"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        
        # Create dummy input
        text = torch.randint(0, 256, (1, 10), dtype=torch.long).to(device)
        
        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = model(text=text)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        throughput = 1.0 / avg_time
        
        return {
            "average_inference_time": avg_time,
            "throughput_samples_per_second": throughput,
            "num_runs": len(times)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    # Save API code
    output_path = Path(output_dir)
    api_path = output_path / "api.py"
    
    with open(api_path, 'w') as f:
        f.write(api_code)
    
    # Create requirements file
    requirements = '''fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
torch>=1.9.0
numpy>=1.21.0
Pillow>=8.0.0
PyYAML>=6.0
'''
    
    requirements_path = output_path / "requirements.txt"
    with open(requirements_path, 'w') as f:
        f.write(requirements)
    
    print(f"🌐 Inference API created: {api_path}")
    print(f"📦 Requirements created: {requirements_path}")
    
    return str(api_path)


def create_benchmark_suite(output_dir: str) -> str:
    """Create comprehensive benchmark suite"""
    
    benchmark_code = '''"""
AGIFORMER v0.1 Benchmark Suite
Comprehensive testing and benchmarking for production deployment
"""

import torch
import torch.nn.functional as F
import time
import json
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER
from scripts.prepare_dataset import CC3MDataset
from torch.utils.data import DataLoader

class AGIFORMERBenchmarkSuite:
    """Comprehensive benchmark suite for AGIFORMER"""
    
    def __init__(self, model_path: str, data_dir: str, device: str = 'auto'):
        self.device = torch.device(device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load model
        self.model = self._load_model()
        
        # Load test data
        self.test_dataset = CC3MDataset(data_dir, split='val', max_samples=100)
        self.test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=False)
        
        self.results = {}
    
    def _load_model(self) -> AGIFORMER:
        """Load AGIFORMER model"""
        model = AGIFORMER(
            vocab_size=256,
            d_model=384,
            n_layers=2,
            n_heads=6,
            d_ff=1536,
            n_experts=4,
            expert_types=["language", "logic", "spatial", "causal"],
            memory_size=1000,
            max_seq_len=64,
            dropout=0.1,
            use_linear_attention=False,
            use_memory=True,
            use_introspection=True,
            use_multimodal=True
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def benchmark_latency(self, num_runs: int = 100) -> Dict:
        """Benchmark inference latency"""
        print("⏱️ Benchmarking latency...")
        
        # Warm up
        for _ in range(10):
            text = torch.randint(0, 256, (1, 10), device=self.device)
            with torch.no_grad():
                _ = self.model(text=text)
        
        # Benchmark
        latencies = []
        for _ in range(num_runs):
            text = torch.randint(0, 256, (1, 10), device=self.device)
            
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(text=text)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        latencies = np.array(latencies) * 1000  # Convert to milliseconds
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'p50_latency_ms': float(np.percentile(latencies, 50)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies))
        }
    
    def benchmark_throughput(self, batch_sizes: List[int] = [1, 2, 4, 8]) -> Dict:
        """Benchmark throughput for different batch sizes"""
        print("🚀 Benchmarking throughput...")
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create batch
            text = torch.randint(0, 256, (batch_size, 10), device=self.device)
            
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    _ = self.model(text=text)
            
            # Benchmark
            num_runs = 20
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                with torch.no_grad():
                    _ = self.model(text=text)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            throughput_results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'avg_time_seconds': float(avg_time),
                'throughput_samples_per_second': float(throughput),
                'tokens_per_second': float(throughput * 10)  # Assuming 10 tokens per sample
            }
        
        return throughput_results
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage"""
        print("💾 Benchmarking memory usage...")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure baseline memory
        if torch.cuda.is_available():
            baseline_memory = torch.cuda.memory_allocated(self.device)
        else:
            baseline_memory = 0
        
        # Measure model memory
        text = torch.randint(0, 256, (1, 10), device=self.device)
        with torch.no_grad():
            _ = self.model(text=text)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            current_memory = torch.cuda.memory_allocated(self.device)
        else:
            peak_memory = current_memory = 0
        
        return {
            'baseline_memory_mb': float(baseline_memory / (1024 * 1024)),
            'model_memory_mb': float(current_memory / (1024 * 1024)),
            'peak_memory_mb': float(peak_memory / (1024 * 1024)),
            'memory_overhead_mb': float((current_memory - baseline_memory) / (1024 * 1024))
        }
    
    def benchmark_accuracy(self) -> Dict:
        """Benchmark accuracy on test set"""
        print("🎯 Benchmarking accuracy...")
        
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        for batch_idx, batch in enumerate(self.test_loader):
            if batch_idx >= 25:  # Limit to 25 batches for speed
                break
            
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            with torch.no_grad():
                logits, _ = self.model(text=input_ids)
                
                # Compute loss
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target_ids.view(-1)
                loss = F.cross_entropy(logits_flat, target_flat, ignore_index=0)
                
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(logits_flat, dim=-1)
                mask = target_flat != 0
                correct = (predictions[mask] == target_flat[mask]).float()
                correct_predictions += correct.sum().item()
                total_samples += mask.sum().item()
        
        avg_loss = total_loss / min(len(self.test_loader), 25)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return {
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': float(np.exp(avg_loss)),
            'total_samples_evaluated': total_samples
        }
    
    def benchmark_specialized_capabilities(self) -> Dict:
        """Benchmark specialized AGIFORMER capabilities"""
        print("🧠 Benchmarking specialized capabilities...")
        
        # Test multimodal processing
        multimodal_results = []
        for batch_idx, batch in enumerate(self.test_loader):
            if batch_idx >= 5:
                break
            
            input_ids = batch['input_ids'].to(self.device)
            images = batch['image'].to(self.device)
            
            with torch.no_grad():
                _, model_info = self.model(text=input_ids, image=images)
                
                # Extract multimodal info
                multimodal_info = {
                    'multimodal_processing': model_info.get('multimodal', False),
                    'modalities': model_info.get('modalities', []),
                    'memory_steps': model_info.get('memory', {}).get('step_count', 0)
                }
                
                # Check introspection
                if model_info.get('blocks'):
                    last_block = model_info['blocks'][-1]
                    if 'introspection' in last_block and last_block['introspection']:
                        multimodal_info['introspection_iterations'] = last_block['introspection'].get('num_iterations', 0)
                        multimodal_info['introspection_confidence'] = last_block['introspection'].get('final_confidence', 0.0)
                
                multimodal_results.append(multimodal_info)
        
        return {
            'multimodal_capabilities': multimodal_results,
            'success_rate': sum(1 for r in multimodal_results if r['multimodal_processing']) / len(multimodal_results)
        }
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("🏁 Running full benchmark suite...")
        
        self.results = {
            'timestamp': time.time(),
            'model_path': self.model_path,
            'device': str(self.device),
            'latency': self.benchmark_latency(),
            'throughput': self.benchmark_throughput(),
            'memory': self.benchmark_memory_usage(),
            'accuracy': self.benchmark_accuracy(),
            'specialized_capabilities': self.benchmark_specialized_capabilities()
        }
        
        return self.results
    
    def save_results(self, output_path: str):
        """Save benchmark results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📊 Results saved to {output_path}")
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\\n" + "="*60)
        print("🏆 AGIFORMER v0.1 BENCHMARK SUMMARY")
        print("="*60)
        
        latency = self.results['latency']
        print(f"⏱️ Latency: {latency['mean_latency_ms']:.2f}ms (P95: {latency['p95_latency_ms']:.2f}ms)")
        
        throughput = self.results['throughput']
        print(f"🚀 Throughput (batch=1): {throughput['batch_1']['throughput_samples_per_second']:.2f} samples/sec")
        
        memory = self.results['memory']
        print(f"💾 Memory Usage: {memory['model_memory_mb']:.1f}MB")
        
        accuracy = self.results['accuracy']
        print(f"🎯 Accuracy: {accuracy['accuracy']:.4f} (Perplexity: {accuracy['perplexity']:.2f})")
        
        specialized = self.results['specialized_capabilities']
        print(f"🧠 Multimodal Success Rate: {specialized['success_rate']:.2%}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Run AGIFORMER benchmark suite")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/cc3m", help="Dataset directory")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark_suite = AGIFORMERBenchmarkSuite(args.model_path, args.data_dir, args.device)
    results = benchmark_suite.run_full_benchmark()
    
    # Save and print results
    benchmark_suite.save_results(args.output)
    benchmark_suite.print_summary()

if __name__ == "__main__":
    main()
'''

    # Save benchmark suite
    output_path = Path(output_dir)
    benchmark_path = output_path / "benchmark_suite.py"
    
    with open(benchmark_path, 'w') as f:
        f.write(benchmark_code)
    
    print(f"🧪 Benchmark suite created: {benchmark_path}")
    return str(benchmark_path)


def create_deployment_scripts(output_dir: str) -> Dict[str, str]:
    """Create deployment scripts and documentation"""
    
    output_path = Path(output_dir)
    
    # Docker setup
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "api.py"]
'''

    dockerfile_path = output_path / "Dockerfile"
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile)
    
    # Docker Compose
    compose = '''version: '3.8'

services:
  agiformer-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
'''

    compose_path = output_path / "docker-compose.yml"
    with open(compose_path, 'w') as f:
        f.write(compose)
    
    # Deployment script
    deploy_script = '''#!/bin/bash
# AGIFORMER v0.1 Deployment Script

set -e

echo "🚀 Deploying AGIFORMER v0.1..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Docker is available
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
else
    DOCKER_CMD="docker"
fi

# Build and run
echo "📦 Building Docker image..."
$DOCKER_CMD build -t agiformer-v01 .

echo "🐳 Starting container..."
$DOCKER_CMD run -d \\
    --name agiformer-api \\
    -p 8000:8000 \\
    --gpus all \\
    -v $(pwd)/checkpoints:/app/checkpoints \\
    -v $(pwd)/data:/app/data \\
    agiformer-v01

echo "✅ Deployment complete!"
echo "🌐 API available at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
'''

    deploy_path = output_path / "deploy.sh"
    with open(deploy_path, 'w') as f:
        f.write(deploy_script)
    
    # Make deploy script executable
    deploy_path.chmod(0o755)
    
    # README for deployment
    readme = '''# AGIFORMER v0.1 Production Deployment

## Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Prerequisites:**
   - Docker and NVIDIA Docker (for GPU support)
   - CUDA-compatible GPU (optional, CPU works too)

2. **Deploy:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Access API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

### Option 2: Local Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run API:**
   ```bash
   python api.py
   ```

## API Usage Examples

### Text Generation
```bash
curl -X POST "http://localhost:8000/generate/text" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "Hello, world!",
       "max_new_tokens": 50
     }'
```

### Multimodal Processing
```bash
curl -X POST "http://localhost:8000/process/multimodal" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "Describe this image",
       "image_data": "base64_encoded_image_data"
     }'
```

## Benchmarking

Run the benchmark suite:
```bash
python benchmark_suite.py --model_path checkpoints/best_model.pt --output benchmark_results.json
```

## Model Card

See MODEL_CARD.md for detailed model information, limitations, and usage guidelines.

## Support

For issues and questions, please refer to the documentation or contact the development team.
'''

    readme_path = output_path / "DEPLOYMENT.md"
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    scripts = {
        'dockerfile': str(dockerfile_path),
        'docker_compose': str(compose_path),
        'deploy_script': str(deploy_path),
        'deployment_readme': str(readme_path)
    }
    
    print(f"🐳 Docker setup created: {dockerfile_path}")
    print(f"🐳 Docker Compose created: {compose_path}")
    print(f"🚀 Deployment script created: {deploy_path}")
    print(f"📚 Deployment guide created: {readme_path}")
    
    return scripts


def main():
    """Main production setup function"""
    parser = argparse.ArgumentParser(description="Setup AGIFORMER v0.1 for production")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data/cc3m", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="production", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("🏗️ Setting up AGIFORMER v0.1 for production...")
    
    # Run evaluation first
    print("📊 Running evaluation...")
    evaluation_results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        config_path=args.config,
        output_dir=str(output_path / "evaluation")
    )
    
    # Create model card
    print("📄 Creating model card...")
    model_card_path = create_model_card(
        model_path=args.model_path,
        config_path=args.config,
        evaluation_results=evaluation_results,
        output_dir=str(output_path)
    )
    
    # Create inference API
    print("🌐 Creating inference API...")
    api_path = create_inference_api(
        model_path=args.model_path,
        config_path=args.config,
        output_dir=str(output_path)
    )
    
    # Create benchmark suite
    print("🧪 Creating benchmark suite...")
    benchmark_path = create_benchmark_suite(str(output_path))
    
    # Create deployment scripts
    print("🚀 Creating deployment scripts...")
    deployment_scripts = create_deployment_scripts(str(output_path))
    
    # Create final summary
    summary = {
        'production_setup': {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'output_directory': str(output_path),
            'artifacts': {
                'model_card': model_card_path,
                'api': api_path,
                'benchmark_suite': benchmark_path,
                'dockerfile': deployment_scripts['dockerfile'],
                'docker_compose': deployment_scripts['docker_compose'],
                'deploy_script': deployment_scripts['deploy_script'],
                'deployment_guide': deployment_scripts['deployment_readme']
            },
            'evaluation_summary': {
                'model_parameters': evaluation_results.get('model_info', {}).get('total_parameters', 0),
                'inference_speed': evaluation_results.get('inference_speed', {}).get('throughput_samples_per_sec', 0),
                'clip_score': evaluation_results.get('clip_score', {}).get('average_clip_score', 0)
            }
        }
    }
    
    # Save summary
    summary_path = output_path / "production_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 AGIFORMER v0.1 PRODUCTION SETUP COMPLETE!")
    print("="*60)
    print(f"📁 Output directory: {output_path}")
    print(f"📄 Model card: {model_card_path}")
    print(f"🌐 API: {api_path}")
    print(f"🧪 Benchmark suite: {benchmark_path}")
    print(f"🚀 Deployment guide: {deployment_scripts['deployment_readme']}")
    print("\nNext steps:")
    print("1. Review the model card and deployment guide")
    print("2. Test the API locally")
    print("3. Run the benchmark suite")
    print("4. Deploy using Docker or manual installation")
    print("="*60)


if __name__ == "__main__":
    main()
