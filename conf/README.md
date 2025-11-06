# Developer: inkbytefo
# Modified: 2025-11-06

# AGIFORMER Configuration Structure

This directory contains the hierarchical configuration system for AGIFORMER using Hydra.

## Structure

```
conf/
├── config.yaml              # Main configuration file
├── model/                   # Model configurations
│   ├── base.yaml           # Base model settings
│   ├── lite.yaml           # Lightweight model
│   └── multimodal.yaml     # Full multimodal model
├── training/                # Training configurations
│   └── base.yaml           # Base training settings
├── hardware/                # Hardware configurations
│   ├── base.yaml           # Base hardware settings
│   ├── default_gpu.yaml    # Default GPU configuration
│   ├── t4_gpu.yaml         # T4 GPU optimized
│   └── cpu.yaml            # CPU configuration
├── logging/                 # Logging configurations
│   └── base.yaml           # Base logging settings
└── experiment/              # Experiment-specific configurations
    ├── phase1_baseline.yaml    # Phase 1 baseline experiment
    └── phase1_lite.yaml        # Phase 1 lite experiment
```

## Usage

### Basic Usage
```bash
python train.py  # Uses default config.yaml
```

### Override Configurations
```bash
# Use different model
python train.py model=lite

# Use different hardware
python train.py hardware=t4_gpu

# Use experiment configuration
python train.py experiment=phase1_baseline

# Combine overrides
python train.py model=multimodal hardware=t4_gpu
```

### Configuration Hierarchy

1. **config.yaml** - Main configuration that composes all defaults
2. **Base configs** - Provide default settings for each component
3. **Specific configs** - Override base settings for particular use cases
4. **Command line overrides** - Highest priority, can override any setting

## Configuration Sections

### Model
- `d_model`: Model dimension
- `n_layers`: Number of transformer layers
- `n_heads`: Number of attention heads
- `vocab_size`: Vocabulary size
- `n_experts`: Number of experts in MoE
- `expert_types`: Types of experts
- Feature flags for various capabilities

### Training
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `epochs`: Number of training epochs
- `optimizer`: Optimizer type
- `use_amp`: Mixed precision training

### Hardware
- `device`: Target device (auto/cuda/cpu)
- `use_amp`: Automatic mixed precision
- Memory management settings
- Performance optimizations

### Logging
- Console and file logging settings
- Weights & Biases integration
- TensorBoard integration

## Best Practices

1. Use specific configurations for experiments rather than command line overrides
2. Keep base configurations minimal and focused
3. Document any non-obvious setting choices
4. Test configuration changes before running long experiments
5. Use version control for configuration changes