# AGIFORMER

// Developer: inkbytefo
// Modified: 2025-11-07

AGIFORMER is a research-grade, production-oriented architecture for AGI-like reasoning with a focus on:

- Mixture-of-Experts (MoE) with dynamic routing
- Unified memory backbone (working + long-term)
- Introspection and meta-learning loops
- Neuro-symbolic reasoning over a global knowledge graph
- Multimodal perception (text, image-ready)
- Deep Turkish language support with agglutinative and morpho-semantic modeling

This repository contains the reference implementation used for controlled experiments, with all components wired through a single, coherent training stack.

---

## Key Design Principles

- Single source of truth:
  - The active training path is [`train.py`](train.py) using Hydra configs under `conf/`.
  - All language-specific capabilities are integrated into `AGIFORMER` via the MoE/experts stack.
- Explicit, debuggable behavior:
  - No silent clamping that hides data errors.
  - Auxiliary losses (e.g. MoE load balancing) are separated from main optimization metrics.
- Turkish-first language modeling:
  - Morphology, semantics, and agglutinative structure are modeled explicitly and consumed in the main model, not via a side-car model.

---

## High-Level Architecture

### 1. Core Model: `AGIFORMER`

Location: [`agiformer/model.py`](agiformer/model.py:142)

AGIFORMER is a Transformer-like backbone augmented with:

- MoE blocks with task-aware routing
- Unified memory backbone
- Optional multimodal perception core
- Optional introspection loop in the top block
- Global knowledge graph integration

Core call:

```python
logits, info = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    morpho_types=morpho_types,                 # optional
    semantic_categories=semantic_categories,   # optional
    image=image,                               # optional
)
```

Key components:

- Embedding:
  - `self.token_embedding`: learned token embeddings from MorphoPiece vocab.
  - Input validation: out-of-range token IDs raise `ValueError` instead of being silently clipped.
- Blocks: `self.blocks: List[AGIFORMERBlock]`
- Output:
  - `self.final_norm` + `self.output_proj` → vocabulary logits.

### 2. Blocks + Mixture-of-Experts

Location: [`agiformer/model.py`](agiformer/model.py:26), [`agiformer/experts/moe.py`](agiformer/experts/moe.py:120)

Each `AGIFORMERBlock`:

- Self-attention (standard or linear)
- TaskTypeClassifier:
  - Predicts domain distribution, used to bias MoE routing.
- MixtureOfExperts:
  - Top-k routing with load-balancing loss.
  - Custom experts: Language, Logic, Spatial, Causal, Neuro-Symbolic.
- Optional IntrospectionLoop (last block).

MoE interface:

```python
output, moe_info = self.moe(
    hidden_states,
    routing_bias=routing_bias,
    attention_mask=attention_mask,
    morpho_types=morpho_types,
    semantic_categories=semantic_categories,
)
```

- `MixtureOfExperts` forwards `**expert_kwargs` into each expert:
  - Experts that care (e.g. LanguageExpert) consume them.
  - Others ignore extra kwargs.

### 3. Language Expert (Integrated TMA-style Capabilities)

Location: [`agiformer/experts/language_expert.py`](agiformer/experts/language_expert.py:1)

The `LanguageExpert` is the canonical Turkish language specialist:

- Uses `AgglutinativeAttention` or standard MHA.
- Optionally enriches hidden states with:
  - `morpho_types`: morphological categories.
  - `semantic_categories`: semantic role/category hints.

Core behavior:

- Receives `x` from the block (already in model space).
- If `morpho_types` and `semantic_categories` are provided:
  - Embeds them via:
    - `morpho_embedding(NUM_MORPHEME_TYPES=23)`
    - `semantic_embedding(NUM_SEMANTIC_CATEGORIES=12)`
  - Adds to hidden states (residual enrichment).
- Runs attention:
  - If agglutinative:
    - Uses `AgglutinativeAttention` with `morpho_types` to bias attention.
  - Else:
    - Uses standard `MultiHeadAttention`.
- Applies FFN + residual.

Result:
- Former `TMA1Model` capabilities are now part of the main AGIFORMER MoE flow.
- No parallel, divergent architecture.

### 4. Neuro-Symbolic Expert

Location: [`agiformer/experts/neuro_symbolic_expert.py`](agiformer/experts/neuro_symbolic_expert.py:33)

- Extracts edges from attention patterns (vectorized with `nonzero` / masking).
- Classifies relations between concept pairs.
- Feeds into a `DynamicKnowledgeGraph` / `GlobalKnowledgeGraph`.
- Projects reasoning results back into the neural space.

Performance:
- Python loops removed from the critical path.
- GPU-friendly tensor operations used for edge construction.

### 5. Memory Backbone

Location: [`agiformer/core/memory_backbone.py`](agiformer/core/memory_backbone.py)

- Provides:
  - Working memory.
  - Long-term memory.
- Integrated directly in `AGIFORMER.forward`:
  - `self.memory(x, use_working_memory=True, use_longterm_memory=True)`

### 6. Multimodal Perception

Location: [`agiformer/core/multimodal_perception.py`](agiformer/core/multimodal_perception.py)

- Processes non-text modalities (e.g., images) into aligned embeddings.
- Controlled by `use_multimodal` flag in model config.
- Integrated in `AGIFORMER.forward` when modalities are present.

### 7. Introspection

Location: [`agiformer/introspection/self_model.py`](agiformer/introspection/self_model.py), [`agiformer/introspection/meta_learning.py`](agiformer/introspection/meta_learning.py)

- `IntrospectionLoop` in the last block:
  - Operates over cumulative hidden states (`previous_states`).
  - Enables meta-reasoning and self-monitoring paths.


---

## Data Pipeline

### 1. Turkish Text Datasets

Location: [`agiformer/datasets/text_datasets.py`](agiformer/datasets/text_datasets.py:19)

`TurkishTextDataset`:

- Supports:
  - `.jsonl` with:
    - `tokens`
    - `morpho_types`
    - `semantic_categories`
  - Plain text with on-the-fly morphology (via `MorphoSplitter`) when enabled.
- Behavior:
  - Encodes tokens via MorphoPiece or char fallback.
  - Generates:
    - `input_ids`, `target_ids`
    - `attention_mask`
    - `morpho_types`, `semantic_categories` when available.
  - No final “blind clamp”:
    - Invalid IDs are mapped to UNK or caught by AGIFORMER’s validation.

`SimpleTextDataset`:
- Lightweight char-level dataset for smoke tests and demos.

### 2. CC / Multimodal Datasets

Location: [`agiformer/datasets/cc_datasets.py`](agiformer/datasets/cc_datasets.py)

- CC12M-style structure with text + image metadata.
- Used when `data_dir` with `metadata_train.json` / `metadata_val.json` is present.

### 3. Batch Preparation

Location: [`train.py`](train.py:108)

`prepare_batch_data`:

- Normalizes all dataset variants into `AGIFORMER.forward` kwargs.
- For dict batches (JSONL):
  - `input_ids`
  - `attention_mask` (if exists)
  - `morpho_types` (if exists)
  - `semantic_categories` (if exists)
  - `target_ids`
- For multimodal:
  - Adds `image` (and future modalities).
- For legacy tuple format:
  - Only `input_ids`, `target_ids`.

---

## Training Loop

Location: [`train.py`](train.py:211)

Entry:
- Hydra-based:
  - `@hydra.main(config_path="conf", config_name="config")`
- Uses `conf/`:
  - `conf/model/*.yaml`
  - `conf/training/*.yaml`
  - `conf/hardware/*.yaml`
  - `conf/logging/*.yaml`
  - `conf/experiment/*.yaml`

Key characteristics:

- Optimizer:
  - AdamW with configurable betas, eps, weight decay.
- Scheduler:
  - `WarmupScheduler` (custom warmup based on `d_model`).
- AMP:
  - `torch.amp.autocast` + `GradScaler`.
- Checkpoints:
  - `CheckpointManager`:
    - `latest.pt`
    - `best_model.pt`
    - rolling `checkpoint_*.pt`.
- Metrics:
  - `MetricsLogger` with optional W&B integration.

Loss handling (fixed and aligned):

- Training:
  - `main_loss = CrossEntropyLoss(ignore_index=0)`
  - `aux_loss = sum(MoE load_balancing_loss from blocks if present)`
  - `total_loss = main_loss + aux_loss`
  - Backprop on `total_loss`.
- Logging:
  - Train:
    - `Training/main_loss`
    - `Training/aux_lb_loss`
  - Validation:
    - `Validation/main_loss` (NO aux terms).
- Checkpoint selection:
  - Uses validation main_loss only.

Result:
- Train and validation curves are directly comparable.
- Regularizers contribute to optimization without contaminating metrics.

---

## Configuration

Configs live under `conf/`:

- `conf/model/base.yaml`:
  - d_model, n_layers, n_heads, d_ff
  - n_experts, expert_types
  - use_memory, use_introspection, use_multimodal
  - use_agglutinative_attention
  - tokenizer_path, max_seq_len
- `conf/training/base.yaml`:
  - batch_size, learning_rate, warmup_steps
  - epochs, max_steps
  - log_interval, eval_interval, save_interval
  - use_gradient_checkpointing, use_amp
- `conf/hardware/*.yaml`:
  - device, T4 / CPU presets, CUDA allocator hints.
- `conf/logging/base.yaml`:
  - console log level, W&B toggles.

Typical run:

```bash
python train.py experiment=phase1_baseline
```

Hydra manages:
- Output directory
- Checkpoints under `<output_dir>/checkpoints`
- Final config snapshot.

---

## Development and Testing

### Linting

Recommended:
- ruff / flake8 for Python.
- Enforce:
  - No unused imports.
  - Type hints for public APIs where reasonable.

Example (if Makefile exists):

```bash
make lint
```

### Tests

Location: `tests/`

- `tests/test_model.py`:
  - Smoke tests for AGIFORMER instantiation and forward.
- Extend with:
  - Dataset shape tests.
  - MoE routing sanity checks.
  - Neuro-symbolic expert edge construction tests.

Run:

```bash
pytest -q
```

### Minimal Smoke Test (Code-Consistent)

Using provided pipeline:

```bash
python train.py \
  experiment=phase1_lite \
  training.max_steps=50 \
  training.batch_size=4 \
  logging.use_wandb=false
```

This should:
- Build AGIFORMER with MoE + LanguageExpert integration.
- Run a short loop without shape/signature errors.
- Log main_loss and aux_lb_loss separately.

---

## Project Status

- DONE:
  - Integrated morpho/semantic-aware LanguageExpert into AGIFORMER MoE.
  - Removed TMA1Model from public API to avoid dual architectures.
  - Fixed:
    - Training/validation loss comparability.
    - Token ID clamping issues.
    - NeuroSymbolicExpert performance bottleneck.
    - attention_mask vs mask inconsistencies in checkpoint paths.
- TODO (docs-level):
  - Keep `docs/` synchronized with this README for future architectural changes.

AGIFORMER is now a single, coherent architecture where advanced Turkish-specific language modeling, memory, introspection, neuro-symbolic reasoning and MoE routing all operate within the same training and deployment graph.
