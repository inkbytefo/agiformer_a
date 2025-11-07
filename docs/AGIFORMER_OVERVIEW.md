// Developer: inkbytefo
// Modified: 2025-11-07

# AGIFORMER Technical Overview

This document is the authoritative, code-synced overview of the AGIFORMER architecture as implemented in this repository. It reflects the current state of the source code, not aspirational or legacy designs.

AGIFORMER is a modular AGI-oriented architecture combining:

- Task-aware Mixture-of-Experts (MoE)
- Unified memory backbone
- Introspection and meta-learning
- Neuro-symbolic reasoning on a global knowledge graph
- Multimodal perception
- Native Turkish morpho-semantic modeling integrated into the main model

---

## 1. Top-Level Composition

Main entrypoint:
- [`train.py`](train.py:211)
- Configured via Hydra:
  - `conf/config.yaml` and profiles in:
    - `conf/model/`
    - `conf/training/`
    - `conf/hardware/`
    - `conf/experiment/`
    - `conf/logging/`

Core model:
- [`agiformer/model.py`](agiformer/model.py:142)
- Public class:
  - `AGIFORMER`

High-level data flow:

1. Data loaders construct:
   - `input_ids`, `attention_mask`
   - `morpho_types`, `semantic_categories` (if available)
   - optional multimodal features (e.g. `image`)
2. `AGIFORMER.forward` embeds tokens, applies memory, then stacks `AGIFORMERBlock`s.
3. Each block:
   - Runs self-attention.
   - Uses `TaskTypeClassifier` to bias MoE routing.
   - Invokes `MixtureOfExperts` with expert-specific kwargs.
4. Experts:
   - LanguageExpert: Turkish morpho/semantic + AgglutinativeAttention
   - Logic, Spatial, Causal experts
   - NeuroSymbolicExpert: neuro-symbolic reasoning over relations/graphs
5. Final normalization + projection to vocab logits.

This is a single coherent graph; no parallel/legacy language stack is used in training.

---

## 2. AGIFORMER Core

Location:
- [`agiformer/model.py`](agiformer/model.py:142)

Constructor (key arguments):

- `tokenizer: MorphoPiece`
- `d_model, n_layers, n_heads, d_ff`
- `n_experts, expert_types`
- `memory_size`
- `max_seq_len`
- `dropout`
- Feature toggles:
  - `use_linear_attention`
  - `use_memory`
  - `use_introspection`
  - `use_multimodal`
  - `use_agglutinative_attention`
  - `use_gradient_checkpointing`

Core components:

- Token embedding:
  - `nn.Embedding(vocab_size, d_model)`
  - input_ids validated against `vocab_size`; violations raise `ValueError`.
- Optional:
  - `MultimodalPerceptionCore` for image/audio/video.
  - `UnifiedMemoryBackbone` for working + long-term memory.
- `GlobalKnowledgeGraph`:
  - Shared across blocks and experts for relational reasoning.
- Blocks:
  - `self.blocks: nn.ModuleList[AGIFORMERBlock]`
- Output:
  - `LayerNorm` + `Linear` to vocab size.

Forward (simplified):

```python
logits, info = model(
    input_ids,
    attention_mask=attention_mask,
    morpho_types=morpho_types,
    semantic_categories=semantic_categories,
    image=image,
)
```

- Applies memory (if enabled).
- Iterates blocks with optional gradient checkpointing.
- Collects per-block diagnostics into `info["blocks"]`.

---

## 3. AGIFORMERBlock and MoE

Location:
- [`agiformer/model.py`](agiformer/model.py:26)
- [`agiformer/experts/moe.py`](agiformer/experts/moe.py:120)

### 3.1 Block Structure

Each `AGIFORMERBlock`:

- Self-attention:
  - `MultiHeadAttention` or `LinearAttention`
- Task-aware routing:
  - `TaskTypeClassifier(d_model)` predicts coarse domains.
  - Domain scores bias MoE routing (`routing_bias`).
- Mixture-of-Experts:
  - `MixtureOfExperts` with:
    - configurable `n_experts`, `k`
    - optional custom experts:
      - LanguageExpert
      - LogicExpert
      - SpatialExpert
      - CausalExpert
      - NeuroSymbolicExpert
- Optional:
  - `IntrospectionLoop` in higher blocks for self-modeling.

### 3.2 MixtureOfExperts Implementation

`MixtureOfExperts.forward`:

```python
output, expert_info = moe(
    hidden_states,
    routing_bias=routing_bias,
    attention_mask=attention_mask,
    morpho_types=morpho_types,
    semantic_categories=semantic_categories,
)
```

Key behaviors:

- Uses `ExpertRouter`:
  - Computes routing logits per token.
  - Applies optional `routing_bias`.
  - Selects top-k experts.
  - Computes load-balancing loss encouraging uniform usage.
- For each expert:
  - Calls `expert(hidden_states, **expert_kwargs)`.
  - Experts may return `(out, info)`; info aggregated.
- Combines outputs:
  - Gathers selected experts.
  - Applies normalized top-k weights.
  - Residual: `output + hidden_states`.

Design:
- All experts share the same hidden state space.
- Extra annotations (mask/morpho/semantic) are passed explicitly via kwargs.
- Non-language experts ignore language-specific kwargs.

---

## 4. Turkish Language Integration

### 4.1 LanguageExpert (Primary Path)

Location:
- [`agiformer/experts/language_expert.py`](agiformer/experts/language_expert.py:1)

Key features:

- Attention:
  - `AgglutinativeAttention` (default) for Turkish:
    - Verb/root/suffix biases.
  - Or fallback to standard `MultiHeadAttention`.
- Morphological and semantic embeddings:
  - `NUM_MORPHEME_TYPES = 23`
  - `NUM_SEMANTIC_CATEGORIES = 12`
  - When both `morpho_types` and `semantic_categories` are provided:
    - Clamps to valid ranges.
    - Embeds and adds to hidden states.
- Fully integrated:
  - Consumes kwargs from `MixtureOfExperts`:
    - `attention_mask`, `morpho_types`, `semantic_categories`.

Result:
- Former standalone `TMA1Model` is superseded.
- All Turkish-specific modeling is part of the main AGIFORMER + MoE flow.

### 4.2 Language Utilities

Location:
- [`agiformer/language/__init__.py`](agiformer/language/__init__.py:1)

Exports:

- `MorphoSplitter` (Regex-based)
- `MorphoPiece`
- `GrammarEngine`
- `AgglutinativeAttention`

`TMA1Model` is intentionally removed from the public API to match the integrated implementation.

---

## 5. Neuro-Symbolic Reasoning

Location:
- [`agiformer/experts/neuro_symbolic_expert.py`](agiformer/experts/neuro_symbolic_expert.py:33)
- [`agiformer/experts/knowledge_graph.py`](agiformer/experts/knowledge_graph.py)

Key steps:

1. Self-attention over token representations.
2. Vectorized edge extraction:
   - From attention matrices:
     - `nonzero` + thresholding + self-loop removal.
3. Relation classification:
   - Predict relation types between concept pairs.
4. Filtering:
   - Drop edges classified as `NONE`.
5. Knowledge graph propagation:
   - Calls global/dynamic knowledge graph module.
6. Projection:
   - Maps reasoned concepts back into the model space.
   - Residual integration into block outputs.

Design goals:

- No Python nested loops in the critical path.
- Differentiable, GPU-friendly neuro-symbolic layer.

---

## 6. Memory & Introspection

### 6.1 Unified Memory Backbone

Location:
- [`agiformer/core/memory_backbone.py`](agiformer/core/memory_backbone.py)

Capabilities:

- Working memory:
  - Short-range context retention.
- Long-term memory:
  - Persistent representations for global knowledge.

Integrated in:
- `AGIFORMER.forward` immediately after token embeddings.

### 6.2 Introspection

Location:
- [`agiformer/introspection/self_model.py`](agiformer/introspection/self_model.py)
- [`agiformer/introspection/meta_learning.py`](agiformer/introspection/meta_learning.py)

Function:

- `IntrospectionLoop`:
  - Applied in selected blocks (typically final).
  - Consumes previous hidden states.
  - Enables self-evaluation and adjustment mechanisms.

---

## 7. Multimodal Perception

Location:
- [`agiformer/core/multimodal_perception.py`](agiformer/core/multimodal_perception.py)

Features:

- Ingests:
  - Text + visual (and extensible to other modalities).
- Produces:
  - Aligned embeddings fused into the AGIFORMER backbone.

Controlled by:
- `use_multimodal` flag in model config.

---

## 8. Data Pipeline & Training Semantics

### 8.1 Datasets

Location:
- [`agiformer/datasets/text_datasets.py`](agiformer/datasets/text_datasets.py)
- [`agiformer/datasets/cc_datasets.py`](agiformer/datasets/cc_datasets.py)
- Fallbacks for simple/legacy scenarios.

Important properties:

- Turkish JSONL:
  - Expected fields:
    - `tokens`
    - `morpho_types`
    - `semantic_categories`
  - Dataset outputs:
    - `input_ids`, `target_ids`
    - `attention_mask`
    - `morpho_types`, `semantic_categories` when present.
- No silent ID clamping at the end of pipeline:
  - Errors surface early.
- SimpleTextDataset:
  - For quick char-based tests.

### 8.2 Training Loop

Location:
- [`train.py`](train.py:211)

Core invariants (enforced):

- Loss:
  - `main_loss = CrossEntropyLoss(ignore_index=0)`
  - `aux_loss = Σ(load_balancing_loss)` from MoE (if any)
  - `total_loss = main_loss + aux_loss`
  - Backprop on `total_loss`
- Logging:
  - Training:
    - `Training/main_loss`
    - `Training/aux_lb_loss`
  - Validation:
    - `Validation/main_loss` only
- Checkpoints:
  - Selection based on validation main_loss (not polluted by aux terms).

Batch preparation:
- `prepare_batch_data` normalizes all inputs to `AGIFORMER.forward` signature consistently.

---

## 9. How to Run

Example (Hydra-based):

```bash
python train.py experiment=phase1_baseline
```

Override examples:

```bash
python train.py \
  experiment=phase1_lite \
  training.max_steps=1000 \
  training.batch_size=8 \
  logging.use_wandb=false
```

Expected:
- Model builds successfully.
- Steps log comparable train/val losses.
- MoE aux losses tracked separately.
- If Turkish JSONL with morpho/semantic fields is provided, LanguageExpert uses them.

---

## 10. Development Guidelines

- Keep docs in `docs/` consistent with:
  - [`README.md`](README.md)
  - Active code under `agiformer/`
  - Training entrypoint [`train.py`](train.py)
- When adding experts:
  - Respect `MixtureOfExperts` contract:
    - `(hidden_states, **kwargs) -> output` or `(output, info)`.
- When extending language features:
  - Integrate via:
    - Datasets → batch → `AGIFORMER.forward` → MoE → LanguageExpert.
  - Do not introduce parallel, disconnected model stacks.

AGIFORMER is designed as a single, auditable architecture where Turkish-specific language modeling, MoE-based specialization, memory, introspection, neuro-symbolic reasoning, and multimodal perception operate cohesively inside one training and deployment graph.