# Developer: inkbytefo
# Modified: 2025-11-07

"""
Professional Training Script for AGIFORMER with Hydra Configuration
Unified training script that consolidates logic from all previous training scripts
Supports flexible configuration management and experiment tracking
"""

import torch
import torch.nn as nn
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, random_split

# Hydra imports
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd

sys.path.insert(0, str(Path(__file__).parent))
from agiformer import AGIFORMER
from agiformer.utils import count_parameters, format_number, WarmupScheduler
from agiformer.datasets import CC12MDataset
from agiformer.language.tokenizer import MorphoPiece
from agiformer.datasets import TurkishTextDataset, TextDataset, SimpleTextDataset, create_dataloader
from agiformer.experts.pseudo_labeler import PseudoLabeler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_val_loss = float('inf')
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                       scheduler: Optional[Any], metrics: Dict[str, float], is_best: bool = False) -> str:
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        checkpoint = {
            'step': step, 
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 
            'metrics': metrics
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
            self.logger.info(f"Best model saved to {self.checkpoint_dir / 'best_model.pt'}")
        self._cleanup_old_checkpoints()
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu')
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(list(self.checkpoint_dir.glob("checkpoint_*.pt")), key=lambda x: x.stat().st_mtime, reverse=True)
        for old in checkpoints[self.keep_last_n:]:
            old.unlink()
            self.logger.info(f"Removed old checkpoint: {old}")


class MetricsLogger:
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        if WANDB_AVAILABLE and config.get('use_wandb', False):
            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger = wandb
        else:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)

    def log_training_metrics(self, step: int, loss: float, model_info: Dict[str, Any]):
        metrics = {'Training/loss': loss, 'Training/step': step}
        if isinstance(self.logger, logging.Logger):
            self.logger.info(f"Step {step}: {metrics}")
        else:
            self.logger.log(metrics)

    def log_validation_metrics(self, step: int, val_loss: float, val_metrics: Dict[str, float]):
        metrics = {'Validation/loss': val_loss, 'Validation/step': step, **val_metrics}
        if isinstance(self.logger, logging.Logger):
            self.logger.info(f"Validation Step {step}: {metrics}")
        else:
            self.logger.log(metrics)

    def finish(self):
        if hasattr(self.logger, 'finish'):
            self.logger.finish()


def prepare_batch_data(batch, device, is_multimodal):
    """
    Normalize batch into AGIFORMER.forward(**model_inputs), target_ids.

    Supports:
    - Multimodal CC datasets
    - TurkishTextDataset JSONL with morpho_types / semantic_categories
    - Legacy (input_ids, target_ids) tuples
    """
    if is_multimodal:
        model_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch.get("attention_mask", None).to(device)
            if batch.get("attention_mask") is not None
            else None,
            "image": batch["image"].to(device),
        }
        if "morpho_types" in batch:
            model_inputs["morpho_types"] = batch["morpho_types"].to(device)
        if "semantic_categories" in batch:
            model_inputs["semantic_categories"] = batch["semantic_categories"].to(device)

        target_ids = batch["target_ids"].to(device)
        return model_inputs, target_ids

    # Text-only paths
    if isinstance(batch, dict):
        model_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch.get("attention_mask", None).to(device)
            if batch.get("attention_mask") is not None
            else None,
        }
        if "morpho_types" in batch:
            model_inputs["morpho_types"] = batch["morpho_types"].to(device)
        if "semantic_categories" in batch:
            model_inputs["semantic_categories"] = batch["semantic_categories"].to(device)

        target_ids = batch["target_ids"].to(device)
        return model_inputs, target_ids

    # Legacy (input_ids, target_ids)
    input_ids, target_ids = batch
    return {"input_ids": input_ids.to(device)}, target_ids.to(device)


def validate_epoch(model, dataloader, criterion, device, use_amp, is_multimodal):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(**model_inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# This function is no longer needed as its logic is integrated into the main loop.


def create_dataset(data_dir, data_path, model_config, train_split, tokenizer=None) -> Tuple[Dataset, Dataset, bool]:
    # Check for CC12M multimodal dataset first
    if data_dir and Path(data_dir).exists():
        train_metadata, val_metadata = Path(data_dir)/"metadata_train.json", Path(data_dir)/"metadata_val.json"
        if train_metadata.exists() and val_metadata.exists():
            print("Found CC12M dataset structure")
            train_ds = CC12MDataset(data_dir, "train", max_text_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
            val_ds = CC12MDataset(data_dir, "val", max_text_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
            return train_ds, val_ds, True

    # Check for Turkish text dataset (JSONL with morpho_types)
    if data_path and Path(data_path).exists() and Path(data_path).suffix == '.jsonl':
        print(f"Loading Turkish text dataset from: {data_path}")

        # <-- DEĞİŞİKLİK: vocab_size'ı tokenizer'dan alarak ilet
        dataset_vocab_size = tokenizer.vocab_size if tokenizer else model_config.get('vocab_size', 32000)

        train_dataset = TurkishTextDataset(
            corpus_file=data_path,
            tokenizer=tokenizer,
            max_seq_len=model_config['max_seq_len'],
            is_jsonl=True,
            vocab_size=dataset_vocab_size # <-- BURAYI GÜNCELLE
        )
        
        # Split into train and validation
        total_size = len(train_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
        return train_ds, val_ds, False

    # Check for plain text dataset
    if data_path and Path(data_path).exists() and Path(data_path).suffix == '.txt':
        print(f"Loading text dataset from: {data_path}")
        if tokenizer:
            train_dataset = TextDataset(
                file_path=data_path,
                tokenizer=tokenizer,
                max_seq_len=model_config['max_seq_len'],
                use_morpho=model_config.get('morphological_analysis', True)
            )
        else:
            # Fallback to SimpleTextDataset
            with open(data_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            train_dataset = SimpleTextDataset(texts, max_seq_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
        
        # Split into train and validation
        total_size = len(train_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
        return train_ds, val_ds, False

    # Fallback to simple text dataset
    print("Using fallback text-only dataset")
    dummy_texts = [
        "Bu bir eğitim metnidir.", "Hızlı kahverengi tilki tembel köpeğin üstünden atlar.",
        "Makine öğrenmesi büyüleyicidir.", "Derin öğrenme modelleri büyük veri kümeleri gerektirir.",
        "Doğal dil işleme hızla gelişmektedir.", "Transformer'lar AI araştırmalarını devrimleştirdi.",
        "Dikkat mekanizmaları güçlüdür.", "AGIFORMER AI'nın geleceğidir."
    ] * 1000
    dataset = SimpleTextDataset(dummy_texts, max_seq_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset, False


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration management
    """
    # Set up logging
    logging.basicConfig(level=getattr(logging, cfg.logging.console_level))
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

    # Set up hardware with safe allocator config handling
    alloc_conf = None
    if hasattr(cfg.hardware, "pytorch_cuda_alloc_conf") and cfg.hardware.pytorch_cuda_alloc_conf:
        alloc_conf = str(cfg.hardware.pytorch_cuda_alloc_conf).strip()
    env_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")

    # Log incoming allocator settings before applying
    logger.info(
        f"Allocator config - from cfg: {alloc_conf!r}, "
        f"from env: {env_alloc_conf!r}"
    )

    def _is_valid_alloc_conf(value: str) -> bool:
        # Minimal defensive validation to avoid triggering known PyTorch parser bugs.
        # Accept simple tokens like "expandable_segments" or key:value pairs
        # without trailing commas or malformed segments.
        v = (value or "").strip()
        if not v:
            return False
        # quick reject for obviously broken patterns
        if ",," in v or v.endswith(",") or "==" in v:
            return False
        # allow single token like "expandable_segments"
        if ":" not in v and "," not in v:
            return True
        # for comma-separated key:value segments, require each to contain ':'
        for seg in v.split(","):
            seg = seg.strip()
            if not seg:
                return False
            if ":" not in seg:
                return False
        return True

    effective_alloc_conf = env_alloc_conf or alloc_conf
    if effective_alloc_conf and _is_valid_alloc_conf(effective_alloc_conf):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = effective_alloc_conf
        logger.info(
            f"Using PYTORCH_CUDA_ALLOC_CONF={effective_alloc_conf!r} "
            f"(source={'env' if env_alloc_conf else 'cfg'})"
        )
    elif effective_alloc_conf:
        # Invalid config detected; do NOT propagate to PyTorch to avoid crashes
        logger.warning(
            f"Ignoring invalid PYTORCH_CUDA_ALLOC_CONF={effective_alloc_conf!r} "
            f"(source={'env' if env_alloc_conf else 'cfg'}) to prevent allocator crash"
        )
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        logger.info("No PYTORCH_CUDA_ALLOC_CONF configured; using PyTorch defaults")

    # Determine device
    if cfg.hardware.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.hardware.device)

    logger.info(f"Using device: {device}")

    # Defensive CUDA allocator + device init:
    # If PYTORCH_CUDA_ALLOC_CONF is malformed for this PyTorch build, trigger it here
    # and fall back cleanly instead of crashing later in training.
    if torch.cuda.is_available() and device.type == "cuda":
        final_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        logger.info(f"Final PYTORCH_CUDA_ALLOC_CONF seen by PyTorch: {final_alloc!r}")
        try:
            # Force lazy init; allocator parse errors will surface here.
            _ = torch.cuda.device_count()
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        except Exception as e:
            logger.error(
                f"CUDA initialization failed with PYTORCH_CUDA_ALLOC_CONF={final_alloc!r}; "
                f"clearing it and falling back to CPU. Error: {e}"
            )
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            device = torch.device("cpu")
            logger.warning("Using CPU due to CUDA allocator/device initialization failure")
    else:
        logger.info("CUDA not available or not selected; running on CPU")

    # Initialize Weights & Biases if enabled
    if cfg.logging.use_wandb:
        if WANDB_AVAILABLE:
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=cfg.logging.wandb_tags,
                notes=cfg.logging.wandb_notes
            )
            logger.info(f"Initialized W&B run: {cfg.run_name}")
        else:
            logger.warning("W&B requested but not available")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / cfg.training.checkpoint_dir),
        keep_last_n=cfg.training.keep_last_n_checkpoints
    )

    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        project_name=cfg.logging.wandb_project if cfg.logging.use_wandb else "agiformer",
        experiment_name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Load tokenizer
    logger.info("Loading MorphoPiece tokenizer...")
    tokenizer_path = getattr(cfg.model, 'tokenizer_path', 'tokenizer/morphopiece.model')
    vocab_size = getattr(cfg.model, 'vocab_size', 256)
    if os.path.exists(tokenizer_path):
        tokenizer = MorphoPiece(tokenizer_path)
        logger.info(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")
    else:
        logger.warning(f"Tokenizer not found at {tokenizer_path}, creating default tokenizer")
        tokenizer = MorphoPiece()  # Create tokenizer without loading file
        tokenizer.vocab_size = vocab_size
        logger.info(f"Created default tokenizer with vocab size: {vocab_size}")

    # Create model
    logger.info("Creating AGIFORMER model...")
    model = AGIFORMER(
        tokenizer=tokenizer,
        use_gradient_checkpointing=cfg.training.use_gradient_checkpointing,
        **{k: v for k, v in cfg.model.items() if k not in ['tokenizer_path', 'vocab_size', 'use_gradient_checkpointing', 'morphological_analysis', 'language_expert']}
    ).to(device)

    # Log model information
    params = count_parameters(model)
    logger.info(f"Model Parameters: Total: {format_number(params['total'])}, Trainable: {format_number(params['trainable'])}")

    # Create datasets
    logger.info("Creating datasets...")
    train_ds, val_ds, is_multimodal = create_dataset(
        cfg.data.data_dir,
        getattr(cfg.data, 'data_path', None),
        OmegaConf.to_container(cfg.model),
        cfg.data.train_split,
        tokenizer
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=False
    )

    logger.info(f"Dataset: {len(train_ds)} train samples, {len(val_ds)} val samples")
    logger.info(f"Multimodal: {is_multimodal}")

    # Create optimizer
    if cfg.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            eps=cfg.training.adam_epsilon
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")

    # Create scheduler
    scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=cfg.training.warmup_steps,
        d_model=cfg.model.d_model
    )

    # Create loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Initialize PseudoLabeler for self-supervised relation learning
    pseudo_labeler = PseudoLabeler()
    logger.info("Initialized PseudoLabeler for self-supervised relation learning")

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Batch size: {cfg.training.batch_size}, LR: {cfg.training.learning_rate}, AMP: {cfg.training.use_amp}")

    best_val_loss = float('inf')
    global_step = 0

    try:
        scaler = torch.amp.GradScaler(device.type, enabled=cfg.training.use_amp)
        for epoch in range(cfg.training.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                global_step += 1

                model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)

                with torch.amp.autocast(device_type=device.type, enabled=cfg.training.use_amp):
                    logits, info = model(**model_inputs)

                    # Primary supervised loss (this is the ONLY loss used for train/val comparison)
                    main_loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )

                    # Optional: aggregate auxiliary losses from MoE/introspection if exposed in info
                    aux_loss = 0.0
                    if isinstance(info, dict):
                        # Example: sum load_balancing_loss across blocks if present
                        blocks = info.get('blocks', [])
                        lb_losses = []
                        for b in blocks:
                            moe_info = b.get('moe', {}) if isinstance(b, dict) else {}
                            if 'load_balancing_loss' in moe_info:
                                lb_losses.append(moe_info['load_balancing_loss'])
                        if lb_losses:
                            aux_loss = sum(lb_losses)

                    total_loss = main_loss + aux_loss

                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Log only the primary supervised loss for comparability with validation
                logger.info(
                    f"Step {global_step}/{cfg.training.max_steps}, "
                    f"Train/main_loss: {main_loss.item():.4f}, Aux/lb_loss: {float(aux_loss):.4f}"
                )

                if metrics_logger and global_step % cfg.training.log_interval == 0:
                    train_metrics = {
                        "Training/main_loss": float(main_loss),
                        "Training/aux_lb_loss": float(aux_loss),
                    }
                    # Preserve original behavior but ensure standardized keys
                    metrics_logger.log_training_metrics(global_step, float(main_loss), train_metrics)

                if global_step % cfg.training.eval_interval == 0:
                    val_loss = validate_epoch(
                        model,
                        val_loader,
                        criterion,
                        device,
                        cfg.training.use_amp,
                        is_multimodal
                    )
                    model.train()

                    if metrics_logger:
                        # Log validation main loss only
                        metrics_logger.log_validation_metrics(
                            global_step,
                            float(val_loss),
                            {"Validation/main_loss": float(val_loss)}
                        )

                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss

                    logger.info(
                        f"Step {global_step}: Validation/main_loss: {val_loss:.4f} "
                        f"{'(Best)' if is_best else ''}"
                    )

                    if is_best:
                        checkpoint_manager.save_checkpoint(
                            global_step,
                            model,
                            optimizer,
                            scheduler,
                            {"val_loss": float(val_loss), "epoch": epoch},
                            is_best=True,
                        )

                if global_step % cfg.training.save_interval == 0:
                    checkpoint_manager.save_checkpoint(
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        {"val_loss": float(best_val_loss), "epoch": epoch},
                        is_best=False,
                    )

                if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                    logger.info(
                        f"Reached max_steps ({cfg.training.max_steps}). Stopping training."
                    )
                    break

            if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                break

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if metrics_logger:
            metrics_logger.finish()
        logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

        # Save final config
        final_config_path = output_dir / "final_config.yaml"
        with open(final_config_path, 'w') as f:
            OmegaConf.save(cfg, f)
        logger.info(f"Final config saved to: {final_config_path}")


if __name__ == "__main__":
    main()
