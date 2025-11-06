# Developer: inkbytefo
# Modified: 2025-11-06

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
from agiformer.data.dataset import TurkishTextDataset, TextDataset, SimpleTextDataset, create_dataloader
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
    if is_multimodal:
        return {
            'input_ids': batch['input_ids'].to(device),
            'morpho_types': batch.get('morpho_types', None),
            'image': batch['image'].to(device)
        }, batch['target_ids'].to(device)
    else:
        if isinstance(batch, dict):
            # JSONL format with morpho_types
            return {
                'input_ids': batch['input_ids'].to(device),
                'morpho_types': batch.get('morpho_types', None)
            }, batch['target_ids'].to(device)
        else:
            # Legacy format
            input_ids, target_ids = batch
            return {'input_ids': input_ids.to(device)}, target_ids.to(device)


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


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp, metrics_logger, 
               step_offset, is_multimodal, pseudo_labeler=None):
    model.train()
    total_loss = 0
    try:
        scaler = torch.amp.GradScaler(enabled=use_amp)
    except AttributeError:
        # Fallback for older PyTorch versions
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch_idx, batch in enumerate(dataloader):
        model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, info = model(**model_inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss_batch = loss
            
            # Add MoE load balancing loss
            for block_info in info.get('blocks', []):
                if 'moe' in block_info and 'load_balancing_loss' in block_info['moe']['router_info']:
                    total_loss_batch += block_info['moe']['router_info']['load_balancing_loss'].detach()

            # Add relation and task losses with pseudo-labeler
            relation_loss = 0.0
            task_loss = 0.0
            if pseudo_labeler and info.get('blocks'):
                for block_info in info['blocks']:
                    # NeuroSymbolicExpert relation loss
                    if 'experts' in block_info:
                        for expert_name, expert_info in block_info['experts'].items():
                            if expert_name == 'neuro_symbolic' and 'expert_info' in expert_info:
                                ns_info = expert_info['expert_info']
                                if 'relation_logits' in ns_info and 'classified_edges' in ns_info:
                                    rel_logits = ns_info['relation_logits']
                                    rel_edges = ns_info['classified_edges']

                                    # Generate pseudo-labels (simplified)
                                    if hasattr(model, 'tokenizer') and model.tokenizer:
                                        try:
                                            batch_tokens = []
                                            for seq in model_inputs['input_ids'][:1]:
                                                tokens = []
                                                for token_id in seq:
                                                    if token_id.item() < model.tokenizer.vocab_size:
                                                        tokens.append(str(token_id.item()))
                                                    else:
                                                        tokens.append("[UNK]")
                                                batch_tokens.append(tokens)
                                            if batch_tokens:
                                                pseudo_labels = pseudo_labeler.generate_labels(
                                                    batch_tokens[0],
                                                    torch.randn(len(batch_tokens[0]), model.d_model).to(device)
                                                )

                                                target_labels = []
                                                for edge in rel_edges:
                                                    if tuple(edge) in pseudo_labels:
                                                        target_labels.append(pseudo_labels[tuple(edge)])
                                                    else:
                                                        target_labels.append(0)

                                                if target_labels:
                                                    target_labels = torch.tensor(target_labels, device=device)
                                                    rel_criterion = nn.CrossEntropyLoss()
                                                    relation_loss += rel_criterion(rel_logits, target_labels)
                                        except Exception:
                                            pass

                    # Task classification loss
                    if 'moe' in block_info and 'task_logits' in block_info['moe']:
                        task_logits = block_info['moe']['task_logits']
                        try:
                            batch_tokens = []
                            for seq in model_inputs['input_ids'][:1]:
                                tokens = []
                                for token_id in seq:
                                    if token_id.item() < model.tokenizer.vocab_size:
                                        tokens.append(str(token_id.item()))
                                    else:
                                        tokens.append("[UNK]")
                                batch_tokens.append(tokens)

                            if batch_tokens:
                                task_label = pseudo_labeler.classify_task_type(batch_tokens[0])
                                task_labels = torch.tensor([task_label], device=device)

                                task_criterion = nn.CrossEntropyLoss()
                                task_loss += task_criterion(task_logits[:1], task_labels)
                        except Exception:
                            pass

            # Add auxiliary losses to main loss
            total_loss_batch += relation_loss * 0.1
            total_loss_batch += task_loss * 0.05

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        current_step = step_offset + batch_idx
        if metrics_logger and batch_idx % 10 == 0:
            metrics_logger.log_training_metrics(current_step, loss.item(), info)
        if batch_idx % 50 == 0:
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Step: {current_step}")

    return total_loss / len(dataloader), current_step + 1


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
        train_dataset = TurkishTextDataset(
            corpus_file=data_path,
            tokenizer=tokenizer,
            max_seq_len=model_config['max_seq_len'],
            is_jsonl=True
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

    # Set up hardware
    if cfg.hardware.pytorch_cuda_alloc_conf:
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = cfg.hardware.pytorch_cuda_alloc_conf
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF for better memory management")

    # Determine device
    if cfg.hardware.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.hardware.device)

    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")

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
        **{k: v for k, v in cfg.model.items() if k not in ['tokenizer_path', 'vocab_size']}
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
        persistent_workers=cfg.data.persistent_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers
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
        for epoch in range(cfg.training.epochs):
            if cfg.training.max_steps and global_step >= cfg.training.max_steps:
                logger.info(f"Reached max_steps ({cfg.training.max_steps}). Stopping training.")
                break

            logger.info(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
            avg_loss, end_step = train_epoch(
                model, train_loader, optimizer, criterion, device,
                cfg.training.use_amp, metrics_logger, global_step, is_multimodal, pseudo_labeler
            )
            global_step = end_step

            # Validation
            val_loss = validate_epoch(model, val_loader, criterion, device, cfg.training.use_amp, is_multimodal)
            scheduler.step()

            if metrics_logger:
                metrics_logger.log_validation_metrics(end_step, val_loss, {})

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            checkpoint_manager.save_checkpoint(
                end_step, model, optimizer, scheduler,
                {'val_loss': val_loss, 'epoch': epoch}, is_best
            )

            logger.info(f"Epoch {epoch + 1} completed: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

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
