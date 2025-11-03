## Developer: inkbytefo
## Modified: 2025-11-03
"""
Professional Training Script for AGIFORMER
*** UPDATED WITH ROBUST ARGUMENT PARSING AND GRADIENT CHECKPOINTING LOGIC ***
"""

import torch, torch.nn as nn, yaml, argparse, json, os, sys, logging
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
from agiformer import AGIFORMER
from agiformer.utils import count_parameters, format_number, WarmupScheduler
from scripts.prepare_cc12m import CC12MDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# (CheckpointManager, MetricsLogger, SimpleTextDataset sınıfları aynı kalabilir)
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir); self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n; self.best_val_loss = float('inf')
        logging.basicConfig(level=logging.INFO); self.logger = logging.getLogger(__name__)
    def save_checkpoint(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], metrics: Dict[str, float], is_best: bool = False) -> str:
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:08d}.pt"
        checkpoint = {'step': step, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'metrics': metrics}
        torch.save(checkpoint, checkpoint_path); self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pt"); self.logger.info(f"Best model saved to {self.checkpoint_dir / 'best_model.pt'}")
        self._cleanup_old_checkpoints()
        return str(checkpoint_path)
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        checkpoint = torch.load(Path(checkpoint_path), map_location='cpu'); self.logger.info(f"Checkpoint loaded from {checkpoint_path}"); return checkpoint
    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(list(self.checkpoint_dir.glob("checkpoint_*.pt")), key=lambda x: x.stat().st_mtime, reverse=True)
        for old in checkpoints[self.keep_last_n:]: old.unlink(); self.logger.info(f"Removed old checkpoint: {old}")

class MetricsLogger:
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any]):
        if WANDB_AVAILABLE and config.get('use_wandb', False):
            wandb.init(project=project_name, name=experiment_name, config=config); self.logger = wandb
        else: self.logger = logging.getLogger(__name__); logging.basicConfig(level=logging.INFO)
    def log_training_metrics(self, step: int, loss: float, model_info: Dict[str, Any]):
        metrics = {'Training/loss': loss, 'Training/step': step}
        # (metrik toplama mantığı aynı)
        if isinstance(self.logger, logging.Logger): self.logger.info(f"Step {step}: {metrics}")
        else: self.logger.log(metrics)
    def log_validation_metrics(self, step: int, val_loss: float, val_metrics: Dict[str, float]):
        metrics = {'Validation/loss': val_loss, 'Validation/step': step, **val_metrics}
        if isinstance(self.logger, logging.Logger): self.logger.info(f"Validation Step {step}: {metrics}")
        else: self.logger.log(metrics)
    def finish(self):
        if not isinstance(self.logger, logging.Logger): self.logger.finish()

class SimpleTextDataset(Dataset):
    def __init__(self, texts, max_seq_len=512, vocab_size=256):
        self.texts, self.max_seq_len, self.vocab_size = texts, max_seq_len, vocab_size
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        char_ids = [ord(c) % self.vocab_size for c in self.texts[idx][:self.max_seq_len]]
        padding = [0] * (self.max_seq_len - len(char_ids)); char_ids.extend(padding)
        input_ids = torch.tensor(char_ids[:-1], dtype=torch.long); target_ids = torch.tensor(char_ids[1:], dtype=torch.long)
        return input_ids, target_ids

def prepare_batch_data(batch, device, is_multimodal):
    if is_multimodal:
        return {'text': batch['input_ids'].to(device), 'image': batch['image'].to(device)}, batch['target_ids'].to(device)
    else:
        input_ids, target_ids = batch; return {'text': input_ids.to(device)}, target_ids.to(device)

def validate_epoch(model, dataloader, criterion, device, use_amp, is_multimodal):
    model.eval(); total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(**model_inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_epoch(model, dataloader, optimizer, criterion, device, use_amp, metrics_logger, step_offset, is_multimodal):
    model.train(); total_loss = 0
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=use_amp)
    for batch_idx, batch in enumerate(dataloader):
        model_inputs, target_ids = prepare_batch_data(batch, device, is_multimodal)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, info = model(**model_inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            total_loss_batch = loss
            for block_info in info.get('blocks', []):
                if 'moe' in block_info and 'load_balancing_loss' in block_info['moe']['router_info']:
                    total_loss_batch += block_info['moe']['router_info']['load_balancing_loss']

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        current_step = step_offset + batch_idx
        if metrics_logger and batch_idx % 10 == 0: metrics_logger.log_training_metrics(current_step, loss.item(), info)
        if batch_idx % 50 == 0:
            if torch.cuda.is_available(): print(f"📊 GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
            print(f"🚀 Batch {batch_idx}, Loss: {loss.item():.4f}, Step: {current_step}")

    return total_loss / len(dataloader), current_step + 1

def create_dataset(data_dir, data_path, model_config, train_split):
    # (Bu fonksiyon aynı kalabilir)
    if data_dir and Path(data_dir).exists():
        train_metadata, val_metadata = Path(data_dir)/"metadata_train.json", Path(data_dir)/"metadata_val.json"
        if train_metadata.exists() and val_metadata.exists():
            print("Found CC12M dataset structure")
            train_ds = CC12MDataset(data_dir, "train", max_text_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
            val_ds = CC12MDataset(data_dir, "val", max_text_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
            return train_ds, val_ds, True
    print("Using fallback text-only dataset")
    dummy_texts = ["Sample text for training."] * 8000
    dataset = SimpleTextDataset(dummy_texts, max_seq_len=model_config['max_seq_len'], vocab_size=model_config['vocab_size'])
    train_size = int(train_split * len(dataset)); val_size = len(dataset) - train_size
    return list(random_split(dataset, [train_size, val_size])) + [False]

def main():
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("🔧 Set PYTORCH_CUDA_ALLOC_CONF for better memory management")

    parser = argparse.ArgumentParser(description="AGIFORMER Training Script")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args, unknown_args = parser.parse_known_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)

    i = 0
    while i < len(unknown_args):
        arg, val_str = unknown_args[i].strip('--'), unknown_args[i+1]
        try:
            val = int(val_str) if '.' not in val_str else float(val_str)
        except ValueError:
            val = True if val_str.lower() == 'true' else False if val_str.lower() == 'false' else val_str
        keys = arg.split('.'); d = config
        for key in keys[:-1]: d = d.setdefault(key, {})
        d[keys[-1]] = val
        i += 2

    print("Updated config:", json.dumps(config, indent=2))
    model_config, train_config = config['model'], config['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
    metrics_logger = MetricsLogger(project_name="agiformer", experiment_name=config.get('experiment_name', 'agiformer_run'), config=config)
    
    # --- DEĞİŞİKLİK: use_gradient_checkpointing'i model'e ilet ---
    model = AGIFORMER(
        use_gradient_checkpointing=train_config.get('use_gradient_checkpointing', False),
        **model_config
    ).to(device)
    # --- BİTTİ ---

    params = count_parameters(model)
    print(f"\nModel Parameters: Total: {format_number(params['total'])}, Trainable: {format_number(params['trainable'])}")

    train_ds, val_ds, is_multimodal = create_dataset(config.get('data_dir'), config.get('data_path'), model_config, 0.9)
    train_loader = DataLoader(train_ds, batch_size=train_config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_config['batch_size'], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
    scheduler = WarmupScheduler(optimizer, warmup_steps=train_config['warmup_steps'], d_model=model_config['d_model'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    start_step, best_val_loss = 0, float('inf')
    print(f"\n🔥 Training Started! Batch size: {train_config['batch_size']}, LR: {train_config['learning_rate']}, AMP: {train_config['use_amp']}")
    
    for epoch in range(train_config.get('epochs', 10)):
        print(f"\n📅 Epoch {epoch + 1}/{train_config.get('epochs', 10)}")
        avg_loss, end_step = train_epoch(model, train_loader, optimizer, criterion, device, train_config['use_amp'], metrics_logger, start_step, is_multimodal)
        val_loss = validate_epoch(model, val_loader, criterion, device, train_config['use_amp'], is_multimodal)
        scheduler.step()
        if metrics_logger: metrics_logger.log_validation_metrics(end_step, val_loss, {})
        
        is_best = val_loss < best_val_loss
        if is_best: best_val_loss = val_loss
        checkpoint_manager.save_checkpoint(end_step, model, optimizer, scheduler, {'val_loss': val_loss}, is_best)
        
        print(f"Epoch {epoch + 1} completed: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        start_step = end_step

    if metrics_logger: metrics_logger.finish()
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
