"""
Inference Example for AGIFORMER
"""

import torch
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agiformer import AGIFORMER

def text_to_ids(text: str, vocab_size: int = 256) -> torch.Tensor:
    """Convert text to character IDs"""
    char_ids = [ord(c) % vocab_size for c in text]
    return torch.tensor([char_ids], dtype=torch.long)

def ids_to_text(ids: torch.Tensor, vocab_size: int = 256) -> str:
    """Convert character IDs to text"""
    return ''.join([chr(c % vocab_size) for c in ids.squeeze().cpu().numpy()])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- DEĞİŞİKLİK: Eğitimde kullandığımız T4 optimize edilmiş modelle uyumlu hale getirelim ---
    # Load config to get model parameters
    config_path = Path(__file__).parent.parent / "configs" / "t4_optimized_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['model']

    model = AGIFORMER(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_experts=1, # Eğitimde 1 uzman kullandık
        expert_types=['language'],
        use_memory=False, # Eğitimde kapalıydı
        use_introspection=False, # Eğitimde kapalıydı
        use_multimodal=False # Eğitimde kapalıydı
    ).to(device)
    # --- BİTTİ ---

    # --- DEĞİŞİKLİK: `best_model.pt` yerine `latest.pt` dosyasını yükle ---
    model_path = Path(__file__).parent.parent / "checkpoints" / "latest.pt"
    if model_path.exists():
        print(f"✅ Loading our latest trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"⚠️  Could not find the trained model at {model_path}. Using random weights.")
    # --- BİTTİ ---
    
    model.eval()

    print("\n" + "="*50)
    print("Example 1: Text Generation")
    print("="*50)
    
    prompt = "The future of AI"
    prompt_ids = text_to_ids(prompt).to(device)
    
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=40
        )
    
    generated_text = ids_to_text(generated_ids)
    print(f"Generated: {generated_text}")

    print("\n" + "="*50)
    print("Inference example completed!")
    print("="*50)

if __name__ == "__main__":
    main()
