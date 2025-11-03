"""
Inference Example for AGIFORMER
"""

import torch
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
    text = ""
    for id_val in ids.squeeze().cpu().numpy():
        if id_val < vocab_size:
            text += chr(id_val)
    return text


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = AGIFORMER(
        vocab_size=256,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        n_experts=4,
        expert_types=['language', 'logic', 'spatial', 'causal'],
        memory_size=10000,
        max_seq_len=2048,
        dropout=0.1,
        use_memory=True,
        use_multimodal=False  # Text-only for this example
    )
    
    # Load the best model from our training run
    model_path = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
    if model_path.exists():
        print(f"✅ Loading our best trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"⚠️  Could not find the trained model at {model_path}. Using random weights.")
    
    model.to(device).half()
    model.eval()
    
    # Example 1: Forward pass
    print("\n" + "="*50)
    print("Example 1: Forward Pass")
    print("="*50)
    
    input_text = "Hello, world!"
    input_ids = text_to_ids(input_text)
    input_ids = input_ids.to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        logits, info = model(text=input_ids)
    
    print(f"Input: {input_text}")
    print(f"Output shape: {logits.shape}")
    print(f"Model info keys: {list(info.keys())}")
    
    # Example 2: Text Generation
    print("\n" + "="*50)
    print("Example 2: Text Generation")
    print("="*50)
    
    prompt = "The future of AI"
    prompt_ids = text_to_ids(prompt)
    prompt_ids = prompt_ids.to(device)
    
    print(f"Prompt: {prompt}")
    print("Generating...")
    
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
    
    generated_text = ids_to_text(generated_ids)
    print(f"Generated: {generated_text}")
    
    # Example 3: Multimodal (if enabled)
    print("\n" + "="*50)
    print("Example 3: Multimodal Input (if available)")
    print("="*50)
    
    # Create model with multimodal support
    multimodal_model = AGIFORMER(
        vocab_size=256,
        d_model=768,
        n_layers=6,  # Smaller for demo
        use_multimodal=True
    ).to(device)
    
    multimodal_model.eval()
    
    # Text input
    text_input = text_to_ids("This is a test").to(device)
    
    # Dummy image input (would be real images in practice)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        logits, info = multimodal_model(
            text=text_input,
            image=dummy_image
        )
    
    print(f"Multimodal output shape: {logits.shape}")
    print(f"Modalities used: {info.get('modalities', [])}")
    
    print("\n" + "="*50)
    print("Inference examples completed!")
    print("="*50)


if __name__ == "__main__":
    main()

