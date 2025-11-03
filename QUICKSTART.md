# AGIFORMER Quick Start Guide

## Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/yourusername/agiformer
cd agiformer

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Veya development mode ile
pip install -e .
```

## Hızlı Başlangıç

### 1. Temel Model Oluşturma

```python
from agiformer import AGIFORMER
import torch

# Model oluştur
model = AGIFORMER(
    vocab_size=256,      # Karakter bazlı
    d_model=768,         # Model boyutu
    n_layers=12,         # Katman sayısı
    n_heads=12,          # Attention head sayısı
    n_experts=4,         # Uzman sayısı
    expert_types=['language', 'logic', 'spatial', 'causal']
)

# Parametre sayısını kontrol et
from agiformer.utils import count_parameters
params = count_parameters(model)
print(f"Total parameters: {params['total']:,}")
```

### 2. Basit Forward Pass

```python
# Metin girişi (karakter ID'leri)
batch_size, seq_len = 2, 32
text_input = torch.randint(0, 256, (batch_size, seq_len))

# Forward pass
logits, info = model(text=text_input)

print(f"Output shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
print(f"Model info: {list(info.keys())}")
```

### 3. Metin Üretimi

```python
# Model'i eval moduna al
model.eval()

# Başlangıç metni
prompt = "The future of"
prompt_ids = torch.tensor([[ord(c) % 256 for c in prompt]])

# Metin üret
with torch.no_grad():
    generated = model.generate(
        prompt_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

# Sonuçları göster
generated_text = ''.join([chr(c % 256) for c in generated[0].cpu().numpy()])
print(generated_text)
```

### 4. Multimodal Kullanım

```python
# Multimodal model oluştur
multimodal_model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    use_multimodal=True
)

# Farklı modalitelerle
text_ids = torch.randint(0, 256, (2, 32))
image_tensor = torch.randn(2, 3, 224, 224)  # RGB images
audio_tensor = torch.randn(2, 16000)  # Audio waveform

logits, info = multimodal_model(
    text=text_ids,
    image=image_tensor,
    audio=audio_tensor
)

print(f"Modalities used: {info.get('modalities', [])}")
```

### 5. Bellek Sistemi ile Kullanım

```python
# Bellek aktif model
memory_model = AGIFORMER(
    vocab_size=256,
    memory_size=10000,
    use_memory=True
)

# İlk dizi
seq1 = torch.randint(0, 256, (1, 50))
logits1, info1 = memory_model(text=seq1, use_memory=True)

# İkinci dizi (bellek devam eder)
seq2 = torch.randint(0, 256, (1, 50))
logits2, info2 = memory_model(text=seq2, use_memory=True)

# Belleği sıfırla
memory_model.reset_memory()
```

### 6. Eğitim Örneği

```python
import torch.nn as nn
from torch.utils.data import DataLoader

# Model
model = AGIFORMER(vocab_size=256, d_model=768)

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Eğitim döngüsü
model.train()
for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
    optimizer.zero_grad()
    
    # Forward
    logits, _ = model(text=input_ids)
    
    # Loss hesapla
    loss = criterion(logits.view(-1, 256), target_ids.view(-1))
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if batch_idx % 100 == 0:
        print(f"Loss: {loss.item():.4f}")
```

### 7. Model Kaydetme ve Yükleme

```python
# Kaydet
torch.save(model.state_dict(), 'agiformer_model.pt')

# Yükle
model = AGIFORMER(vocab_size=256, d_model=768)
model.load_state_dict(torch.load('agiformer_model.pt'))
model.eval()
```

### 8. Farklı Konfigürasyonlar

```python
# Küçük model (hızlı test için)
small_model = AGIFORMER(
    vocab_size=256,
    d_model=128,
    n_layers=2,
    n_heads=4,
    d_ff=512,
    n_experts=2
)

# Büyük model (daha iyi performans)
large_model = AGIFORMER(
    vocab_size=256,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    d_ff=4096,
    n_experts=8
)

# Linear attention ile (büyük diziler için)
linear_model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    use_linear_attention=True
)

# Introspection ile
introspection_model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    use_introspection=True
)
```

## İleri Seviye Kullanım

### Özel Expert Oluşturma

```python
from agiformer.experts.moe import Expert

class CustomExpert(Expert):
    def forward(self, x):
        # Özel işlemler
        return super().forward(x)

# Özel expert ile model
custom_experts = [CustomExpert(768, 3072) for _ in range(4)]
model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    n_experts=4,
    custom_experts=custom_experts
)
```

### Model Bilgisi Alma

```python
from agiformer.utils import get_model_size_mb, format_number

# Model boyutu
size_mb = get_model_size_mb(model)
print(f"Model size: {size_mb:.2f} MB")

# Parametre sayısı
params = count_parameters(model)
print(f"Parameters: {format_number(params['total'])}")
```

## Sorun Giderme

### Out of Memory

```python
# Daha küçük model kullan
model = AGIFORMER(d_model=512, n_layers=6)

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

# Mixed precision
from torch.cuda.amp import autocast
with autocast():
    logits, _ = model(text=input_ids)
```

### Yavaş Eğitim

```python
# Linear attention kullan
model = AGIFORMER(use_linear_attention=True)

# Daha az expert
model = AGIFORMER(n_experts=2)

# Bellek sistemi kapat
model = AGIFORMER(use_memory=False)
```

## Daha Fazla Bilgi

- Detaylı mimari: [ARCHITECTURE.md](ARCHITECTURE.md)
- Örnekler: [examples/](examples/)
- Testler: [tests/](tests/)

