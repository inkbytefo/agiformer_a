# AGIFORMER v0.1 Hızlı Başlangıç Kılavuzu

## İçerik
1. [Kurulum](#kurulum)
2. [Hızlı Başlangıç](#hızlı-başlangıç)
3. [Temel Kullanım](#temel-kullanım)
4. [Gelişmiş Özellikler](#gelişmiş-özellikler)
5. [Eğitim](#eğitim)
6. [Sıkça Sorulan Sorular](#sıkça-sorulan-sorular)

---

## Kurulum

### Gereksinimler
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU için isteğe bağlı)

### Kurulum Adımları

```bash
# 1. Repoyu klonla
git clone https://github.com/inkbytefo/agiformer.git
cd agiformer

# 2. Python ortamını oluştur
conda create -n agiformer python=3.9
conda activate agiformer

# 3. Bağımlılıkları kur
pip install -r requirements.txt

# 4. Paketi kur
pip install -e .
```

### requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.24.0
einops>=0.7.0
pyyaml>=6.0
tqdm>=4.65.0
transformers>=4.30.0
```

### GPU Desteği

```bash
# CUDA kontrolü
python -c "import torch; print(torch.cuda.is_available())"

# GPU versiyonu (recommend)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Hızlı Başlangıç

### 5 Dakikada İlk Çalıştırma

```python
import torch
from agiformer import AGIFORMER

# Model oluştur
model = AGIFORMER(
    vocab_size=256,
    d_model=384,      # Küçük model için hızlı başlangıç
    n_layers=2,
    n_heads=6,
    use_multimodal=True,
    use_memory=True,
    use_introspection=True
)

# Basit metin üretimi
text = "Merhaba dünya!"
char_ids = [ord(c) % 256 for c in text]
input_tensor = torch.tensor([char_ids], dtype=torch.long)

# Üretim yap
model.eval()
with torch.no_grad():
    generated = model.generate(
        input_tensor,
        max_new_tokens=20,
        temperature=0.8
    )

# Sonucu göster
result = ''.join([chr(c % 256) for c in generated[0].cpu().numpy()])
print(f"Girdi: {text}")
print(f"Çıktı: {result}")
```

### Multimodal Örnek

```python
import torch
from agiformer import AGIFORMER

# Multimodal model
model = AGIFORMER(use_multimodal=True)
model.eval()

# Girdiler
text = torch.randint(0, 256, (1, 16))      # Metin
image = torch.randn(1, 3, 224, 224)       # Görüntü

# İşle
with torch.no_grad():
    logits, info = model(text=text, image=image)

print(f"Çıktı şekli: {logits.shape}")
print(f"Modaliteler: {info['modalities']}")
print(f"Multimodal: {info['multimodal']}")
```

---

## Temel Kullanım

### 1. Model Oluşturma

```python
from agiformer import AGIFORMER

# Temel model
model = AGIFORMER()

# Özel konfigürasyon
model = AGIFORMER(
    vocab_size=256,           # Karacter vocab boyutu
    d_model=768,             # Model boyutu
    n_layers=12,             # Katman sayısı
    n_heads=12,              # Attention başlıkları
    d_ff=3072,               # Feed-forward boyutu
    n_experts=4,             # Uzman sayısı
    expert_types=["language", "logic", "spatial", "causal"],
    memory_size=10000,       # Bellek boyutu
    max_seq_len=2048,        # Maksimum dizi uzunluğu
    dropout=0.1,             # Dropout
    use_memory=True,         # Bellek sistemi
    use_introspection=True,  # İç gözlem
    use_multimodal=True      # Multimodal
)
```

### 2. Forward Pass

```python
# Sadece metin
logits, info = model(text=input_ids)

# Sadece görüntü
logits, info = model(image=image_tensor)

# Metin + görüntü
logits, info = model(text=input_ids, image=image_tensor)

# Tüm modaliteler
logits, info = model(
    text=input_ids,
    image=image_tensor,
    audio=audio_tensor,
    video=video_tensor
)
```

### 3. Metin Üretimi

```python
# Basit üretim
generated = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)

# Kontrolsüz üretim (daha yaratıcı)
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1.5,
    top_k=0,
    top_p=0.95
)

# Kontrollü üretim (daha tutarlı)
generated = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_k=40,
    top_p=0.8
)
```

### 4. Model Bilgisi

```python
from agiformer.utils import count_parameters, format_number

# Parametre sayısı
params = count_parameters(model)
print(f"Toplam: {format_number(params['total'])}")
print(f"Eğitilebilir: {format_number(params['trainable'])}")

# Model bilgisi
logits, info = model(text=input_ids)
print(f"Model info keys: {list(info.keys())}")

# Uzman kullanımı
for i, block in enumerate(info['blocks']):
    if 'moe' in block:
        usage = block['moe']['router_info']['expert_usage']
        print(f"Block {i} expert usage: {usage}")
```

---

## Gelişmiş Özellikler

### 1. Bellek Sistemi

```python
# Bellek sıfırlama
model.reset_memory()

# Bellek durumunu kontrol et
logits, info = model(text=input_ids)
memory_info = info['memory']
print(f"Memory steps: {memory_info['step_count']}")
print(f"Memory read weights: {memory_info['memory_read_weights']}")

# Bellek boyutunu ayarla
model = AGIFORMER(memory_size=5000)  # Daha küçük bellek
```

### 2. İç Gözlem

```python
# İç gözlem bilgilerini kontrol et
logits, info = model(text=input_ids)
last_block = info['blocks'][-1]

if 'introspection' in last_block:
    introspection = last_block['introspection']
    print(f"Iterations: {introspection['num_iterations']}")
    print(f"Confidence: {introspection['final_confidence']:.3f}")
    print(f"Error: {introspection['final_error']:.3f}")
```

### 3. Uzman Sistemi

```python
# Özel uzman tipleri
model = AGIFORMER(
    n_experts=4,
    expert_types=["language", "logic", "spatial", "causal"]
)

# Uzman kullanımını izle
logits, info = model(text=input_ids)
for i, block in enumerate(info['blocks']):
    if 'moe' in block:
        router_info = block['moe']['router_info']
        print(f"Block {i}:")
        print(f"  Expert usage: {router_info['expert_usage']}")
        print(f"  Router confidence: {router_info['avg_router_confidence']:.3f}")
        if 'load_balancing_loss' in router_info:
            print(f"  Load balancing loss: {router_info['load_balancing_loss']:.6f}")
```

### 4. Multimodal İşleme

```python
# Modalite encoder'larını kontrol et
if hasattr(model, 'multimodal_perception'):
    # Metin encoder
    text_emb = model.multimodal_perception.text_encoder(text_ids)
    
    # Görüntü encoder (CLIP tabanlı)
    image_emb = model.multimodal_perception.image_encoder(image_tensor)
    
    # Cross-modal attention
    modality_embeds, unified = model.multimodal_perception(
        text=text_ids,
        image=image_tensor
    )
```

---

## Eğitim

### 1. Basit Eğitim Örneği

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from agiformer import AGIFORMER

# Model ve optimizer
model = AGIFORMER(d_model=384, n_layers=2)  # Küçük model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Eğitim döngüsü
model.train()
for epoch in range(10):
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        logits, info = model(text=input_ids)
        
        # Kayıp hesabı
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_ids.view(-1)
        loss = criterion(logits_flat, target_flat)
        
        # MoE load balancing ekle
        total_loss = loss
        for block_info in info['blocks']:
            if 'moe' in block_info and 'load_balancing_loss' in block_info['moe']['router_info']:
                lb_loss = block_info['moe']['router_info']['load_balancing_loss']
                total_loss = total_loss + lb_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

### 2. Gelişmiş Eğitim Script'i

```python
# examples/training_example.py dosyasını kullanın
python examples/training_example.py \
    --config configs/base_config.yaml \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epochs 10
```

### 3. Konfigürasyon ile Eğitim

```python
import yaml
from agiformer import AGIFORMER

# Konfigürasyon yükle
with open('configs/base_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model oluştur
model = AGIFORMER(**config['model'])

# Eğitim parametreleri
train_config = config['training']
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config['learning_rate'],
    betas=(train_config['adam_beta1'], train_config['adam_beta2']),
    eps=train_config['adam_epsilon'],
    weight_decay=train_config['weight_decay']
)
```

### 4. Checkpoint Kaydetme/Yükleme

```python
# Kaydet
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Yükle
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## Testler ve Doğrulama

### 1. Testleri Çalıştırma

```bash
# Tüm testler
python -m pytest tests/ -v

# Belirli test
python tests/test_model.py

# Örnek testler
python examples/multimodal_test.py
python examples/moe_test.py
python examples/memory_test.py
python examples/introspection_test.py
```

### 2. Model Doğrulama

```python
# Model testi
python test_fix.py

# Konfigürasyon testi
python test_fix.py --config configs/base_config.yaml
python test_fix.py --config configs/colab_config.yaml
```

---

## Sıkça Sorulan Sorular

### Q: Hangi Python sürümünü kullanmalıyım?
**A:** Python 3.8+ önerilir. En iyi uyum için Python 3.9 veya 3.10 kullanın.

### Q: GPU gerekiyor mu?
**A:** Hayır, CPU üzerinde de çalışır ancak eğitim çok yavaş olacaktır. GPU ile önemli hız artışı sağlanır.

### Q: Model ne kadar hafıza kullanıyor?
**A:** Temel konfigürasyon (~150M parametre):
- CPU: ~2GB RAM
- GPU: ~4GB VRAM (batch_size=32)

### Q: Nasıl daha hızlı eğitim yapabilirim?
**A:** 
- Mixed precision kullanın (`use_amp=True`)
- Batch boyutunu artırın
- Gradient accumulation kullanın
- Daha küçük model kullanın

### Q: Kendi verisetimiyle nasıl eğitebilirim?
**A:** 
- Verinizi PyTorch Dataset formatına dönüştürün
- Karakter ID'lerine çevirin (ord() % vocab_size)
- DataLoader ile yükleyin
- Örnek training script'ini kullanın

### Q: Model nasıl çalıştığını anlamak için nereye bakmalıyım?
**A:** 
- [`AGIFORMER_TECHNICAL_DOCUMENTATION.md`](AGIFORMER_TECHNICAL_DOCUMENTATION.md)
- [`AGIFORMER_ARCHITECTURE_DIAGRAMS.md`](AGIFORMER_ARCHITECTURE_DIAGRAMS.md)
- [`examples/`](examples/) dizinindeki örnekler

### Q: Hata aldım, ne yapmalıyım?
**A:** 
1. Önce testleri çalıştırın: `python test_fix.py`
2. Konfigürasyonu kontrol edin
3. GPU hafızasını kontrol edin
4. GitHub issues'da arama yapın

### Q: Model nasıl genişletilebilir?
**A:** 
- Yeni uzman tipleri ekleyin
- Modalite encoder'ları ekleyin
- Bellek mekanizmalarını geliştirin
- İç gözlem algoritmalarını değiştirin

### Q: Performansı nasıl optimize edebilirim?
**A:** 
- Mixed precision training
- Gradient checkpointing
- Expert caching
- Sequence packing
- Model paralelizasyonu

---

## Sonraki Adımlar

1. **Teknik Dökümantasyon**: Detaylı mimari bilgisi için
2. **Mimari Diyagramları**: Görsel anlam için
3. **Örnekler**: [`examples/`](examples/) dizininde
4. **Testler**: [`tests/`](tests/) dizininde
5. **Script'ler**: [`scripts/`](scripts/) dizininde

---

### Yardım ve Destek

- **GitHub**: https://github.com/inkbytefo/agiformer
- **Issues**: https://github.com/inkbytefo/agiformer/issues
- **Wiki**: https://github.com/inkbytefo/agiformer/wiki

---

*Lisans: MIT License*  
*Version: v0.1*  
*Son Güncelleme: 2024*