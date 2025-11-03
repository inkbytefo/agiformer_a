# AGIFORMER: Artificial General Intelligence Transformer v0.1

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](https://github.com/yourusername/agiformer)

AGIFORMER, Yapay Genel Zeka'ya yönelik geliştirilmiş devrim niteliğinde bir Transformer mimarisidir. Çoklu modalite işleme, uzmanlaşmış akıl yürütme motorları, bellek sistemi ve iç gözlem yeteneklerini bir araya getirerek geleneksel dil modellerinin ötesine geçmeyi hedefler.

## ✨ Ana Özellikler

- 🧠 **Mixture of Experts (MoE)**: 4 uzmanlaşmış akıl yürütme motoru (Dil, Mantık, Mekansal, Nedensel)
- 🎯 **Multimodal Algı**: Metin, görüntü, ses ve video işleme
- 💾 **Gelişmiş Bellek Sistemi**: Çalışma belleği + uzun süreli bellek
- 🔍 **İç Gözlem**: Kendi kendini gözlemleme ve iteratif iyileştirme
- 📝 **Morfo-Sematik Tokenizer**: Karakter seviyesinde zenginleştirilmiş tokenizasyon
- ⚡ **Optimize Edilmiş Performans**: Mixed precision, gradient_checkpointing desteği

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repoyu klonla
git clone https://github.com/yourusername/agiformer.git
cd agiformer

# Ortam oluştur
conda create -n agiformer python=3.9
conda activate agiformer

# Kurulum yap
pip install -r requirements.txt
pip install -e .
```

### İlk Deneme

```python
import torch
from agiformer import AGIFORMER

# Model oluştur
model = AGIFORMER(
    vocab_size=256,
    d_model=384,      # Küçük model için hızlı başlangıç
    n_layers=2,
    use_multimodal=True,
    use_memory=True,
    use_introspection=True
)

# Metin üretimi
text = "Merhaba dünya!"
char_ids = [ord(c) % 256 for c in text]
input_tensor = torch.tensor([char_ids], dtype=torch.long)

model.eval()
with torch.no_grad():
    generated = model.generate(input_tensor, max_new_tokens=20)

result = ''.join([chr(c % 256) for c in generated[0].cpu().numpy()])
print(f"Çıktı: {result}")
```

## 📊 Mimari

```
┌─────────────────────────────────────────────────────────────┐
│                    AGIFORMER v0.1                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Metin     │  │   Görüntü   │  │    Ses      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │               │                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │        Multimodal Perception Core              │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │            Bellek Sistemi                        │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │          AGIFORMER Block Stack (N=12)           │     │
│  │  Attention + MoE + Introspection                │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │              Output Projection                   │     │
│  └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 Bileşenler

### 1. Uzman Sistemi (MoE)
- **Dil Uzmanı**: Qwen3-0.6B LLM entegrasyonu
- **Mantık Uzmanı**: İlişkisel akıl yürütme
- **Mekansal Uzman**: Geometrik özellik çıkarma
- **Nedensel Uzman**: Sebep-sonuç ilişkileri

### 2. Multimodal Algı
- **Görüntü**: CLIP tabanlı encoder
- **Ses**: 1D evrişimli ağlar
- **Video**: Spatio-temporal işleme
- **Metin**: Morfo-semantic tokenizasyon

### 3. Bellek Sistemi
- **Working Memory**: Segment-level recurrence
- **Long-term Memory**: Dış bellek bankası
- **Memory Fusion**: Üçlü füzyon mekanizması

### 4. İç Gözlem
- **Self-Model**: Kendi durumunu gözlemleme
- **Error Detection**: Hata tespiti
- **Confidence Estimation**: Güven skoru tahmini
- **Correction Network**: Kendi kendini düzeltme

## 📚 Dökümantasyon

| Dökümantasyon | Açıklama |
|----------------|----------|
| [📖 Teknik Dökümantasyon](AGIFORMER_TECHNICAL_DOCUMENTATION.md) | Detaylı mimari ve API referansı |
| [🎨 Mimari Diyagramları](AGIFORMER_ARCHITECTURE_DIAGRAMS.md) | Görsel mimari diyagramları |
| [⚡ Hızlı Başlangıç](AGIFORMER_QUICK_START_GUIDE.md) | Kapsamlı başlangıç kılavuzu |
| [📓 Colab Rehberi](COLAB_GUIDE.md) | Google Colab'da çalıştırma |

## 🛠️ Kullanım

### Temel Kullanım

```python
from agiformer import AGIFORMER

# Model oluştur
model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_experts=4,
    use_multimodal=True,
    use_memory=True,
    use_introspection=True
)

# İleri geçiş
logits, info = model(text=input_ids, image=image_tensor)

# Metin üretimi
generated = model.generate(input_ids, max_new_tokens=50)
```

### Multimodal İşleme

```python
# Farklı modaliteler
logits, info = model(
    text=text_ids,
    image=image_tensor,
    audio=audio_tensor,
    video=video_tensor
)

# Model bilgisi
print(f"Modaliteler: {info['modalities']}")
print(f"Uzman kullanımı: {info['blocks'][0]['moe']['router_info']['expert_usage']}")
```

### Eğitim

```python
# Eğitim script'i
python train.py \
    --config configs/base_config.yaml \
    --batch_size 16 \
    --learning_rate 1e-4

# Veya özel
python examples/training_example.py
```

## 🧪 Testler

```bash
# Tüm testler
python -m pytest tests/ -v

# Bileşen testleri
python examples/multimodal_test.py
python examples/moe_test.py
python examples/memory_test.py
python examples/introspection_test.py

# Konfigürasyon testi
python test_fix.py
```

## 📈 Performans

### Model Boyutları
- **Temel Konfigürasyon**: ~150M parametre
- **Hafıza Kullanımı**: ~2-4GB GPU
- **İnference Hızı**: ~50ms/sequence (V100)

### Optimizasyonlar
- ✅ Mixed precision training
- ✅ Gradient checkpointing
- ✅ Expert caching
- ✅ Sequence packing

## 🎯 Örnekler

### 1. Metin Üretimi
```python
# Yaratıcı metin üretimi
prompt = "Gelecekte yapay zeka"
generated = model.generate(prompt_ids, temperature=1.2, top_p=0.9)
```

### 2. Görüntü-Metin
```python
# Görüntü açıklama
image = load_image("example.jpg")
logits, info = model(text=prompt_ids, image=image)
```

### 3. Bellek Analizi
```python
# Bellek kullanımını izle
logits, info = model(text=input_ids)
memory_info = info['memory']
print(f"Bellek adımları: {memory_info['step_count']}")
```

## 🔧 Konfigürasyon

### Temel Konfigürasyon ([`configs/base_config.yaml`](configs/base_config.yaml))
```yaml
model:
  vocab_size: 256
  d_model: 768
  n_layers: 12
  n_heads: 12
  n_experts: 4
  expert_types: ["language", "logic", "spatial", "causal"]
  use_memory: true
  use_introspection: true
  use_multimodal: true

training:
  batch_size: 32
  learning_rate: 0.0001
  use_amp: true
```

### Colab Konfigürasyonu ([`configs/colab_config.yaml`](configs/colab_config.yaml))
- Daha küçük model boyutu
- Azaltılmış batch size
- Optimize edilmiş for Colab

## 🏗️ Proje Yapısı

```
agiformer/
├── agiformer/                 # Ana paket
│   ├── core/                 # Çekirdek bileşenler
│   │   ├── attention.py      # Attention mekanizmaları
│   │   ├── memory_backbone.py # Bellek sistemi
│   │   ├── multimodal_perception.py # Multimodal
│   │   └── morfo_semantic_tokenizer.py # Tokenizer
│   ├── experts/              # Uzman sistemleri
│   │   ├── moe.py           # MoE yönlendirme
│   │   ├── language_expert.py # Dil uzmanı
│   │   ├── logic_expert.py   # Mantık uzmanı
│   │   ├── spatial_expert.py # Mekansal uzman
│   │   └── causal_expert.py  # Nedensel uzman
│   ├── introspection/        # İç gözlem sistemi
│   │   ├── self_model.py    # Self-model
│   │   └── meta_learning.py # Meta-learning
│   └── model.py              # Ana model
├── configs/                  # Konfigürasyon dosyaları
├── examples/                 # Kullanım örnekleri
├── scripts/                  # Yardımcı script'ler
├── tests/                    # Testler
└── train.py                  # Eğitim script'i
```

## 🤝 Katkı

Katkılarınızı bekliyoruz! Lütfen aşağıdaki adımları izleyin:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull request açın

### Geliştirme Kurulumu

```bash
# Geliştirme ortamı
git clone https://github.com/yourusername/agiformer.git
cd agiformer

# Development modunda kur
pip install -e ".[dev]"

# Testleri çalıştır
python -m pytest tests/ -v

# Kod formatlama
black agiformer/
flake8 agiformer/
```

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- **Transformer** mimarisi
- **Mixture of Experts** araştırmaları
- **CLIP** multimodal öğrenme
- **Charformer** morfo-semantik tokenizasyon
- **Transformer-XL** bellek mekanizmaları

## 📞 İletişim

- **Proje**: https://github.com/yourusername/agiformer
- **Issues**: https://github.com/yourusername/agiformer/issues
- **Discussions**: https://github.com/yourusername/agiformer/discussions

## 🗺️ Yol Haritası

### v0.2 (Planlanan)
- [ ] Daha fazla modalite (3D, sensör verileri)
- [ ] Gelişmiş uzmanlar (matematik, kod, müzik)
- [ ] Hiyerarşik bellek sistemi
- [ ] Meta-öğrenme Yetenekleri

### v0.3 (Uzun vadeli)
- [ ] Dağıtık eğitim desteği
- [ ] Mobil optimizasyon
- [ ] Web arayüzü
- [ ] API hizmeti

---

<div align="center">

**AGIFORMER** - Yapay Genel Zeka'ya giden yolculukta bir adım

[![Star](https://img.shields.io/github/stars/yourusername/agiformer.svg?style=social&label=Star)](https://github.com/yourusername/agiformer)
[![Fork](https://img.shields.io/github/forks/yourusername/agiformer.svg?style=social&label=Fork)](https://github.com/yourusername/agiformer/fork)
[![Watch](https://img.shields.io/github/watchers/yourusername/agiformer.svg?style=social&label=Watch)](https://github.com/yourusername/agiformer)

</div>