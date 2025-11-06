# AGIFORMER: Experimental AGI Research Framework v0.1

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Private-green.svg)](LICENSE.txt)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange.svg)](https://github.com/inkbytefo/agiformer_a)

AGIFORMER, Yapay Genel Zeka'ya yönelik yenilikçi mimari konseptlerini araştıran deneysel bir framework'tür. Geliştirilme aşamasındaki bileşenleri (uzmanlaşmış akıl yürütme motorları, bellek sistemi, iç gözlem yetenekleri) bir araya getirerek geleneksel Transformer mimarilerinin ötesine geçmeye yönelik kavramsal araştırmalar yürütmektedir.

## ✨ Experimental Features (Under Development)

- 🧠 **Mixture of Experts (MoE)**: 4 specialized reasoning engines (Language, Logic, Spatial, Causal) - *Conceptual implementation*
- 🎯 **Multimodal Perception**: Text, image, audio and video processing - *Research framework*
- 💾 **Advanced Memory System**: Working memory + long-term memory - *Architectural concept*
- 🔍 **Introspection**: Self-observation and iterative improvement - *Experimental phase*
- 📝 **MorphoPiece Tokenizer**: Turkish morphological awareness tokenization - *Basic implementation*
- 🇹🇷 **Turkish Language Processing**: TMA-1 integration for advanced Turkish understanding - *Development stage*
- ⚡ **Performance Optimizations**: Mixed precision, gradient_checkpointing support - *Infrastructure ready*

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repoyu klonla
git clone https://github.com/inkbytefo/agiformer_a.git
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
│  ┌─────────────────────────────────────────────────┐     │
│  │           TMA-1 (Türkçe Mantık Ağı)             │     │
│  │        AgglutinativeAttention + Grammar         │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Metin     │  │   Görüntü   │  │    Ses      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │               │               │                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │        Multimodal Perception Core              │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │            Memory Backbone                       │     │
│  │    MemoryBank + WorkingMemory + UnifiedMemory   │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │          Expert Stack (Mixture of Experts)      │     │
│  │   Language │ Logic │ Spatial │ Causal │ MoE     │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │              AGIFORMER Block (N layers)         │     │
│  └─────────────────────────────────────────────────┘     │
│                           │                                 │
│  ┌─────────────────────────────────────────────────┐     │
│  │              Output Projection                   │     │
│  └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 🧩 Bileşenler

### 1. TMA-1 (Türkçe Mantık Ağı)
- **AgglutinativeAttention**: Türkçe'nin eklemeli yapısına özel attention mekanizması
- **MorphoPiece Tokenizer**: Morfolojik farkındalıklı tokenizasyon
- **Grammar Engine**: Türkçe dilbilgisi kuralları ve ses uyumu kontrolü
- **Morpho Splitter**: Regex ve Java tabanlı morfem ayrımı

### 2. Uzman Sistemi (MoE)
- **ExpertRouter**: Dinamik uzman yönlendirme
- **Language Expert**: Dil işleme ve morfolojik analiz
- **Logic Expert**: Mantıksal akıl yürütme
- **Spatial Expert**: Mekansal ve geometrik işleme
- **Causal Expert**: Nedensel ilişki analizi
- **Neuro-Symbolic Expert**: Sembolik-mantıksal hibrit akıl yürütme

### 3. Multimodal Algı
- **TextEncoder**: Karakter/seviye veya token-seviye metin işleme
- **ImageEncoder**: CLIP tabanlı görüntü encoder'ı
- **AudioEncoder**: Mel-spektrogram tabanlı ses işleme
- **VideoEncoder**: Spatio-temporal video analizi

### 4. Bellek Sistemi
- **MemoryBank**: Uzun süreli bellek deposu
- **WorkingMemory**: Segment-seviye çalışma belleği
- **UnifiedMemoryBackbone**: Tümleşik bellek yönetimi

### 5. Knowledge Graph Sistemi
- **GlobalKnowledgeGraph**: Küresel bilgi grafiği
- **DynamicKnowledgeGraph**: Dinamik kavram ilişkileri
- **RelationClassifier**: İlişki tipi sınıflandırma

### 6. İç Gözlem
- **Self-Model**: Kendi durumunu gözlemleme
- **Meta Learning**: Öğrenmeyi öğrenme yetenekleri
- **Task Classifier**: Görev tipi otomatik sınıflandırma
- **Pseudo Labeler**: Otomatik veri etiketleme

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
# Yeni birleştirilmiş eğitim script'i - Hydra konfigürasyonu ile
python train.py experiment=phase1_lite hardware=t4_gpu

# Farklı deneyler
python train.py experiment=phase1_baseline hardware=default_gpu
python train.py experiment=phase1_lite hardware=cpu

# Özel veri ile eğitim
python train.py experiment=phase1_lite hardware=t4_gpu data.data_path=turkish_dataset.jsonl

# Mevcut konfigürasyonları görüntüle
python train.py --help
```

#### Konfigürasyon Yapısı

Yeni konfigürasyon sistemi üç ana kategoriye ayrılmıştır:

- **`conf/experiment/`**: Deney spesifik ayarlar (phase1_lite, phase1_baseline)
- **`conf/hardware/`**: Donanım optimizasyonları (cpu, t4_gpu, default_gpu)
- **`conf/base/`**: Temel model ve eğitim ayarları

#### Örnek Konfigürasyonlar

```yaml
# conf/experiment/phase1_lite.yaml
d_model: 512
n_layers: 6
use_agglutinative_attention: true
morphological_analysis: true

# conf/hardware/t4_gpu.yaml
device: cuda
batch_size: 16
use_amp: true
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

## 📈 Current Development Status

### Framework Architecture (Under Development)
- **Basic Model Framework**: Conceptual implementation of MoE architecture
- **Memory Usage**: Infrastructure ready for memory optimization
- **Training Pipeline**: Basic training loop with room for optimization
- **Research Focus**: Architectural experimentation, not performance benchmarking

### Infrastructure Status
- ✅ Mixed precision training infrastructure
- ✅ Gradient checkpointing support
- ✅ Configurable model architecture
- 🔄 Training optimization (in progress)

## 🎯 Research Vision (Long-term Goals)

**Note**: The following represents our long-term research vision and experimental goals, not current achieved results.

### Target Performance Goals
- **SOTA Reasoning**: Mixture of Experts for specialized cognitive tasks
- **Multimodal Integration**: Unified text, image, audio, video understanding
- **Advanced Memory**: Persistent knowledge and context awareness
- **Self-Introspection**: Meta-learning and self-improvement capabilities
- **Turkish Language Mastery**: Native-level Turkish language understanding

**Important**: These are research objectives and experimental goals, not currently achieved benchmarks. The project is in early research and development phase.

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
│   │   ├── attention.py      # MultiHead, Linear, SyntaxAware, CrossModal attention
│   │   ├── base_components.py # LayerNorm, PositionalEncoding, FeedForward
│   │   ├── memory_backbone.py # MemoryBank, WorkingMemory, UnifiedMemoryBackbone
│   │   └── multimodal_perception.py # Text/Image/Audio/Video encoders
│   ├── language/             # Türkçe dil işleme modülleri (TMA-1)
│   │   ├── model.py          # TMA1Model (Türkçe Mantık Ağı)
│   │   ├── attention.py      # AgglutinativeAttention (eklemeli yapı)
│   │   ├── morpho_splitter.py # Regex ve Java tabanlı morfem ayrımı
│   │   ├── tokenizer.py      # MorphoPiece tokenizer
│   │   └── grammar_engine.py # Türkçe dilbilgisi kuralları motoru
│   ├── experts/              # Mixture of Experts sistemi
│   │   ├── moe.py           # ExpertRouter, Expert, MixtureOfExperts
│   │   ├── language_expert.py # Dil uzmanı
│   │   ├── logic_expert.py   # Mantık uzmanı
│   │   ├── spatial_expert.py # Mekansal uzman
│   │   ├── causal_expert.py  # Nedensel uzman
│   │   ├── knowledge_graph.py # Global/Dynamic knowledge graphs
│   │   ├── neuro_symbolic_expert.py # Neuro-symbolic reasoning
│   │   ├── pseudo_labeler.py # Otomatik etiketleme
│   │   ├── task_classifier.py # Görev tipi sınıflandırma
│   │   └── relations.py      # İlişki işleme
│   ├── introspection/        # İç gözlem sistemi
│   │   ├── self_model.py    # Self-model gözlemi
│   │   └── meta_learning.py # Meta-öğrenme
│   ├── data/                 # Birleştirilmiş veri işleme modülü
│   │   └── dataset.py        # Tüm dataset sınıfları (TurkishTextDataset, TextDataset, vb.)
│   ├── datasets/             # Multimodal veri setleri
│   │   ├── base_dataset.py   # Temel dataset sınıfı
│   │   └── cc_datasets.py    # Common Crawl veri işleme
│   ├── __init__.py
│   ├── model.py              # AGIFORMER ana model
│   ├── data_quality.py       # Veri kalitesi kontrolü
│   └── utils.py              # Yardımcı fonksiyonlar
├── conf/                     # Yeni konfigürasyon yapısı
│   ├── config.yaml           # Ana konfigürasyon girişi
│   ├── base/                 # Temel ayarlar
│   │   ├── model.yaml        # Temel model mimarisi
│   │   └── training.yaml     # Temel eğitim ayarları
│   ├── experiment/           # Deney spesifik konfigürasyonlar
│   │   ├── phase1_lite.yaml  # Hafif model deneyi
│   │   └── phase1_baseline.yaml # Karşılaştırma deneyi
│   ├── hardware/             # Donanım optimizasyonları
│   │   ├── cpu.yaml          # CPU optimizasyonu
│   │   ├── t4_gpu.yaml       # T4 GPU optimizasyonu
│   │   └── default_gpu.yaml  # Varsayılan GPU ayarları
│   ├── logging/              # Log ayarları
│   └── model/                # Eski model konfigürasyonları (arşiv)
├── archive/                  # Arşivlenmiş eski script'ler
│   ├── train_phase1.py       # Eski Phase 1 eğitim script'i
│   ├── training_example.py   # Eski eğitim örneği
│   ├── quick_test.py         # Eski test script'i
│   └── old_train_backup.py    # Eski train.py yedeği
├── examples/                 # Kullanım örnekleri
├── scripts/                  # Yardımcı script'ler
│   ├── analyze_data_quality.py
│   ├── clean_corpus.py
│   ├── download_real_datasets.py
│   ├── prepare_cc12m.py
│   ├── preprocess_language_data.py
│   └── train_tokenizer.py
├── tests/                    # Testler
└── train.py                  # Yeni birleştirilmiş eğitim script'i
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
git clone https://github.com/inkbytefo/agiformer_a.git
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

Bu proje **özel mülkiyet lisansı** altında lisanslanmıştır - [LICENSE.txt](LICENSE.txt) dosyasına bakın. Tüm fikri mülkiyet hakları Tevfik İşkın'a aittir.

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

## 🗺️ Development Roadmap

### Current Status (v0.1)
- ✅ Basic framework architecture
- ✅ Initial MoE conceptual implementation
- ✅ Training infrastructure setup
- 🔄 Real dataset integration (in progress)
- 🔄 Component testing and validation

### v0.2 (Next Development Phase)
- [ ] Complete real dataset training verification
- [ ] Enhanced MoE expert implementations
- [ ] Improved memory system architecture
- [ ] Basic multimodal integration testing

### v0.3 (Long-term Research Goals)
- [ ] Advanced expert specializations
- [ ] Distributed training capabilities
- [ ] Mobile optimization research
- [ ] API service development

**Note**: All roadmap items are development goals, not guaranteed deliverables. This is research-focused experimental work.

---

<div align="center">

**AGIFORMER** - Yapay Genel Zeka'ya giden yolculukta bir adım

[![Star](https://img.shields.io/github/stars/yourusername/agiformer.svg?style=social&label=Star)](https://github.com/inkbytefo/agiformer_a)
[![Fork](https://img.shields.io/github/forks/yourusername/agiformer.svg?style=social&label=Fork)](https://github.com/inkbytefo/agiformer_a/fork)
[![Watch](https://img.shields.io/github/watchers/yourusername/agiformer.svg?style=social&label=Watch)](https://github.com/inkbytefo/agiformer_a)

</div>
