# AGIFORMER Architecture Documentation

## Genel Bakış

AGIFORMER, yapay genel zeka (AGI) yolunda tasarlanmış, devrimci bir Transformer mimarisidir. Bu mimari, modern yapay zeka araştırmalarının en ileri konseptlerini bir araya getirir.

## Mimari Bileşenler

### 1. Multimodal Algı Çekirdeği (Multimodal Perception Core)

**Dosya:** `agiformer/core/multimodal_perception.py`

Farklı modaliteleri (metin, görüntü, ses, video) ortak bir anlamsal uzaya dönüştüren birleşik bir algı sistemidir.

**Özellikler:**
- **TextEncoder**: Karakter veya token seviyesinde metin girişi için
- **ImageEncoder**: Vision Transformer yaklaşımı ile görüntü işleme
- **AudioEncoder**: Ham ses dalgalarını mel-spektrogram özelliklerine dönüştürme
- **VideoEncoder**: Zamansal ve uzamsal bilgiyi birleştiren video işleme
- **Cross-Modal Attention**: Farklı modaliteler arasındaki ilişkileri öğrenme

**Kullanım:**
```python
from agiformer.core import MultimodalPerceptionCore

perception = MultimodalPerceptionCore(d_model=768)
modality_embeds, unified = perception(
    text=text_tensor,
    image=image_tensor,
    audio=audio_tensor
)
```

### 2. Öğrenilebilir Morfo-Semantik Tokenizasyon

**Dosya:** `agiformer/core/morfo_semantic_tokenizer.py`

Charformer'ın öğrenilebilir tokenizasyon yaklaşımını ve TMA-1'in morfolojik farkındalığını birleştirir.

**Özellikler:**
- **GBST Block**: Gradient-Based Subword Tokenization
- **Morphological Feature Extractor**: Morfolojik özellikleri çıkarır
- **Semantic Projection**: Semantik bilgiyi dahil eder
- **Learnable Boundaries**: Token sınırlarını öğrenir

**Avantajlar:**
- OOV (Out-of-Vocabulary) sorunu yok
- Göreve özel tokenizasyon
- Gradient-based öğrenme ile optimizasyon

### 3. Birleşik Bellek Omurgası (Unified Memory Backbone)

**Dosya:** `agiformer/core/memory_backbone.py`

Kısa vadeli (working memory) ve uzun vadeli (long-term memory) belleği yönetir.

**Bileşenler:**
- **WorkingMemory**: Transformer-XL benzeri segment-level recurrence
- **MemoryBank**: Harici, öğrenilebilir bellek bankası
- **Memory Fusion**: Farklı bellek türlerini birleştirme

**Özellikler:**
- Dinamik bellek erişimi
- Öğrenilebilir okuma/yazma mekanizmaları
- Segment-level context propagation

### 4. Gelişmiş Attention Mekanizmaları

**Dosya:** `agiformer/core/attention.py`

**Türler:**
- **MultiHeadAttention**: Standart multi-head self-attention
- **LinearAttention**: O(n) karmaşıklığı ile lineer attention
- **SyntaxAwareAttention**: Sözdizimi farkındalıklı attention
- **CrossModalAttention**: Cross-modal fusion için

### 5. Mixture of Experts (MoE) Sistemi

**Dosya:** `agiformer/experts/moe.py`

Dinamik routing ile uzmanlaşmış akıl yürütme motorları.

**Uzmanlar:**
1. **Language Expert** (`language_expert.py`): Dil anlama ve üretimi
   - Morfo-semantik farkındalık
   - Syntax-aware attention
   - TMA-1 ilhamlı özellikler

2. **Logic Expert** (`logic_expert.py`): Mantıksal akıl yürütme
   - İlişkisel akıl yürütme
   - Mantıksal yapı kodlama
   - Matematiksel modelleme

3. **Spatial Expert** (`spatial_expert.py`): Uzamsal ilişkiler
   - Geometrik özellik çıkarımı
   - Uzamsal attention
   - Mesafe/angle hesaplamaları

4. **Causal Expert** (`causal_expert.py`): Nedensellik
   - Cause-effect ilişkileri
   - Zamansal kodlama
   - Yönlendirilmiş attention

**Routing:**
- Top-k expert seçimi
- Load balancing
- Dinamik ağırlıklandırma

### 6. İç Gözlem Döngüsü ve Öz-Model

**Dosya:** `agiformer/introspection/self_model.py`

Modelin kendi düşünce süreçlerini gözlemlemesi ve değerlendirmesi.

**Bileşenler:**
- **SelfModel**: Kendi durumunu gözlemleme
- **IntrospectionLoop**: İteratif self-reflection
- **Error Detection**: Hata tespiti
- **Confidence Estimation**: Güven tahmini
- **Self-Correction**: Kendi kendini düzeltme

**Meta-Learning** (`meta_learning.py`):
- Hızlı uyarlama (fast adaptation)
- Görev embedding'i
- MAML ilhamlı yaklaşım

### 7. Ana AGIFORMER Modeli

**Dosya:** `agiformer/model.py`

Tüm bileşenleri birleştiren ana model.

**Mimari:**
```
Input (Text/Image/Audio/Video)
    ↓
[Multimodal Perception OR Tokenizer]
    ↓
[Memory Backbone]
    ↓
[AGIFORMER Blocks × N]
    ├── Attention
    ├── MoE (Experts)
    └── Introspection (last layer)
    ↓
[Meta-Learner]
    ↓
Output Projection
```

**AGIFORMER Block:**
- Self-attention (linear or standard)
- Mixture of Experts
- Optional introspection loop

## Model Konfigürasyonu

### Temel Parametreler

- `vocab_size`: 256 (karakter bazlı)
- `d_model`: 768 (vektör boyutu)
- `n_layers`: 12 (blok sayısı)
- `n_heads`: 12 (attention head sayısı)
- `d_ff`: 3072 (feed-forward boyutu)
- `n_experts`: 4 (uzman sayısı)

### Özellik Bayrakları

- `use_linear_attention`: O(n) attention kullan
- `use_memory`: Bellek sistemi aktif
- `use_introspection`: İç gözlem döngüsü
- `use_multimodal`: Multimodal desteği

## Eğitim Stratejisi

### Loss Functions

1. **Next-Token Prediction**: Standart dil modelleme
2. **Load Balancing Loss**: MoE'de uniform expert kullanımı
3. **Memory Loss**: Bellek sisteminin optimize edilmesi

### Optimizer

- AdamW optimizer
- Warmup scheduler (Transformer learning rate schedule)
- Gradient clipping

### Mixed Precision

- Automatic Mixed Precision (AMP) desteği
- Float16 training

## Kullanım Senaryoları

### 1. Text-Only Model

```python
model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    use_multimodal=False
)

logits, info = model(text=input_ids)
```

### 2. Multimodal Model

```python
model = AGIFORMER(
    vocab_size=256,
    d_model=768,
    use_multimodal=True
)

logits, info = model(
    text=text_ids,
    image=image_tensor,
    audio=audio_tensor
)
```

### 3. Memory-Enabled Model

```python
model = AGIFORMER(
    vocab_size=256,
    memory_size=10000,
    use_memory=True
)

# First sequence
logits1, _ = model(text=seq1, use_memory=True)

# Second sequence (memory persists)
logits2, _ = model(text=seq2, use_memory=True)

# Reset memory
model.reset_memory()
```

### 4. Text Generation

```python
model.eval()
generated = model.generate(
    prompt_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

## Performans Optimizasyonları

### 1. Linear Attention

Büyük diziler için O(n) karmaşıklığı:
```python
model = AGIFORMER(use_linear_attention=True)
```

### 2. Gradient Checkpointing

Bellek tasarrufu için:
```python
from torch.utils.checkpoint import checkpoint
```

### 3. Distributed Training

Multi-GPU training desteği

## Araştırma Yönleri

### 1. Daha Fazla Uzman Türü
- Temporal Expert
- Perceptual Expert
- Abstract Reasoning Expert

### 2. Gelişmiş Bellek
- Hierarchical memory
- Episodic memory
- Semantic memory

### 3. Daha İyi Introspection
- Chain-of-thought reasoning
- Self-explanatory models
- Meta-reasoning

## Referanslar

- Charformer: Google Research
- Transformer-XL: CMU
- Mixture of Experts: Google Brain
- TMA-1: Morfo-Semantic Awareness
- Flash Attention: Stanford DAWN

## Lisans

MIT License

