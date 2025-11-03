# AGIFORMER Faz 4.2: İlk "Gerçek" Eğitim - Google Colab Guide

## 🎯 Hedef
AGIFORMER'ı ilk kez gerçek dünya multimodal verisiyle eğitmek ve tüm profesyonel altyapının çalıştığını doğrulamak.

## 📋 Ön Koşullar

### Google Colab Kurulumu
```bash
# GPU kontrolü
!nvidia-smi

# Gerekli kütüphaneler
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install wandb pillow requests tqdm pandas pathlib
!pip install yaml
```

### W&B (Weights & Biases) Kurulumu
```bash
# W&B login
import wandb
wandb.login()
# API key gerekecek: https://wandb.ai/authorize
```

## 🚀 Adım Adım Kurulum ve Eğitim

### Adım 1: Projeyi Colab'a Yükle
```python
# Google Drive'ı mount et
from google.colab import drive
drive.mount('/content/drive')

# Projeyi kopyala
!git clone <repository_url> /content/agiformer_b
%cd /content/agiformer_b

# Alternatif: Drive'dan kopyala
!cp -r /content/drive/MyDrive/agiformer_b /content/
%cd /content/agiformer_b
```

### Adım 2: CC12M Veri Setini Hazırla
```bash
# 10,000 örneklik CC12M alt kümesi oluştur
!python scripts/prepare_cc12m.py --num_samples 10000 --output_dir data/cc12m_10k

# Beklenen çıktı:
# ✓ 10,000 train örnek
# ✓ 1,000 validation örnek  
# ✓ 2,000 adet 224x224 sentetik görüntü
# ✓ metadata_train.json ve metadata_val.json
```

### Adım 3: Hızlı Test (1 epoch)
```bash
# Küçük model ile hızlı test
!python train.py \
    --config configs/colab_config.yaml \
    --data_dir data/cc12m_10k \
    --output_dir checkpoints/test_run \
    --experiment_name "AGIFORMER_Colab_Test" \
    --epochs 1 \
    --batch_size 2 \
    --no_wandb
```

### Adım 4: Tam Eğitim (Colab Optimize)
```bash
# Optimizasyonlu eğitim (3-5 epoch)
!python train.py \
    --config configs/colab_config.yaml \
    --data_dir data/cc12m_10k \
    --output_dir checkpoints/cc12m_colab_run1 \
    --experiment_name "AGIFORMER_v0.1_CC12M_10k_Colab" \
    --epochs 3 \
    --batch_size 4 \
    --use_wandb
```

### Adım 5: Uzun Eğitim (İsteğe Bağlı)
```bash
# Daha uzun eğitim için (Colab Pro gerekebilir)
!python train.py \
    --config configs/colab_config.yaml \
    --data_dir data/cc12m_10k \
    --output_dir checkpoints/cc12m_colab_run2 \
    --experiment_name "AGIFORMER_v0.1_CC12M_10k_Extended" \
    --epochs 10 \
    --batch_size 4 \
    --use_wandb
```

## 📊 İzlenecek Metrikler

### W&B Dashboard'da Kontrol Edilecekler:
1. **Loss Metrikleri**
   - `Training/loss` (düşmeli)
   - `Validation/loss` (düşmeli)
   - `Validation/perplexity` (düşmeli)

2. **MoE (Mixture of Experts) Metrikleri**
   - `Training/expert_usage_*_0` (Language expert)
   - `Training/expert_usage_*_1` (Logic expert)
   - `Training/expert_usage_*_2` (Spatial expert)
   - `Training/expert_usage_*_3` (Causal expert)
   - `Training/load_balancing_loss_*` (düşmeli veya stabil)

3. **Memory Sistemi**
   - `Training/memory_step_count` (artmalı)

4. **Introspection**
   - `Training/introspection_confidence_*` (0-1 arası)

5. **Multimodal**
   - `Training/multimodal_active` = 1
   - `Training/modality_image` = 1

## 🔧 Hata Ayıklama

### Yaygın Sorunlar ve Çözümleri:

#### 1. CUDA Out of Memory
```bash
# Çözüm: Batch size'ı düşür
--batch_size 2

# Çözüm: Gradient accumulation kullan
# (colab_config.yaml'de zaten ayarlı)
gradient_accumulation_steps: 4
```

#### 2. W&B Connection Error
```bash
# Çözüm: W&B olmadan çalıştır
--no_wandb

# Çözüm: W&B yeniden login
import wandb
wandb.login()
```

#### 3. Veri Yükleme Hatası
```bash
# Çözüm: Veri setini yeniden oluştur
!python scripts/prepare_cc12m.py --num_samples 1000 --output_dir data/cc12m_test

# Çözüm: Veri setini validate et
!python scripts/prepare_cc12m.py --validate --output_dir data/cc12m_10k
```

#### 4. Model Yükleme Hatası
```bash
# Çözüm: Konfigürasyonu kontrol et
!python -c "import yaml; print(yaml.safe_load(open('configs/colab_config.yaml')))"
```

## 📈 Başarı Kriterleri

### Minimum Başarı (1 epoch sonrası):
- [ ] Training loss < 5.0
- [ ] Validation loss < 5.5
- [ ] Hiçbir expert kullanımı 0 değil
- [ ] Memory step_count > 0
- [ ] Sistem çökmedi

### İyi Performans (3 epoch sonrası):
- [ ] Training loss < 3.0
- [ ] Validation loss < 3.5
- [ ] Validation perplexity < 35
- [ ] Expert kullanımı dengeli dağılmış
- [ ] Memory sistemi aktif çalışıyor

### Mükemmel Performans (5+ epoch):
- [ ] Training loss < 2.0
- [ ] Validation loss < 2.5
- [ ] Validation perplexity < 15
- [ ] Tüm uzmanlar aktif kullanılıyor
- [ ] Introspection confidence > 0.7

## 🎁 Ek Özellikler

### Checkpoint'ten Devam Etme:
```bash
# En son checkpoint'ten devam et
!python train.py \
    --config configs/colab_config.yaml \
    --data_dir data/cc12m_10k \
    --resume checkpoints/latest.pt \
    --epochs 5 \
    --use_wandb
```

### Farklı Veri Boyutları:
```bash
# Daha küçük veri seti için hızlı test
!python scripts/prepare_cc12m.py --num_samples 1000 --output_dir data/cc12m_1k

# Daha büyük veri seti için
!python scripts/prepare_cc12m.py --num_samples 50000 --output_dir data/cc12m_50k
```

## 🔍 İleri Analiz

### Model Davranışını Anlama:
```python
# Eğitim sonrası model analizi
import torch
from agiformer import AGIFORMER

# Modeli yükle
checkpoint = torch.load('checkpoints/best_model.pt')
model = AGIFORMER(use_multimodal=True)
model.load_state_dict(checkpoint['model_state_dict'])

# Test örneği
sample = {
    'image': torch.randn(1, 3, 224, 224),
    'input_ids': torch.randint(0, 256, (1, 50))
}

with torch.no_grad():
    logits, info = model(text=sample['input_ids'], image=sample['image'])
    
print("Model info keys:", info.keys())
print("Multimodal active:", info.get('multimodal', False))
print("Number of blocks:", len(info.get('blocks', [])))
```

## 🚀 Sonraki Adımlar

Başarılı eğitim sonrası:

1. **Modeli Kaydet**: Drive'a kopyala
2. **Sonuçları Analiz Et**: W&B dashboard
3. **Hiperparametre Optimize**: Farklı learning rate'ler dene
4. **Büyük Veri Setleri**: 50k+ örneklerle eğit
5. **Real CC12M**: Gerçek CC12M verisiyle eğit

---

**Hazır!** AGIFORMER'ı ilk gerçek eğitimine başlamak için yukarıdaki komutları sırasıyla çalıştırın. İyi eğitimler! 🚀
