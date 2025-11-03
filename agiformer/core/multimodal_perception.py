"""
Multimodal Perception Core
Handles different input modalities (text, image, audio, video) and maps them to a unified semantic space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .attention import CrossModalAttention

# Developer: inkbytefo
# Modified: 2025-11-03
from transformers import CLIPVisionModel, CLIPImageProcessor


class TextEncoder(nn.Module):
    """Text encoder for character-level or token-level input"""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_ids: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = text_ids.size(1)
        x = self.embedding(text_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        return self.layer_norm(x)


class ImageEncoder(nn.Module):
    """
    Image encoder using a pre-trained CLIPVisionModel.
    This acts as the "eye" of AGIFORMER, leveraging a powerful, ready-made vision system.
    """
    
    def __init__(self, d_model: int, model_name="openai/clip-vit-base-patch32"):
        """
        Args:
            d_model (int): The internal dimension of AGIFORMER.
            model_name (str): The name of the pre-trained CLIP model from Hugging Face.
        """
        super().__init__()
        
        # 1. Önceden eğitilmiş CLIP modelini ve görüntü işlemcisini yükle.
        #    Bu, internetten indirileceği için ilk çalıştırmada biraz zaman alabilir.
        print(f"Loading pre-trained vision model: {model_name}")
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # 2. CLIP modelinin parametrelerini dondur (isteğe bağlı ama önerilir).
        #    Bu, eğitim sırasında devasa görüntü modelini yeniden eğitmemizi engeller,
        #    böylece kaynaklarımızı AGIFORMER'ın orkestrasyon yeteneklerini öğrenmesine odaklarız.
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # 3. Projeksiyon katmanı. CLIP'in çıktı boyutunu (örn: 768) AGIFORMER'ın
        #    kendi iç boyutuna (d_model) dönüştürür. Bu, iki sistem arasında bir adaptör görevi görür.
        clip_output_dim = self.vision_model.config.hidden_size
        self.projection = nn.Linear(clip_output_dim, d_model)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of images and returns their embeddings in AGIFORMER's dimension.
        
        Args:
            images (torch.Tensor): A tensor of images with shape [batch_size, 3, H, W].
                                   Images should be in PIL format or a tensor.
        
        Returns:
            torch.Tensor: Image embeddings of shape [batch_size, num_patches, d_model].
        """
        # Görüntüleri CLIP modelinin anlayacağı formata getir.
        # Bu işlem normalizasyon, yeniden boyutlandırma gibi adımları içerir.
        inputs = self.processor(images=images, return_tensors="pt").to(images.device)
        
        # Görüntüleri dondurulmuş CLIP modelinden geçir.
        # `torch.no_grad()` kullanmak, gradyan hesaplanmayacağını garanti eder.
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
        
        # CLIP'in son katman çıktısını al. Bu, her bir görüntü "patch"i için bir vektör içerir.
        # Shape: [batch_size, num_patches + 1, clip_hidden_size] (+1 CLS token'ı için)
        patch_embeddings = outputs.last_hidden_state
        
        # Bu vektörleri AGIFORMER'ın iç boyutuna (d_model) project et.
        projected_embeddings = self.projection(patch_embeddings)
        
        return projected_embeddings


class AudioEncoder(nn.Module):
    """Audio encoder for raw audio waveforms"""
    
    def __init__(self, d_model: int, n_mels: int = 80, hop_length: int = 160):
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        
        # Mel-spectrogram feature extraction
        # In practice, you might use librosa or torchaudio
        self.spectrogram = nn.Sequential(
            nn.Conv1d(1, d_model // 4, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        
        # Positional encoding
        self.max_seq_len = 2048
        self.pos_encoding = nn.Parameter(torch.randn(1, self.max_seq_len, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [batch_size, audio_length] or [batch_size, 1, audio_length]
        Returns:
            [batch_size, seq_len, d_model]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [batch, 1, length]
        
        # Extract features
        x = self.spectrogram(audio)  # [batch, d_model, new_length]
        x = x.transpose(1, 2)  # [batch, new_length, d_model]
        
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        return self.layer_norm(x)


class VideoEncoder(nn.Module):
    """Video encoder combining spatial and temporal information"""
    
    def __init__(self, d_model: int, patch_size: int = 16, frame_size: int = 224):
        super().__init__()
        # Use image encoder for spatial features
        self.spatial_encoder = ImageEncoder(d_model)
        
        # Temporal modeling
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [batch_size, num_frames, 3, H, W]
        Returns:
            [batch_size, num_frames * num_patches, d_model]
        """
        batch_size, num_frames = video.size(0), video.size(1)
        
        # Encode each frame
        video = video.view(batch_size * num_frames, *video.shape[2:])
        spatial_features = self.spatial_encoder(video)  # [batch*frames, patches, d_model]
        
        # Reshape for temporal modeling
        num_patches = spatial_features.size(1)
        spatial_features = spatial_features.view(
            batch_size, num_frames, num_patches, spatial_features.size(-1)
        )  # [batch, frames, patches, d_model]
        
        # Temporal convolution along frame dimension
        # [batch, frames, patches, d_model] -> [batch, patches, frames, d_model]
        spatial_features = spatial_features.permute(0, 2, 1, 3).contiguous()
        spatial_features = spatial_features.view(batch_size * num_patches, num_frames, -1)
        
        # Temporal processing
        temporal_features = self.temporal_conv(spatial_features.transpose(1, 2)).transpose(1, 2)
        
        # Reshape back
        temporal_features = temporal_features.view(batch_size, num_patches, num_frames, -1)
        temporal_features = temporal_features.permute(0, 2, 1, 3).contiguous()
        
        # Flatten
        output = temporal_features.view(batch_size, num_frames * num_patches, -1)
        return self.layer_norm(output)


class MultimodalPerceptionCore(nn.Module):
    """
    Unified multimodal perception core
    Maps different modalities to a common semantic space
    """
    
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 256,
        n_cross_modal_layers: int = 2,
        n_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Individual encoders for each modality
        self.text_encoder = TextEncoder(vocab_size, d_model)
        self.image_encoder = ImageEncoder(d_model)
        self.audio_encoder = AudioEncoder(d_model)
        self.video_encoder = VideoEncoder(d_model)
        
        # Cross-modal attention layers for fusion
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_cross_modal_layers)
        ])
        
        # Unified projection to common space
        self.unified_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process multiple modalities and return unified representation
        
        Returns:
            modality_embeddings: Dict mapping modality names to embeddings
            unified_embedding: Combined representation
        """
        modality_embeddings = {}
        
        # Encode each modality
        if text is not None:
            modality_embeddings['text'] = self.text_encoder(text)
        
        if image is not None:
            modality_embeddings['image'] = self.image_encoder(image)
        
        if audio is not None:
            modality_embeddings['audio'] = self.audio_encoder(audio)
        
        if video is not None:
            modality_embeddings['video'] = self.video_encoder(video)
        
        # If only one modality, return it
        if len(modality_embeddings) == 1:
            single_emb = list(modality_embeddings.values())[0]
            unified = self.unified_proj(single_emb)
            unified = self.layer_norm(unified)
            return modality_embeddings, unified
        
        # Cross-modal fusion
        # Use text as query if available, otherwise use first available modality
        query_modality = modality_embeddings.get('text') or list(modality_embeddings.values())[0]
        
        # Apply cross-modal attention
        fused_features = query_modality
        for cross_modal_layer in self.cross_modal_layers:
            # Attend to all other modalities
            for key, value in modality_embeddings.items():
                if value is not query_modality:
                    fused_features = cross_modal_layer(
                        query_modality,
                        value,
                        value
                    )
        
        # Unified projection
        unified = self.unified_proj(fused_features)
        unified = self.layer_norm(self.dropout(unified))
        
        return modality_embeddings, unified
