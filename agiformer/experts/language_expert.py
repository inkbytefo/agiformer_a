# Developer: inkbytefo
# Modified: 2025-11-03

"""
Language Expert
Specialized for language understanding and generation
Now powered by pre-trained Qwen3-0.6B LLM with LAZY LOADING
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# LLM imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: transformers not available. LanguageExpert will use fallback implementation.")


class LanguageExpert(nn.Module):
    """
    Language Expert that leverages a pre-trained Large Language Model (LLM)
    with lazy loading to prevent multiple downloads and memory issues.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int = None,
        n_heads: int = 12,
        dropout: float = 0.1,
        model_name="Qwen/Qwen3-0.6B"
    ):
        super().__init__()
        self.d_model = d_model
        self.model_name = model_name

        # --- YENİ: Lazy Loading Değişkenleri ---
        self.llm = None
        self.input_projection = None
        self.output_projection = None
        self._model_loaded = False
        # --- BİTTİ ---

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _load_model(self):
        """
        Loads the pre-trained LLM only when it's first needed.
        """
        if self._model_loaded:
            return

        if not LLM_AVAILABLE:
            raise ImportError("Hugging Face transformers library is required for the LanguageExpert.")

        print(f"🔄 Lazily loading pre-trained language model: {self.model_name}")
        print("⚠️  This may take 1-2 minutes on the very first run...")

        # 1. Modeli yükle
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )

        # 2. Parametreleri dondur
        for param in self.llm.parameters():
            param.requires_grad = False

        llm_hidden_size = self.llm.config.hidden_size

        # 3. Adaptör katmanlarını oluştur
        self.input_projection = nn.Linear(self.d_model, llm_hidden_size)
        self.output_projection = nn.Linear(llm_hidden_size, self.d_model)

        # Modeli GPU'ya taşı
        device = next(self.layer_norm.parameters()).device
        self.llm.to(device)
        self.input_projection.to(device)
        self.output_projection.to(device)

        self._model_loaded = True
        print("✅ Language model loaded successfully!")

    def forward(
        self,
        x: torch.Tensor,
        syntax_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # Check if we should use LLM (lazy loading approach)
        if not LLM_AVAILABLE:
            # Fallback implementation if transformers not available
            if not hasattr(self, 'fallback_ff'):
                self.d_ff = self.d_ff or (self.d_model * 4)
                self.fallback_ff = nn.Sequential(
                    nn.Linear(self.d_model, self.d_ff),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.d_ff, self.d_model),
                    nn.LayerNorm(self.d_model)
                ).to(x.device)
            return self.fallback_ff(x)

        # --- YENİ: Modeli burada yükle ---
        self._load_model()
        # --- BİTTİ ---

        residual = x
        batch_size, seq_len = x.size(0), x.size(1)

        # AGIFORMER durumunu LLM girdi formatına project et
        llm_input_embeds = self.input_projection(x)

        # LLM'i dondurulmuş modda çalıştır
        with torch.no_grad():
            llm_outputs = self.llm(
                inputs_embeds=llm_input_embeds,
                attention_mask=mask,
                output_hidden_states=True
            )

        # LLM'in son katman çıktısını al
        llm_hidden_states = llm_outputs.hidden_states[-1]

        # LLM çıktısını AGIFORMER boyutuna project et
        processed_x = self.output_projection(llm_hidden_states)

        # Dropout ve residual connection ekle
        x = residual + self.dropout(processed_x)
        x = self.layer_norm(x)

        return x
