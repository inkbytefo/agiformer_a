"""
Language Expert
Specialized for language understanding and generation
Now powered by a SHARED pre-trained Qwen3-0.6B LLM instance
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import AutoModelForCausalLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: transformers not available. LanguageExpert will use fallback implementation.")

class LanguageExpert(nn.Module):
    """
    Language Expert that leverages a SINGLE, SHARED instance of a pre-trained LLM.
    This prevents multiple copies from being loaded into GPU memory.
    """
    # --- DEĞİŞİKLİK: Paylaşılan model ve adaptörler için sınıf değişkenleri ---
    _llm_instance = None
    _input_projection = None
    _output_projection = None
    _model_loaded = False
    # --- BİTTİ ---

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

        # Bu katmanlar her uzman için özel olabilir, bu yüzden 'self' içinde kalıyor
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Paylaşılan modeli yüklemek için çağrı yap
        self._load_shared_model()

    def _load_shared_model(self):
        """
        Loads the pre-trained LLM only ONCE and stores it in class variables.
        """
        # Sadece ilk uzman oluşturulduğunda modeli yükle
        if LanguageExpert._model_loaded:
            return

        if not LLM_AVAILABLE:
            raise ImportError("Hugging Face transformers library is required for the LanguageExpert.")

        print(f"🔄 Sharing and Lazily loading ONE instance of: {self.model_name}")
        print("⚠️  This will happen only once...")

        # 1. Modeli yükle
        llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 # Bellek kullanımı için float16'ya düşür
        )

        # 2. Parametreleri dondur
        for param in llm.parameters():
            param.requires_grad = False

        llm_hidden_size = llm.config.hidden_size

        # 3. Paylaşılan adaptör katmanlarını oluştur
        LanguageExpert._llm_instance = llm
        LanguageExpert._input_projection = nn.Linear(self.d_model, llm_hidden_size)
        LanguageExpert._output_projection = nn.Linear(llm_hidden_size, self.d_model)
        LanguageExpert._model_loaded = True
        
        print("✅ Shared Language model loaded successfully!")

    def forward(
        self,
        x: torch.Tensor,
        syntax_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # Modeli ve adaptörleri GPU'ya taşı (sadece ilk seferde)
        device = x.device
        if LanguageExpert._llm_instance.device != device:
            LanguageExpert._llm_instance.to(device)
            LanguageExpert._input_projection.to(device)
            LanguageExpert._output_projection.to(device)
            
        residual = x
        
        # AGIFORMER durumunu LLM girdi formatına project et
        llm_input_embeds = LanguageExpert._input_projection(x)

        # LLM'i dondurulmuş modda çalıştır
        with torch.no_grad():
            llm_outputs = Language.llm_instance(
                inputs_embeds=llm_input_embeds,
                attention_mask=mask,
                output_hidden_states=True
            )

        llm_hidden_states = llm_outputs.hidden_states[-1]

        # LLM çıktısını AGIFORMER boyutuna project et
        processed_x = LanguageExpert._output_projection(llm_hidden_states)

        x = residual + self.dropout(processed_x)
        x = self.layer_norm(x)

        return x
