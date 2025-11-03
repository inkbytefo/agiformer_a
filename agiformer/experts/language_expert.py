"""
Language Expert
Specialized for language understanding and generation
Now powered by pre-trained Qwen3-0.6B LLM
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
    Language Expert that leverages a pre-trained Large Language Model (LLM).
    This module acts as an interface between AGIFORMER's internal state
    and the powerful capabilities of a ready-made LLM.
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
        
        if not LLM_AVAILABLE:
            # Fallback implementation if transformers not available
            print("Warning: Using fallback LanguageExpert implementation")
            self.use_llm = False
            self.d_ff = d_ff or (d_model * 4)
            
            # Simple feed-forward as fallback
            self.fallback_ff = nn.Sequential(
                nn.Linear(d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_ff, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            self.use_llm = True
            
            # 1. Önceden eğitilmiş Qwen3-0.6B modelini yükle.
            print(f"Loading pre-trained language model: {model_name}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            
            # 2. LLM'in parametrelerini dondur (en önemli adım).
            #    AGIFORMER, LLM'i yeniden eğitmek yerine onu "nasıl kullanacağını" öğrenecek.
            for param in self.llm.parameters():
                param.requires_grad = False
            
            llm_hidden_size = self.llm.config.hidden_size
            
            # 3. Girdi Projeksiyonu (AGIFORMER -> LLM).
            #    AGIFORMER'ın iç vektörlerini, LLM'in embedding katmanının anlayacağı
            #    formata dönüştüren bir adaptör.
            self.input_projection = nn.Linear(d_model, llm_hidden_size)
            
            # 4. Çıktı Projeksiyonu (LLM -> AGIFORMER).
            #    LLM'in çıktı vektörlerini, tekrar AGIFORMER'ın kendi iç boyutuna
            #    dönüştüren bir adaptör.
            self.output_projection = nn.Linear(llm_hidden_size, d_model)
            
            self.layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        syntax_features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] - AGIFORMER'dan gelen iç durum.
            mask: Optional attention mask.
        
        Returns:
            torch.Tensor: LLM tarafından işlenmiş ve AGIFORMER formatına
                          geri dönüştürülmüş vektörler.
        """
        if self.use_llm:
            residual = x
            batch_size, seq_len = x.size(0), x.size(1)
            
            # AGIFORMER'ın durumunu LLM'in girdi formatına project et.
            llm_input_embeds = self.input_projection(x)
            
            # LLM'i dondurulmuş modda çalıştır
            with torch.no_grad():
                # .transformer veya .model demek yerine, doğrudan modelin kendisini çağır.
                # Bu, Hugging Face'in standart forward geçişini tetikler ve
                # Qwen2 gibi modern modellerle uyumluluğu sağlar.
                llm_outputs = self.llm(
                    inputs_embeds=llm_input_embeds,
                    attention_mask=mask,
                    output_hidden_states=True
                )

            # LLM'in son katman çıktısını al.
            llm_hidden_states = llm_outputs.hidden_states[-1]
            
            # LLM'in çıktısını tekrar AGIFORMER'ın boyutuna project et.
            processed_x = self.output_projection(llm_hidden_states)
            
            # Dropout ve residual connection ekleyerek sonucu birleştir.
            x = residual + self.dropout(processed_x)
            x = self.layer_norm(x)
            
            return x
        else:
            # Fallback implementation
            return self.fallback_ff(x)
