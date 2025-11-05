# Developer: inkbytefo
# Modified: 2025-11-06

import torch
import torch.nn as nn

# Uzman türlerini merkezi bir yerden yönetelim
EXPERT_DOMAINS = {"LINGUISTIC": 0, "SYMBOLIC": 1}
NUM_DOMAINS = len(EXPERT_DOMAINS)

class TaskTypeClassifier(nn.Module):
    """
    Girdi dizisinin doğasını (örn: dilsel, sembolik) sınıflandırır.
    """
    def __init__(self, d_model: int, num_domains: int = NUM_DOMAINS):
        super().__init__()
        # Basit bir sınıflandırıcı: Ortalama pooling + lineer katman
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_domains)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): Girdi dizisi. Shape: [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Her alan için logit'ler. Shape: [batch_size, num_domains]
        """
        # Tüm diziyi temsil etmek için ortalama al
        pooled_state = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_state)
        return logits
