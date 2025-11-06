# Developer: inkbytefo
# Modified: 2025-11-06

import torch
from .relations import RELATION_TYPES

class PseudoLabeler:
    """
    Metin verisinden öz-denetimli bir şekilde ilişki etiketleri üretir.
    """
    def __init__(self):
        try:
            import spacy
            # Try Turkish model first, fallback to English
            try:
                self.nlp = spacy.load("tr_core_news_sm")
                print("spaCy Turkish model loaded successfully.")
            except IOError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("spaCy English model loaded as fallback.")
                except IOError:
                    print("WARNING: No spaCy model found. Install with: python -m spacy download en_core_web_sm")
                    self.nlp = None
        except ImportError:
            print("WARNING: spaCy not installed. Install with: pip install spacy")
            self.nlp = None

        # TODO: Önceden eğitilmiş Word2Vec modeli yüklenecek (örn. Gensim)
        self.word_vectors = None

    def generate_labels(self, tokens: list[str], token_embeddings: torch.Tensor) -> dict:
        """
        Bir cümledeki token'lar için potansiyel ilişkileri etiketler.

        Args:
            tokens (list[str]): Cümlenin token'ları.
            token_embeddings (torch.Tensor): Token'ların embedding'leri.

        Returns:
            dict: {(token1_idx, token2_idx): relation_id} formatında bir sözlük.
        """
        labels = {}
        sentence = " ".join(tokens)

        # 1. Sözdizimsel İlişkiler (spaCy ile)
        if self.nlp:
            doc = self.nlp(sentence)
            for token in doc:
                if token.head is not None and token.i != token.head.i:
                    # head -> child ilişkisi bir sözdizimsel bağdır
                    labels[(token.head.i, token.i)] = RELATION_TYPES["SYNTACTIC"]

        # 2. Nedensellik İlişkileri (Anahtar Kelime ile)
        causal_keywords = ["çünkü", "nedeniyle", "bu yüzden", "dolayısıyla"]
        for i, token in enumerate(tokens):
            if token in causal_keywords and i > 0 and i < len(tokens) - 1:
                # Basit kural: anahtar kelimenin solu ve sağı arasında nedensellik vardır
                labels[(i - 1, i + 1)] = RELATION_TYPES["CAUSALITY"]

        # 3. Benzerlik İlişkileri (Word Embedding ile)
        # TODO: Bu kısım word_vectors yüklendiğinde aktive edilecek.
        # if self.word_vectors:
        #     for i in range(len(tokens)):
        #         for j in range(i + 1, len(tokens)):
        #             # embedding'ler arası cosine similarity > 0.7 ise SIMILARITY
        #             pass

        return labels

    def classify_task_type(self, tokens: list[str]) -> int:
        """
        Cümlenin genel görev türünü sınıflandırır.

        Args:
            tokens (list[str]): Cümlenin token'ları.

        Returns:
            int: Görev türü ID'si (0: LINGUISTIC, 1: SYMBOLIC)
        """
        from .task_classifier import EXPERT_DOMAINS

        sentence = " ".join(tokens)

        # Sembolik görev göstergeleri
        symbolic_indicators = [
            "çünkü", "nedeniyle", "bu yüzden", "dolayısıyla",  # Nedensellik
            "eğer", "ise", "ise", "ise",  # Koşulluluk
            "tüm", "bazı", "hiçbir", "her",  # Niceleyiciler
            "mantıklı", "çıkarım", "sonuç", "çözüm"  # Mantık terimleri
        ]

        # Sembolik göstergelerin sayısı
        symbolic_count = sum(1 for token in tokens if token.lower() in symbolic_indicators)

        # Eğer sembolik göstergeler varsa veya ilişki etiketleri varsa SYMBOLIC
        if symbolic_count > 0 or self.generate_labels(tokens, torch.randn(len(tokens), 128)):
            return EXPERT_DOMAINS["SYMBOLIC"]
        else:
            return EXPERT_DOMAINS["LINGUISTIC"]
