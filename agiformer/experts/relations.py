# Developer: inkbytefo
# Modified: 2025-11-06

# Temel anlamsal ve mantıksal ilişkiler
RELATION_TYPES = {
    "NONE": 0,          # İlişki yok veya belirsiz
    "SIMILARITY": 1,    # Anlamsal benzerlik (kedi ~ kaplan)
    "CAUSALITY": 2,     # Nedensellik (yağmur -> ıslaklık)
    "BELONGS_TO": 3,    # Aidiyet / Parça-bütün (parmak -> el)
    "SYNTACTIC": 4,     # Sözdizimsel bağımlılık (sıfat -> isim)
    # Gelecekte eklenebilecekler: CONTRADICTION, ENTAILMENT, TEMPORAL_AFTER, etc.
}

NUM_RELATIONS = len(RELATION_TYPES)
