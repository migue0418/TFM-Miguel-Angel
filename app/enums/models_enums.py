from enum import Enum


class ModelsEnum(str, Enum):
    BERT_BASE = "bert-base-uncased"
    MODERN_BERT = "answerdotai/ModernBERT-base"
