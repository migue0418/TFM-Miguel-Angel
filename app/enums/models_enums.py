from enum import Enum


class ModelsEnum(str, Enum):
    BERT_BASE = "bert-base-uncased"
    MODERN_BERT = "answerdotai/ModernBERT-base"


class ModelsGenerativeEnum(str, Enum):
    GEMMA = "google/gemma-2-2b"
    LLAMA = "meta-llama/Llama-3.2-1B"
    LLAMA_SEXISM = "MatteoWood/llama-sexism-classifier-v1"  # Me peta el pc
    MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
    # MODERN_BERT = "answerdotai/ModernBERT-base"
