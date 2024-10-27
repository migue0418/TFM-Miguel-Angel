from enum import Enum


class GPTModel(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4o-mini"

    @classmethod
    def list_models(cls):
        """Devuelve una lista con los nombres de los modelos disponibles."""
        return [model.value for model in cls]
