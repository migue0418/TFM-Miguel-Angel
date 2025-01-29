import re


def clean_text_from_edos(text: str) -> str:
    """
    Limpia el texto (limpieza suave) de elementos innecesarios:
      1. Elimina tokens [URL] y [USER].
      2. (Opcional) Elimina o transforma hashtags.
      3. Limpia comillas duplicadas.
      4. Elimina espacios de más.
      5. (Opcional) Normaliza caracteres especiales o emojis.
    """
    # Eliminar [URL] y [USER]
    text = re.sub(r"\[URL\]", "", text)
    text = re.sub(r"\[USER\]", "", text)

    # Eliminar emojis o caracteres no estándar.
    text = re.sub(r"[^\w\s\,\.\!\?\-\']", "", text)

    # Normalización de mayúsculas/minúsculas
    text = text.lower()

    # Limpiar comillas duplicadas o secuencias de comillas
    text = re.sub(r"\"+", '"', text)

    # Eliminar espacios de más (espacios múltiples, espacios al inicio/fin)
    text = re.sub(r"\s+", " ", text).strip()

    return text
