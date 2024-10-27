def get_rewrite_prompt(text: str, style: str = "formal") -> list[str]:
    """
    Prompt para reescribir un documento dado un estilo
    """
    return [
        {
            "role": "system",
            "content": f"""Eres un asistente que reescribe textos.
            Tienes que escribir usando un estilo {style}.""",
        },
        {"role": "user", "content": f"Reescribe el siguiente texto:\n {text}"},
    ]
