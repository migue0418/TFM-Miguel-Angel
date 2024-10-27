import os
from dotenv import load_dotenv

# Cargar las variables de entorno del archivo .env
load_dotenv()


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "FastAPI - TFM Tests")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    OPEN_AI_KEY: str = os.getenv("OPEN_AI_KEY")


settings = Settings()
