import os
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

# Load the environment variables .env
load_dotenv()


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "FastAPI - TFM Tests")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    OPEN_AI_KEY: str = os.getenv("OPEN_AI_KEY")
    REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET")
    REDDITBIAS_DATA_PATH: str = "app/files/redditbias_data"
    REDDITBIAS_FILES_PATH: str = "app/files/redditbias_text_files"
    REDDITBIAS_EXECUTION_LOGS_PATH: str = "app/logs/execution_logs"
    FILES_PATH: str = "app/files"
    BIAS_MODEL_PATH = "app/models/bias_classifier"
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN")


settings = Settings()
redditbias_data_path = Path(settings.REDDITBIAS_DATA_PATH)
files_path = Path(settings.FILES_PATH)
redditbias_files_path = Path(settings.REDDITBIAS_FILES_PATH)
redditbias_execution_logs_path = Path(settings.REDDITBIAS_EXECUTION_LOGS_PATH)

# OAuth
ALGORITHM_JWT: Optional[str] = os.getenv("ALGORITHM_JWT")
SECRET_KEY_JWT: Optional[str] = os.getenv("SECRET_KEY_JWT")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 15))
# -- OAuth context & scheme
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
