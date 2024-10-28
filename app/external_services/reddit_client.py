import praw
from app.core.config import settings


def get_reddit_client(user_agent: str):
    """Configurar cliente de Reddit"""
    return praw.Reddit(
        client_id=settings.REDDIT_CLIENT_ID,
        client_secret=settings.REDDIT_CLIENT_SECRET,
        user_agent=user_agent,
    )
