from fastapi import APIRouter, HTTPException
from app.external_services.reddit_client import get_reddit_client
from app.utils.reddit_fetcher import fetch_submission_data
from app.utils.reddit_fetcher import fetch_submissions_by_topic
from app.schemas.reddit import Submission
from typing import List


router = APIRouter(
    prefix="/reddit",
    tags=["Reddit"],
)


@router.get("/reddit/topic/{topic}", response_model=List[Submission])
def get_submissions_by_topic(topic: str, limit: int = 10):
    try:
        reddit_client = get_reddit_client(user_agent="get_reddit_posts")
        return fetch_submissions_by_topic(reddit_client, topic, limit)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Topic not found or limit too high: {str(e)}"
        )


@router.get(
    "/reddit/{submission_id}",
    response_model=Submission,
    summary="Obtiene la información de un submission específico",
)
def get_submission(submission_id: str):
    try:
        # Configurar cliente de Reddit
        reddit_client = get_reddit_client(user_agent="get_reddit_posts")
        # Obtiene la información del submission
        return fetch_submission_data(reddit_client, submission_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Submission not found: {str(e)}")
