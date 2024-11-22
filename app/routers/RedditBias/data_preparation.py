from app.schemas.redditbias import Topic
from app.core.config import data_path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError
from typing import Literal

from app.utils.redditbias_preprocessing import (
    get_raw_reddit_comments,
    reddit_data_phrases,
    reddit_data_process,
)


router = APIRouter(
    prefix="/reddit-bias/data-preparation",
    tags=["Reddit Bias: Data Preparation"],
)


@router.get("/get-raw-comments/{topic}")
def get_raw_comments_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    size: int = 400,
    chunks: int = 40,
):
    """
    Get comments from Reddit for an specific topic and store them in a CSV file.
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get and store the raw reddit comments
        get_raw_reddit_comments(topic_instance=topic_instance, size=size, chunks=chunks)

        return {"message": "The comments were successfully obtained and stored"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/download-reddit-raw-comments/{topic}/{group}/{iteration}")
async def download_reddit_comments(topic: str, group: str, iteration: int):
    """
    Download the generated CSV file for an specific topic, minority group and iteration.
    """
    file_path = (
        data_path / topic / f"reddit_comments_{topic}_{group}_raw_{iteration}.csv"
    )

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="text/csv", filename=file_path.name)


@router.post("/process-raw-comments/{topic}")
def process_raw_comments_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    process_demo1: bool = True,
):
    """
    Process raw comments from Reddit for an specific topic and store them in a CSV file.
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get and store the raw reddit comments
        reddit_data_process(topic_instance=topic_instance, process_demo1=process_demo1)

        return {"message": "The raw comments were successfully processed"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/process-phrases/{topic}")
def process_phrases_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    remove_no_attribute_in_window: bool = True,
):
    """
    Process raw comments from Reddit for an specific topic and store them in a CSV file.
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get and store the raw reddit comments
        reddit_data_phrases(
            topic_instance=topic_instance,
            remove_no_attribute_in_window=remove_no_attribute_in_window,
        )

        return {"message": "The phrases were successfully processed"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
