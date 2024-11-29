from app.schemas.redditbias import Topic
from app.core.config import redditbias_data_path as data_path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import ValidationError
from typing import Literal

from app.utils.redditbias_preprocessing import (
    get_raw_reddit_comments,
    reddit_data_phrases,
    reddit_data_phrases_replace_target,
    reddit_data_process,
    reddit_data_text_demo1_demo2,
    reddit_data_text_train_test,
    reddit_data_valid_test_reduced,
    reddit_reduce_for_annotation,
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
        get_raw_reddit_comments(topic_ins=topic_instance, size=size, chunks=chunks)

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
        reddit_data_process(topic_ins=topic_instance, process_demo1=process_demo1)

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
    Generates phrases from processed Reddit comments for an specific topic and
    store them in a CSV file.
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get and store the raw reddit comments
        reddit_data_phrases(
            topic_ins=topic_instance,
            remove_no_attribute_in_window=remove_no_attribute_in_window,
        )

        return {"message": "The phrases were successfully generated"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/process-phrases-for-annotation/{topic}")
def process_phrases_for_annotation_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"]
):
    """
    Phrases with attributes related to career and interests are retained
    from the earlier extracted Reddit phrases.
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get and store the raw reddit comments
        reddit_reduce_for_annotation(topic_ins=topic_instance)

        return {"message": "The phrases were successfully processed for annotation"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/replace-targets-phrases/{topic}")
def replace_targets_phrases_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    bias_type: str = "bias",
):
    """
    Extracts Reddit phrases manually annotated as Biased and corresponding
    generates Counter target dataset
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Replace the target in the phrases
        reddit_data_phrases_replace_target(
            topic_ins=topic_instance, bias_type=bias_type
        )

        return {"message": "The phrases' targets were successfully replaced"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/reddit-train-test-biased/{topic}")
def train_test_biased_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    bias_type: str = "bias",
):
    """
    This script generates csv and text files of train and test split for biased reddit dataset
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Create a train/test biased reddit dataset
        reddit_data_text_train_test(
            topic_ins=topic_instance, group="min", bias_type=bias_type
        )
        reddit_data_text_train_test(
            topic_ins=topic_instance, group="max", bias_type=bias_type
        )

        return {"message": "The phrases' targets were successfully replaced"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/reddit-data-valid-test-reduced/{topic}")
def data_valid_test_reduced_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"]
):
    """
    Create test set and validation set split on the test dataset with removed perplexity outliers
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Create a train/test biased reddit dataset
        reddit_data_valid_test_reduced(topic_ins=topic_instance)

        return {"message": "Successfully created the reduced dataset"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/reddit-data-text-demo1-demo2/{topic}")
def data_text_demo1_demo2(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"]
):
    """
    Generates text files of train datasets of Counter target data augmentation
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Creates counter target augmented data
        reddit_data_text_demo1_demo2(topic_ins=topic_instance)

        return {"message": "Successfully created the reduced dataset"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
