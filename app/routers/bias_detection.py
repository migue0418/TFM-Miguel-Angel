import pandas as pd
from app.schemas.redditbias import Topic
from app.utils.bias_classification import (
    get_pretrained_model_bias_classificator,
    predict_bias,
    read_phrases_files,
)
from app.core.config import files_path
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from typing import Literal


router = APIRouter(
    prefix="/bias",
    tags=["BIAS Detection"],
)


@router.post("/detection/train-model/{topic}")
def train_bias_detection_model_from_topic(
    edos: bool = False,
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"] = None,
    force_training: bool = False,
):
    """
    Train a bias detection model from the annotated dataset extracted from RedditBias
    for each topic
    """
    try:
        # Instanciate and validate the topic
        topic_instance = None if edos else Topic(name=topic)

        # Train the model based on the topic
        get_pretrained_model_bias_classificator(
            topic_ins=topic_instance, force_training=force_training, edos=edos
        )

        return {"message": f"Created the model for the topic {topic}"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/prediction/{topic}")
def bias_prediction_from_topic(
    topic: Literal["gender", "race", "orientation", "religion1", "religion2"],
    text: str,
    edos: bool = False,
):
    """
    Get a bias probability based in a topic and a text
    """
    try:
        # Instanciate and validate the topic
        topic_instance = Topic(name=topic)

        # Get the bias probability of the text
        prob = predict_bias(topic_ins=topic_instance, edos=edos, text=text)

        return {"message": f"The text has a {prob} of being biased"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/prediction/detect-files-bias")
def bias_prediction_detect_files_bias():
    """
    Detect bias in files phrases and get a comparative of each model
    """
    try:
        # Instanciate and validate the gender topic
        topic_instance = Topic(name="gender")

        # Read the three phrases files with polars
        phrases_dict = read_phrases_files()

        # Iterate the phrases and get the bias probability of each one with each model
        for phrase in phrases_dict:
            # Get the gender model probability
            prob = predict_bias(topic_ins=topic_instance, text=phrase["text"])
            phrase["reddit_bias_score"] = prob
            # Get the gender model probability
            prob = predict_bias(
                topic_ins=topic_instance, edos=True, text=phrase["text"]
            )
            phrase["edos_score"] = prob

        # Get the bias probability of the text
        prob = predict_bias(topic_ins=topic_instance, text=phrase["text"])

        # Write the csv file with the results using pandas
        df = pd.DataFrame(phrases_dict)
        df.to_csv(files_path / "phrases_bias_detection.csv", index=False)

        return phrases_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
