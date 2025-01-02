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

        # Convert to DataFrame for analysis
        df = pd.DataFrame(phrases_dict)

        # Calculate the mean absolute error for each model and bias type
        df["reddit_bias_error"] = abs(df["manual_score"] - df["reddit_bias_score"])
        df["edos_error"] = abs(df["manual_score"] - df["edos_score"])

        results = {}
        for bias_type in ["high_bias", "neutral", "small_bias"]:
            bias_df = df[
                df["manual_score"]
                == {"high_bias": 1, "neutral": 0, "small_bias": 0.5}[bias_type]
            ]
            reddit_bias_mae = bias_df["reddit_bias_error"].mean()
            edos_mae = bias_df["edos_error"].mean()
            results[bias_type] = {
                "reddit_bias_error": reddit_bias_mae,
                "edos_error": edos_mae,
                "best_model": "reddit_bias" if reddit_bias_mae < edos_mae else "edos",
            }

        # Calculate overall mean absolute error
        overall_reddit_bias_mae = df["reddit_bias_error"].mean()
        overall_edos_mae = df["edos_error"].mean()
        results["overall"] = {
            "reddit_bias_error": overall_reddit_bias_mae,
            "edos_error": overall_edos_mae,
            "best_model": (
                "reddit_bias" if overall_reddit_bias_mae < overall_edos_mae else "edos"
            ),
        }

        # Write the csv file with the results using pandas
        df.to_csv(files_path / "phrases_bias_detection.csv", index=False)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/prediction/detect-text-bias")
def bias_prediction_detect_text_bias(text: str, text_score: float = None):
    """
    Detect bias in text and divide it in phrases for detecting. Get a comparative of each model
    """
    try:
        # Instanciate and validate the gender topic
        topic_instance = Topic(name="gender")

        # Calculate bias for the text
        results = {
            "text": text,
            "scores": {
                "manual_score": text_score,
                "edos_score": predict_bias(
                    topic_ins=topic_instance, edos=True, text=text
                ),
                "reddit_bias_score": predict_bias(topic_ins=topic_instance, text=text),
            },
            "phrases": [],
        }

        # Calculate the bias for each phrase
        for phrase in text.split("."):
            phrase = phrase.strip()
            if phrase:
                results["phrases"].append(
                    {
                        "text": phrase,
                        "scores": {
                            "edos_score": predict_bias(
                                topic_ins=topic_instance, edos=True, text=phrase
                            ),
                            "reddit_bias_score": predict_bias(
                                topic_ins=topic_instance, text=phrase
                            ),
                        },
                    }
                )

        # Calculate the mean absolute error for each model for the text and its phrases
        if text_score is not None:
            results["scores"]["edos_bias_error"] = abs(
                results["scores"]["manual_score"] - results["scores"]["edos_score"]
            )
            results["scores"]["reddit_bias_error"] = abs(
                results["scores"]["manual_score"]
                - results["scores"]["reddit_bias_score"]
            )

        # Calculate the mean score for each model with the phrases' bias score
        results["mean_score"] = {
            "edos_score": sum(
                [phrase["scores"]["edos_score"] for phrase in results["phrases"]]
            )
            / len(results["phrases"]),
            "reddit_bias_score": sum(
                [phrase["scores"]["reddit_bias_score"] for phrase in results["phrases"]]
            )
            / len(results["phrases"]),
        }

        # Get the best score for each model between the text score and the mean phrases score
        if text_score is not None:
            results["best_score"] = {
                "edos": (
                    "text_score"
                    if results["scores"]["edos_bias_error"]
                    < abs(
                        results["scores"]["manual_score"]
                        - results["mean_score"]["edos_score"]
                    )
                    else "phrases_score"
                ),
                "reddit_bias": (
                    "text_score"
                    if results["scores"]["reddit_bias_error"]
                    < abs(
                        results["scores"]["manual_score"]
                        - results["mean_score"]["reddit_bias_score"]
                    )
                    else "phrases_score"
                ),
            }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
