import pandas as pd
from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum
from app.schemas.redditbias import Topic
from app.utils.bias_classification import (
    generate_consensus_datasets,
    get_model_path,
    get_pretrained_model_bias_classificator,
    predict_bias,
    predict_sexism_batch,
    read_phrases_files,
    reduce_edos_dataset,
    sexism_evaluator_test_samples,
    train_model,
)
from app.core.config import files_path
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from typing import Literal


router = APIRouter(
    prefix="/bias",
    tags=["BIAS Detection"],
)


@router.post("/train-model")
def train_new_model(
    dataset: DatasetEnum,
    model: ModelsEnum,
):
    """
    Train a bias detection model with the specified dataset and model
    """
    try:
        # Train the model
        train_model(dataset=dataset, model=model)

        return {
            "message": f"Created the model {model.name} for the dataset {dataset.name}"
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/predict-sexism")
def predict_sexism_of_text(
    text: str,
    dataset: DatasetEnum = DatasetEnum.EDOS_REDUCED,
    model: ModelsEnum = ModelsEnum.MODERN_BERT,
):
    """
    Predict the sexism in the consensus datasets with the specified model
    """
    try:
        # Path donde está guardado el modelo
        model_path = get_model_path(dataset, model)

        # Lo convierte en lista porque el modelo espera una lista de textos
        texts_to_classify = [text]

        # Clasifica el texto
        output = predict_sexism_batch(model_path, texts_to_classify)

        # output será una lista de dicts con 'label' y 'score'
        return zip(texts_to_classify, output)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/tests/consensus/predict-sexism")
def predict_sexism_with_consensus_csv(
    dataset: DatasetEnum,
    model: ModelsEnum,
    consensus: Literal["all_sexist", "all_not_sexist", "mixed"],
):
    """
    Predict the sexism in the consensus datasets with the specified model
    """
    try:
        # Path donde está guardado el modelo
        model_path = get_model_path(dataset, model)

        # Obtiene los textos de los datasets de consenso
        consensus_path = files_path / f"edos_labelled_{consensus}.csv"
        texts_to_classify = pd.read_csv(consensus_path)["text"].tolist()

        # Clasifica todos los textos de una vez
        output = predict_sexism_batch(model_path, texts_to_classify)

        # output será una lista de dicts con 'label' y 'score'
        return zip(texts_to_classify, output)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/training/reduce-dataset")
def training_reduce_edos_dataset():
    """
    Reduce the edos dataset for a quicker training of the model
    """
    try:
        # Reduce the dataset
        reduce_edos_dataset()

        return {"message": "The dataset has been reduced"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/training/test-samples-consensus")
def training_test_samples_consensus():
    """
    Generate the consensus datasets for the test samples
    """
    try:
        # Generate the consensus datasets
        generate_consensus_datasets()

        return {"message": "The consensus datasets have been generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/test-consensus-sexism-samples")
def evaluate_consensus_samples(
    dataset: DatasetEnum,
    model: ModelsEnum,
    consensus: Literal["all_sexist", "all_not_sexist", "mixed"],
):
    """
    Evaluate the consensus samples with the specified model
    """
    try:
        # Path donde está guardado el modelo
        model_path = get_model_path(dataset, model)

        # Obtiene los textos de los datasets de consenso
        consensus_path = files_path / f"edos_labelled_{consensus}.csv"

        # Obtiene la ruta donde se guardará la evaluación
        eval_path = (
            "app/models/evaluations/"
            + dataset.model_folder_path
            + "/"
            + consensus
            + ".log"
        )
        sexism_evaluator_test_samples(
            eval_path=eval_path, model_path=model_path, csv_path=consensus_path
        )

        return {"message": "The consensus datasets have been generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/tmp/consensus-mixed-fix")
def tmp_consensus_mixed_fix(
    dataset: DatasetEnum,
    model: ModelsEnum,
    consensus: Literal["all_sexist", "all_not_sexist", "mixed"],
):
    """
    Arregla el archivo de consenso mixed cogiendo el valor label del archivo
    app/files/edos_labelled_aggregated.csv
    """
    try:
        # Obtiene el archivo de consenso mixed
        mixed_consensus_path = files_path / f"edos_labelled_{consensus}.csv"

        # Carga el dataframe
        mixed_consensus_df = pd.read_csv(mixed_consensus_path)

        # Obtiene el archivo de consenso agregado
        aggregated_consensus_path = files_path / "edos_labelled_aggregated.csv"

        # Carga el dataframe
        aggregated_consensus_df = pd.read_csv(aggregated_consensus_path)

        # Reemplaza la columna label_sexist del dataframe de consensus por la
        # del aggregated haciendo join por rewire_id
        mixed_consensus_df = mixed_consensus_df.merge(
            aggregated_consensus_df[["rewire_id", "label_sexist"]],
            on="rewire_id",
            how="left",
        )

        # Renombra la columna label_sexist_x a label_sexist y elimina label_sexist_y
        mixed_consensus_df = mixed_consensus_df.rename(
            columns={"label_sexist_x": "label_sexist"}
        )
        mixed_consensus_df = mixed_consensus_df.drop(columns=["label_sexist_y"])

        # Guarda el archivo
        mixed_consensus_df.to_csv(mixed_consensus_path, index=False)

        return {"message": "The consensus datasets have been generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
            # Get the reddit bias model probability
            prob = predict_bias(topic_ins=topic_instance, text=phrase["text"])
            phrase["reddit_bias_score"] = prob
            # Get the edos model probability
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
