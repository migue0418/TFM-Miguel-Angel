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
from fastapi import APIRouter, File, HTTPException, UploadFile
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


@router.post("/training/edos-reduced-and-redditbias-dataset")
def training_reduce_edos_and_reddit_bias_dataset():
    """
    Reduce the edos dataset for a quicker training of the model
    """
    try:
        # Get edos reduced dataset (10k samples)
        df_edos = pd.read_csv(DatasetEnum.EDOS_REDUCED_FULL.csv_path)
        # Filtramos las columnas necesarias
        df_edos = df_edos[["rewire_id", "text", "label_sexist", "split"]]

        # Get reddit bias dataset
        df_reddit_bias = pd.read_csv(DatasetEnum.REDDIT_BIAS.csv_path)

        # Cambiamos el nombre a las columnas de redditbias para que sean iguales a las de edos
        df_reddit_bias = df_reddit_bias.rename(
            columns={"comment": "text", "bias_sent": "label_sexist"}
        )

        # Cambiamos en label_sexist 0 por not_sexist y 1 por sexist
        df_reddit_bias["label_sexist"] = df_reddit_bias["label_sexist"].replace(
            {0: "not_sexist", 1: "sexist"}
        )

        # Filtramos las columnas necesarias
        df_reddit_bias = df_reddit_bias[["text", "label_sexist"]]

        # Filtramos el dataset para eliminar las label que no son sexist o not_sexist
        df_reddit_bias = df_reddit_bias[
            df_reddit_bias["label_sexist"].isin(["sexist", "not_sexist"])
        ]
        # Obtenemos el número de ejemplos por cada label_sexist
        counts = df_reddit_bias["label_sexist"].value_counts()
        # Obtener el número mínimo de ejemplos entre las dos clases
        min_count = counts.min()

        # Muestreamos el dataset de reddit bias para que tenga el mismo número de sexist y not_sexist
        df_reddit_bias = (
            df_reddit_bias.groupby("label_sexist")
            .apply(lambda x: x.sample(min(len(x), min_count), random_state=42))
            .reset_index(drop=True)
        )

        # Dividimos el dataset de reddit bias en 3 partes, train, dev y test quedándonos con un 70% train, 10% dev y 20% test
        # Además cada split debe tener el mismo número de sexist y not_sexist
        df_reddit_bias["split"] = "train"
        df_reddit_bias.loc[df_reddit_bias.sample(frac=0.1).index, "split"] = "dev"
        df_reddit_bias.loc[df_reddit_bias.sample(frac=0.2).index, "split"] = "test"

        print(df_reddit_bias["split"].value_counts())
        print(df_reddit_bias.groupby("split")["label_sexist"].value_counts())

        # Añadimos el indice de rewire_id al dataset de reddit bias
        df_reddit_bias["rewire_id"] = df_reddit_bias.index.to_series().apply(
            lambda i: f"sexism_reddit_bias-{i}"
        )

        # Concatenate the two dataframes
        data_concat = pd.concat([df_edos, df_reddit_bias])

        # Hacemos un shuffle para mezclar los datos
        data_concat = data_concat.sample(frac=1).reset_index(drop=True)

        # Save the reduced dataset
        output_path = files_path / "edos_reduced_reddit_bias.csv"
        data_concat.to_csv(output_path, index=False)

        return {"message": "The dataset has been reduced"}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/training/test-length-reduced-edos-dataset")
def training_test_length_reduce_edos_dataset():
    """
    Reduce the edos dataset for a quicker training of the model
    """
    try:
        import numpy as np

        df = pd.read_csv(DatasetEnum.EDOS_REDUCED_FULL.csv_path)

        output_path = files_path / "edos_reduced_test_length.csv"

        # 1. Filtrar solo el split "test"
        df_test = df[df["split"] == "test"].copy()

        # 2. Calcular la longitud de cada texto
        df_test["length"] = df_test["text"].str.len()

        # 3. Obtener mínimo y máximo
        min_len = df_test["length"].min()
        max_len = df_test["length"].max()

        # 4. Crear 4 intervalos equiespaciados entre min_len y max_len
        #    Necesitamos 5 puntos para 4 bins
        bins = np.linspace(min_len, max_len, num=5)

        # 5. Definir etiquetas para cada intervalo
        labels = [f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)]

        # 6. Asignar cada texto a un intervalo
        df_test["length_interval"] = pd.cut(
            df_test["length"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        # 7. Revisar distribución por intervalo
        print(df_test["length_interval"].value_counts().sort_index())

        # (Opcional) si quieres ver el DataFrame resultante:
        print(df_test[["text", "length", "length_interval"]].head())

        df_test.to_csv(output_path, index=False)

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


@router.post("/evaluation/test-samples-edos-reduced-length")
def evaluation_test_samples_edos_reduced_length(
    dataset: DatasetEnum = DatasetEnum.EDOS_REDUCED_FULL,
    model: ModelsEnum = ModelsEnum.MODERN_BERT,
):
    """
    Evaluate the test samples with the specified model
    """
    try:
        # Path donde está guardado el modelo
        model_path = get_model_path(dataset=dataset, model_name=model)

        # Leemos el dataset de test divido en intervalos de longitud
        dataset_csv = files_path / "edos_reduced_test_length.csv"
        df_test = pd.read_csv(dataset_csv)

        # Obtenemos los intervalos de longitud
        length_intervals = df_test["length_interval"].unique()
        for interval in length_intervals:
            # Obtiene la ruta donde se guardará la evaluación
            eval_path = (
                "app/files/edos_reduced_test_length"
                + "/"
                + model.name
                + "_test_length"
                + str(interval)
                + ".log"
            )

            metrics = sexism_evaluator_test_samples(
                eval_path=eval_path,
                model_path=model_path,
                csv_path=dataset_csv,
                interval_filter=str(interval),
            )

            print(f"Interval: {interval}, Metrics: {metrics}")

        return {"message": "The test samples have been evaluated"}
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


@router.post("/evaluation/metrics-test")
def evaluate_metrics_in_test(dataset: DatasetEnum, model: ModelsEnum):
    """
    Evaluate the consensus samples with the specified model
    """
    try:
        # Path donde está guardado el modelo
        model_path = get_model_path(dataset, model)

        # Obtiene la ruta donde se guardará la evaluación
        eval_path = (
            "app/models/evaluations/"
            + dataset.model_folder_path
            + "/"
            + model.name
            + ".log"
        )
        sexism_evaluator_test_samples(
            eval_path=eval_path, model_path=model_path, csv_path=dataset.csv_path
        )

        return {"message": "The model evaluations have been generated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/detection/train-model/{topic}", deprecated=True)
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


@router.post("/prediction/{topic}", deprecated=True)
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


@router.get("/prediction/detect-files-bias", deprecated=True)
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


@router.get("/prediction/detect-text-bias", deprecated=True)
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
