import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import os
import torch
from app.core.config import settings
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum, ModelsGenerativeEnum
from app.utils.bias_classification import get_model_path
from app.utils.sexism_classification import (
    generate_4_sexism_labels_dataset,
    predict_sexism_generative_model,
    train_model_4_labels,
)


router = APIRouter(
    prefix="/sexism-detection",
    tags=["Sexism Detection"],
)


@router.post("/preprocessing/generate-dataset-with-4-sexism-labels")
def generate_dataset_with_4_sexism_labels():
    """
    Generate a dataset with 4 labels of sexism (not sexist, sexist (low confidence),
    sexist (high confidence), sexist) based on the individual annotations.
    Also generates 4 test dataset to evaluate the model and a reduced one for easier training.
    """
    try:
        # Generate the datasets
        generate_4_sexism_labels_dataset()

        return {"message": "The datasets has been generated successfully."}

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/training/model-4-sexism-grades")
def train_model_4_sexism_grades(
    dataset: DatasetEnum,
    model: ModelsEnum,
):
    """
    Train a sexism detection model of 4 grades with the specified dataset and model
    """
    try:
        # Train the model
        train_model_4_labels(dataset=dataset, model_name=model)

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


@router.post("/prediction/generative-ai")
def predict_generative_ai(
    dataset: DatasetEnum,
    model: ModelsGenerativeEnum,
    batch_size: int = 8,
    limit: int = 200,
):
    """
    Make predictions with a Generative AI model for sexism detection.
    """
    try:
        # Cargar el dataset
        df = pd.read_csv(dataset.csv_path)
        df["label_sexist"] = df["label_sexist"].map({"sexist": 1, "not sexist": 0})

        # Filtramos solo por los textos de test
        df = df[df["split"] == "test"]

        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            "generative_results",  # Los metemos dentro de la carpeta de generative_results para no mezclar
            dataset.model_folder_path,
            model.value.split("/")[-1],
        )
        predictions_file = os.path.join(
            result_path, f"predictions_{model.value.split("/")[-1]}.csv"
        )
        result_file = os.path.join(
            result_path, f"results_{model.value.split('/')[-1]}.csv"
        )

        # Crear el archivo si no está ya creado
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if not os.path.exists(predictions_file):
            # Si no está creado, crea un dataset vacio con las columnas "rewire_id" y "predictions"
            df_results = pd.DataFrame(
                columns=["rewire_id", "text", "label_sexist", "preds"]
            )
            df_results.to_csv(predictions_file, index=False)
        else:
            # Si lo está, cargar los datos que ya se han calculado
            df_results = pd.read_csv(predictions_file)

        # Obtener los datos del dataset (df) que todavía no se han calculado ordenados por rewire_id
        df_to_calculate = df[~df["rewire_id"].isin(df_results["rewire_id"])][
            ["rewire_id", "text", "label_sexist"]
        ].sort_values(by="rewire_id")

        # Limitamos a los 400 primeros
        df_to_calculate = df_to_calculate.head(limit)

        # Obtenemos los textos que no se han calculado
        texts = df_to_calculate["text"].tolist()

        # Obtenemos las labels que no se han calculado
        y_true = df_to_calculate["label_sexist"].tolist()

        # Calcular las predicciones que faltan
        df_to_calculate["preds"] = predict_sexism_generative_model(
            model_name=model, texts=texts, batch_size=batch_size
        )

        # Guardar los resultados en el archivo CSV con la lista de predicciones y la lista de rewire_id
        df_results = pd.concat([df_results, df_to_calculate])

        # Guardamos el archivo CSV con los resultados
        df_results.to_csv(predictions_file, index=False)

        # Cargar las métricas
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")

        # Evaluar los resultados
        accuracy_score = accuracy.compute(
            references=y_true, predictions=df_to_calculate["preds"].tolist()
        )["accuracy"]
        f1_score = f1.compute(
            references=y_true, predictions=df_to_calculate["preds"].tolist()
        )["f1"]
        precision_score = precision.compute(
            references=y_true, predictions=df_to_calculate["preds"].tolist()
        )["precision"]
        recall_score = recall.compute(
            references=y_true, predictions=df_to_calculate["preds"].tolist()
        )["recall"]
        results = (
            {
                "accuracy": accuracy_score,
                "f1_score": f1_score,
                "precision": precision_score,
                "recall": recall_score,
            },
        )

        # Guardar los resultados en el archivo CSV
        pd.DataFrame(results).to_csv(result_file, index=False)

        return {
            "message": f"Results with {model.name} for the dataset {dataset.name}",
            "results": results,
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/prediction/synthetic-dataset/generative-ai")
def predict_generative_ai_synthetic(
    dataset: DatasetEnum = DatasetEnum.SYNTHETIC_PHRASES,
    model: ModelsGenerativeEnum = ModelsGenerativeEnum.GEMMA,
    batch_size: int = 8,
    limit: int = 200,
):
    """
    Make predictions with a Generative AI model for sexism detection.
    """
    try:
        # Cargar el dataset
        df = pd.read_csv(dataset.csv_path)

        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            "generative_results",  # Los metemos dentro de la carpeta de generative_results para no mezclar
            dataset.model_folder_path,
            model.value.split("/")[-1],
        )
        predictions_file = os.path.join(
            result_path, f"predictions_{model.value.split("/")[-1]}.csv"
        )

        # Crear el archivo si no está ya creado
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if not os.path.exists(predictions_file):
            # Si no está creado, crea un dataset vacio con las columnas "rewire_id" y "predictions"
            df_results = pd.DataFrame(columns=["text", "manual_score", "preds"])
            df_results.to_csv(predictions_file, index=False)
        else:
            # Si lo está, cargar los datos que ya se han calculado
            df_results = pd.read_csv(predictions_file)

        # Obtener los datos del dataset (df) que todavía no se han calculado ordenados por rewire_id
        df_to_calculate = df[~df["text"].isin(df_results["text"])][
            ["text", "manual_score"]
        ]

        # Limitamos a los 400 primeros
        df_to_calculate = df_to_calculate.head(limit)

        # Obtenemos los textos que no se han calculado
        texts = df_to_calculate["text"].tolist()

        # Calcular las predicciones que faltan
        df_to_calculate["preds"] = predict_sexism_generative_model(
            model_name=model, texts=texts, batch_size=batch_size
        )

        # Guardar los resultados en el archivo CSV con la lista de predicciones y la lista de rewire_id
        df_results = pd.concat([df_results, df_to_calculate])

        # Guardamos el archivo CSV con los resultados
        df_results.to_csv(predictions_file, index=False)

        return {
            "message": f"Predictions with {model.name} for the dataset {dataset.name}",
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/generative-ai")
def evaluate_generation_ai_model(dataset: DatasetEnum, model: ModelsGenerativeEnum):
    """
    Make predictions with a Generative AI model for sexism detection.
    """
    try:
        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            "generative_results",  # Los metemos dentro de la carpeta de generative_results para no mezclar
            dataset.model_folder_path,
            model.value.split("/")[-1],
        )
        predictions_file = os.path.join(
            result_path, f"predictions_{model.value.split('/')[-1]}.csv"
        )
        result_file = os.path.join(
            result_path, f"results_{model.value.split('/')[-1]}.csv"
        )

        # Comprobamos si el archivo ya existe
        if not os.path.exists(predictions_file):
            raise HTTPException(status_code=404, detail="Result file not found.")

        # Cargar el dataset
        df = pd.read_csv(predictions_file)

        # Obtenemos las predicciones y los resultados reales
        y_true = df["label_sexist"].tolist()
        y_pred = df["preds"].tolist()

        # Cargar las métricas
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")

        # Evaluar los resultados
        accuracy_score = accuracy.compute(references=y_true, predictions=y_pred)[
            "accuracy"
        ]
        f1_score = f1.compute(references=y_true, predictions=y_pred)["f1"]
        precision_score = precision.compute(references=y_true, predictions=y_pred)[
            "precision"
        ]
        recall_score = recall.compute(references=y_true, predictions=y_pred)["recall"]
        results = (
            {
                "accuracy": accuracy_score,
                "f1_score": f1_score,
                "precision": precision_score,
                "recall": recall_score,
            },
        )

        # Guardar los resultados en el archivo CSV
        pd.DataFrame(results).to_csv(result_file, index=False)

        return {
            "message": f"Results with {model.name} for the dataset {dataset.name}",
            "results": results,
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/classification-models")
def evaluate_classification_model(
    dataset: DatasetEnum,
    model: ModelsEnum,
    unsure: bool = False,
    folder: str = "reddit_bias_edos_prediction",
    column_label: str = "label_reddit_bias",
):
    """
    Make predictions with a classification model for sexism detection.
    """
    try:
        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            folder,
            model.value.split("/")[-1],
        )
        predictions_file = os.path.join(
            result_path, f"predictions_{model.value.split('/')[-1]}.csv"
        )

        # Si no existe la carpeta, se crea
        if not os.path.exists(os.path.dirname(predictions_file)):
            os.makedirs(os.path.dirname(predictions_file))

        # Cargar el dataset
        df = pd.read_csv(dataset.csv_path)

        if dataset.name == "REDDIT_BIAS":
            # Cambiamos el nombre a las columnas de redditbias para que sean iguales a las de edos
            df = df.rename(
                columns={"comment": "text", "bias_sent": "label_reddit_bias"}
            )

        # Obtenemos el modelo entrenado con el dataset de edos
        dataset_edos = DatasetEnum.EDOS_REDUCED_FULL
        model_path = get_model_path(dataset_edos, model)

        # Creamos el pipeline de clasificación
        device = 0 if torch.cuda.is_available() else -1
        clf = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(model_path),
            tokenizer=AutoTokenizer.from_pretrained(model_path),
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=128,
        )

        # Inferencia en lotes
        texts = df["text"].tolist()
        batch_size = 32
        preds, score_not, score_sex = [], [], []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            outputs = clf(batch)  # lista de listas (dos dicts por ejemplo)

            for out in outputs:
                # se asume orden   0 → not sexist, 1 → sexist
                not_s, sex_s = out[0]["score"], out[1]["score"]
                score_not.append(not_s)
                score_sex.append(sex_s)
                preds.append("sexist" if sex_s >= not_s else "not sexist")

        # Añadimos los resultados al dataframe
        df["predict_edos"] = preds
        df["score_not"] = score_not
        df["score_sexist"] = score_sex

        if unsure:
            # Añadimos la columna unsure cuando la probabilidad está entre 0.4 y 0.6
            df["predict_edos"] = df.apply(
                lambda x: (
                    "unsure"
                    if (x["score_not"] >= 0.4 and x["score_not"] <= 0.6)
                    else x["predict_edos"]
                ),
                axis=1,
            )

        # Guardar los resultados en el archivo CSV
        cols_out = [
            "text",
            column_label,
            "predict_edos",
            "score_not",
            "score_sexist",
        ]
        df[cols_out].to_csv(predictions_file, index=False)

        return {
            "message": f"Results with {model.name} for the dataset {dataset.name}",
            "results": preds,
        }

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/classification-models-classes")
def evaluate_classification_model_classes(
    dataset: DatasetEnum,
    model: ModelsEnum,
    folder: str = "reddit_bias_edos_prediction",
    column_label: str = "label_reddit_bias",
):
    """
    Make predictions with a classification model for sexism detection.
    """
    try:
        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            folder,
            model.value.split("/")[-1],
        )
        predictions_file = os.path.join(
            result_path, f"predictions_{model.value.split('/')[-1]}.csv"
        )

        # Cargar el dataset
        df = pd.read_csv(predictions_file)

        # Agrupamos por label_sexist y predict para ver cuántos hay de cada uno
        df_grouped = (
            df.groupby([column_label, "predict_edos"]).size().reset_index(name="count")
        )

        # Obtenemos
        print(df_grouped)

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/evaluation/generative-models-classes")
def evaluate_generative_model_classes(
    dataset: DatasetEnum,
    model: ModelsGenerativeEnum,
    folder: str = "reddit_bias_edos_prediction",
    column_label: str = "label_reddit_bias",
):
    """
    Make predictions with a classification model for sexism detection.
    """
    try:
        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            folder,
            model.value.split("/")[-1],
        )
        predictions_file = os.path.join(
            result_path, f"predictions_{model.value.split('/')[-1]}.csv"
        )

        # Cargar el dataset
        df = pd.read_csv(predictions_file)
        df["predict_edos"] = df["predict_edos"].map(
            {1: "sexist", 0: "not sexist"}
        )  # Convertimos a 0 y 1

        # Agrupamos por label_sexist y predict para ver cuántos hay de cada uno
        df_grouped = (
            df.groupby([column_label, "predict_edos"]).size().reset_index(name="count")
        )

        # Obtenemos
        print(df_grouped)

    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Validation error: {', '.join([str(err) for err in e.errors()])}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
