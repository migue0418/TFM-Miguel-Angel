import evaluate
import pandas as pd
import os
from app.core.config import settings
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum, ModelsGenerativeEnum
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
