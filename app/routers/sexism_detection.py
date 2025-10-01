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
    generate_3_sexism_labels_dataset,
    predict_sexism_generative_model,
    predict_sexism_text,
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


@router.post("/preprocessing/generate-dataset-with-3-sexism-labels")
def generate_dataset_with_3_sexism_labels():
    """
    Generate a dataset with 3 labels of sexism (not sexist, unsure (high+low), sexist) based on the individual annotations.
    Also generates 3 test dataset to evaluate the model and a reduced one for easier training.
    """
    try:
        # Generate the datasets
        generate_3_sexism_labels_dataset()

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
    Train a sexism detection model of 4 grades with the specified dataset and model. Look Jupyter notebook for last version.
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
        label = "label_sexist" if dataset.name != "REDDIT_BIAS" else "label_reddit_bias"
        if dataset.name == "SYNTHETIC_PHRASES":
            label = "manual_score"
        preds = "preds" if dataset.name != "REDDIT_BIAS" else "predict_edos"
        y_true = df[label].tolist()
        y_pred = df[preds].tolist()

        # Cargar las métricas
        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")

        # Evaluar los resultados
        accuracy_score = accuracy.compute(references=y_true, predictions=y_pred)[
            "accuracy"
        ]
        f1_score = f1.compute(references=y_true, predictions=y_pred, average="macro")[
            "f1"
        ]
        precision_score = precision.compute(
            references=y_true, predictions=y_pred, average="macro"
        )["precision"]
        recall_score = recall.compute(
            references=y_true, predictions=y_pred, average="macro"
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


@router.post("/evaluation/classification-models")
def evaluate_classification_model(
    dataset: DatasetEnum,
    model: ModelsEnum,
    dataset_model: DatasetEnum = DatasetEnum.EDOS_REDUCED_FULL,
    unsure: bool = False,
    folder: str = "reddit_bias_edos_prediction",
    column_label: str = "label_reddit_bias",
):
    """
    Make predictions with a classification model for sexism detection.
    """
    try:
        threshold_unsure: tuple = (0.4, 0.6)
        # Obtener ruta de los resultados
        result_path = os.path.join(
            settings.FILES_PATH,
            folder,
            dataset.name,
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

        df = df[df["split"] == "test"]  # Filtramos solo por los textos de test

        # Obtenemos el modelo entrenado con el dataset de edos
        model_path = get_model_path(dataset_model, model)
        device = 0 if torch.cuda.is_available() else -1

        hf_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        labels = list(
            hf_model.config.id2label.values()
        )  # ['not sexist', 'sexist', ...]
        n_labels = len(labels)

        clf = pipeline(
            "text-classification",
            model=hf_model,
            tokenizer=hf_tokenizer,
            device=device,
            return_all_scores=True,
            truncation=True,
            max_length=128,
        )

        batch_size = 32
        texts = df["text"].tolist()

        # Dinámicamente creamos una columna por score_<etiqueta>
        id2label = hf_model.config.id2label
        label2id = {v: k for k, v in id2label.items()}
        labels = list(id2label.values())
        n_labels = len(labels)

        score_cols = {f"score_{lbl.replace(' ', '_')}": [] for lbl in labels}
        preds = []

        for i in range(0, len(texts), batch_size):
            outs = clf(texts[i : i + batch_size])
            for out in outs:
                score_dict = {item["label"]: item["score"] for item in out}

                # guardar scores
                for lbl in labels:
                    score_cols[f"score_{lbl.replace(' ', '_')}"].append(score_dict[lbl])

                # predicción principal
                pred_name = max(out, key=lambda x: x["score"])["label"]

                # opcional unsure (solo si 2 clases)
                if unsure and n_labels == 2:
                    pos_label = labels[1]  # la segunda suele ser la “positiva”
                    if (
                        threshold_unsure[0]
                        <= score_dict[pos_label]
                        <= threshold_unsure[1]
                    ):
                        pred_name = "unsure"

                preds.append(pred_name)

        # Metemos las predicciones en el dataset
        df["predict_edos"] = preds
        for col, values in score_cols.items():
            df[col] = values

        out_cols = ["text", column_label, "predict_edos"] + list(score_cols.keys())
        df[out_cols].to_csv(predictions_file, index=False)

        # Normaliza etiquetas reales y predichas a minúsculas
        y_true = df[column_label].astype(str).str.strip().str.lower()
        y_pred = df["predict_edos"].astype(str).str.strip().str.lower()

        # Clasificación y matriz de confusión (scikit-learn)
        from sklearn.metrics import classification_report, confusion_matrix

        cls_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        # Guarda las métricas a disco (opcional)
        (
            pd.DataFrame(cls_report)
            .T.round(4)
            .to_csv(
                os.path.join(result_path, "classification_report.csv"),
                index_label="class",
            )
        )

        # Calculamos la matriz de confusión
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

        import matplotlib.pyplot as plt
        import seaborn as sns

        # Pintamos la figura
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix")

        # La guardamos
        img_name = f"confusion_matrix.png"
        img_path = os.path.join(result_path, img_name)
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()  # libera memoria
        print(f"Matriz de confusión guardada en: {img_path}")

        return {
            "message": f"Results with {model.name} for the dataset {dataset.name}",
            "metrics": {
                "accuracy": round(cls_report["accuracy"], 4),
                "precision": round(cls_report["macro avg"]["precision"], 4),
                "recall": round(cls_report["macro avg"]["recall"], 4),
                "f1_macro": round(cls_report["macro avg"]["f1-score"], 4),
            },
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
            "app/results/reddit_bias",
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


@router.post("/sexism-detection/text")
def detect_sexism_in_text(
    model: ModelsEnum,
    text: str,
):
    """
    Detect sexism in a given text using a specified model.
    """
    try:
        return predict_sexism_text(texts=[text], model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
