from pathlib import Path
import evaluate
from collections import defaultdict
from fastapi.params import File
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, UploadFile
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.core.config import settings
from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum, ModelsGenerativeEnum
from app.utils.bias_classification import get_model_path, predict_sexism_batch
from app.utils.results import _best_run, get_latex_table, txt_to_int

router = APIRouter(
    prefix="/results",
    tags=["Results Functions"],
)


@router.get(
    "/hyperpatameters/get-best-result",
    summary="Get the best result from hyperparameter tuning",
)
async def get_best_results_hyperparameters(
    dataset: Literal["reduced_edos", "reduced_edos_10k"],
    model: Literal["bert-base-uncased", "ModernBERT-base"],
):
    try:
        # Import the results CSV file based on the dataset and model
        file_path = f"app/results/{dataset}/{model}/resultados.csv"

        # Get the best result
        results_df = _best_run(file_path)
        best_result = results_df.to_dict()

        return best_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get(
    "/hyperpatameters/dataset-latex-table",
    summary="Get the latex table for the best results from hyperparameter tuning",
)
async def get_dataset_table_best_results(
    dataset: Literal[
        "reduced_edos", "reduced_edos_10k", "reddit_bias", "synthetic_phrases"
    ],
    full_results: bool = False,
    caption: str = "Resultados de clasificación binaria en EDOS (reduced 5k)",
    dataset_model: DatasetEnum = DatasetEnum.EDOS_3_SEXISM_REDUCED,
):
    try:
        # Import the results CSV file based on the dataset and model
        latex_table = get_latex_table(
            dataset=dataset,
            full_results=full_results,
            caption=caption,
            dataset_model=dataset_model,
        )

        return latex_table
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Mapeo base (luego puede sobreescribirse)
MAP_TXT_NUM = {
    "sexist": 3,
    "not sexist": 0,
    "sexist (high confidence)": 2,
    "sexist (low confidence)": 1,
    "unsure": 1,
}

MAP_NUM_GOLD = {
    "0": 0,
    "0.0": 0,
    "1": 3,
    "1.0": 3,  # 1 = sexist
    "2": 1,
    "2.0": 1,  # 2 en redditbias
    "0.5": 1,  # 0.5 en synthetic_phrases
}


@router.post("/evaluation/classification-models")
def evaluate_classification_model(
    dataset: DatasetEnum,
    model: ModelsEnum,
    classification_prompt: Literal["3labels", "4labels", "binary"],
):
    """
    Evalúa cualquier archivo predictions_<model>.csv (2, 3 o 4 clases).
    Genera results_<model>.csv con classification_report completo.
    """
    try:
        dataset_folder = str(dataset.csv_path).split("\\")[-1].split(".csv")[0]
        folder = (
            "3_labels_classification"
            if "3_SEXISM" in dataset.name
            else "4_labels_classification"
        )
        base = model.value.split("/")[-1]
        result_dir = Path("app/results") / folder / dataset.name.lower() / base
        pred_csv = result_dir / f"predictions_{base}.csv"
        out_csv = result_dir / f"resultados.csv"

        if not pred_csv.exists():
            raise HTTPException(404, "predictions_*.csv no encontrado")

        df = pd.read_csv(pred_csv)

        label_col = (
            "manual_score"
            if dataset.name == "SYNTHETIC_PHRASES"
            else (
                "label_reddit_bias" if dataset.name == "REDDIT_BIAS" else "label_sexist"
            )
        )
        pred_col = "predict_edos"

        y_true_num = txt_to_int(MAP_NUM_GOLD, df[label_col])
        y_pred_num = txt_to_int(MAP_TXT_NUM, df[pred_col])

        if y_true_num.isna().any() or y_pred_num.isna().any():
            raise ValueError("Alguna etiqueta no coincide con el mapeo MAP_TXT_NUM")

        # sklearn acepta floats como etiquetas discretas
        y_true = y_true_num.tolist()
        y_pred = y_pred_num.tolist()

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        pd.DataFrame([metrics]).to_csv(out_csv, index=False)

        print("Resultados guardados en", out_csv)
        return metrics

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@router.post("/print-diagram")
def diagram_results(
    classification_prompt: Literal["3labels", "4labels", "binary"] = "binary",
):
    """
    Pinta una gráfica de barras con los resultados de los modelos
    en cada dataset.
    """
    try:
        datasets = [DatasetEnum.REDDIT_BIAS, DatasetEnum.SYNTHETIC_PHRASES]
        if classification_prompt == "3labels":
            datasets += [DatasetEnum.EDOS_3_SEXISM_REDUCED]
        elif classification_prompt == "4labels":
            datasets += [DatasetEnum.EDOS_4_SEXISM_REDUCED]
        elif classification_prompt == "binary":
            datasets += [DatasetEnum.EDOS_REDUCED_FULL, DatasetEnum.EDOS_REDUCED]

        # Ordenamos los datasets por nombre
        datasets = sorted(datasets, key=lambda d: d.name)

        results = []
        models = [model.value.split("/")[-1] for model in ModelsEnum] + [
            model.value.split("/")[-1] for model in ModelsGenerativeEnum
        ]
        for dataset in datasets:
            for model in models:
                if (
                    classification_prompt == "binary"
                    and dataset.name == "SYNTHETIC_PHRASES"
                ):
                    suffixs = ["_no_soft", "_soft_is_sexist"]
                else:
                    suffixs = [""]

                for suffix in suffixs:
                    folder = f"app/results/{str(dataset.csv_path).split('\\')[-1].split('.csv')[0]}"
                    # Leemos el CSV de resultados
                    csv_path = (
                        f"{folder}/results_{classification_prompt}_{model}{suffix}.csv"
                    )
                    if not os.path.exists(csv_path):
                        print(f"⨯  No file for {dataset.name}{suffix} | {model}")
                        continue

                    df = pd.read_csv(csv_path)

                    # Añadimos los resultados a la lista
                    results.append(
                        {
                            "dataset": dataset.name
                            + (
                                f" ({suffix.replace('_', ' ').strip()})"
                                if suffix
                                else ""
                            ),
                            "model": model,
                            "f1": df["f1"].iloc[0],
                        }
                    )

        # Convertimos a DataFrame para facilitar el manejo
        results_df = pd.DataFrame(results)

        # Paleta de colores que vamos a usar
        palette = {
            "bert-base-uncased": "#1f77b4",  # azul
            "ModernBERT-base": "#2ca02c",  # verde
            "gemma-2-2b": "#ff7f0e",  # naranja
            "Llama-3.2-1B": "#d62728",  # rojo
            "llama-sexism-classifier-v1": "#9467bd",  # morado
            "Mistral-7B-Instruct-v0.3": "#8c564b",  # marrón
        }

        dataset_order = (
            results_df["dataset"].unique().tolist()
        )  # orden deseado en eje X
        model_order = list(palette.keys())  # orden de la leyenda

        # Malla completa de dataset x model
        full_index = pd.MultiIndex.from_product(
            [dataset_order, model_order], names=["dataset", "model"]
        )

        full_df = (
            results_df.set_index(["dataset", "model"]).reindex(full_index).reset_index()
        )

        # Obtenemos la mejor barra
        full_df["is_best"] = full_df.groupby("dataset")["f1"].transform(
            lambda s: s == s.max()
        )

        # Convertimos a categóricos ordenados: esto hará que `.sort_values`
        # respete exactamente 'model_order' y 'dataset_order'.
        full_df["dataset"] = pd.Categorical(
            full_df["dataset"], categories=dataset_order, ordered=True
        )
        full_df["model"] = pd.Categorical(
            full_df["model"], categories=model_order, ordered=True
        )

        # Ordenamos por dataset y modelo
        full_df_sorted = full_df.sort_values(["model", "dataset"]).reset_index(
            drop=True
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=full_df_sorted,
            x="dataset",
            y="f1",
            hue="model",
            order=dataset_order,
            hue_order=model_order,
            palette=palette,
            ax=ax,
        )

        # Borde negro a la mejor
        for patch, row in zip(ax.patches, full_df_sorted.itertuples(index=False)):
            if row.is_best and not np.isnan(row.f1):
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)
                patch.set_zorder(3)

        ax.set_title("F1 por dataset y modelo")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("F1")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        ax.legend(
            title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
        )

        plt.tight_layout()
        fig.savefig(f"app/results/{classification_prompt}_results_diagram.png", dpi=300)
        plt.close(fig)

        return {"status": "OK"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post("/predictions/classification-models")
def predictions_classification_models(
    dataset: DatasetEnum,
    dataset_model: DatasetEnum,
    model: ModelsEnum,
    classification_prompt: Literal["3labels", "4labels", "binary"],
):
    """
    Genera un CSV de predicciones para el modelo y dataset especificados.
    """
    try:
        # Obtenemos la carpeta del dataset
        dataset_name = str(dataset.csv_path).split("\\")[-1].split(".csv")[0]
        carpeta = f"app/results/{dataset_name}"

        # Cargar el dataset
        df = pd.read_csv(dataset.csv_path)

        # Si el dataset es reddit_bias, renombramos la columna de texto y de label
        if dataset.name == "REDDIT_BIAS":
            df.rename(
                columns={"comment": "text", "bias_sent": "label_sexist"}, inplace=True
            )

        # Ordenamos por longitud del texto
        df = df.sort_values("text", key=lambda s: s.str.len())

        # Se queda solo con los de test si es EDOS
        if "edos" in dataset_name.lower():
            df = df[df["split"] == "test"]
        elif "reddit" in dataset_name.lower():
            df = df[df["label_sexist"] != 2]  # eliminamos esta que no dice nada

        # Archivo de predicciones
        base_name = model.value.split("/")[-1]
        predictions_file = (
            f"{carpeta}/predictions_{classification_prompt}_{base_name}.csv"
        )

        # Obtenemos la lista de textos
        texts = df["text"].tolist()

        # Path donde está guardado el modelo
        model_path = get_model_path(dataset_model, model)

        # Clasifica el texto
        df["preds"] = predict_sexism_batch(model_path, texts)

        # Aplicamos el mapeo de texto a número
        if classification_prompt == "binary":
            map_type = {
                "sexist": 1,
                "not sexist": 0,
            }
        elif classification_prompt == "3labels":
            map_type = {
                "sexist": 2,
                "not sexist": 0,
                "unsure": 1,
            }
        elif classification_prompt == "4labels":
            map_type = {
                "sexist (high confidence)": 2,
                "sexist (low confidence)": 1,
                "not sexist": 0,
                "sexist": 3,
            }

        df["preds"] = txt_to_int(map_type, df["preds"])

        # Guardamos el archivo CSV con los resultados
        df.to_csv(predictions_file, index=False)
        print("Predicciones guardadas en", predictions_file)

        return {
            "message": f"Terminado para {model.name}. Nuevas filas calculadas: {len(texts)}"
        }

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@router.post("/evaluation/generative-models")
def evaluate_classification_model(
    dataset: DatasetEnum,
    classification_prompt: Literal["3labels", "4labels", "binary"],
    caption: str = "Resultados de clasificación binaria en EDOS (reduced 5k)",
    suffix: Optional[str] = "",
):
    """
    Evalúa cualquier archivo predictions_<model>.csv (2, 3 o 4 clases).
    Genera results_<model>.csv con classification_report completo.
    """
    try:
        dataset_folder = str(dataset.csv_path).split("\\")[-1].split(".csv")[0]
        carpeta = f"app/results/{dataset_folder}"

        GENERATIVE_SHOW_COLS = [
            ("f1", "F1"),
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]

        MAP_PRED = {
            "0": 0,
            "0.0": 0,
            "1": 1,
            "1.0": 1,
            "2": 2,
            "2.0": 2,
            "3": 3,
            "3.0": 3,
        }

        if classification_prompt == "binary":
            if dataset.name == "SYNTHETIC_PHRASES":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 1,
                    "1.0": 1,  # 1 = sexist
                    "0.5": 1,  # 0.5 en synthetic_phrases
                }
            elif dataset.name == "REDDIT_BIAS":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 1,
                    "1.0": 1,  # 1 = sexist
                    "2": 0,
                    "2.0": 0,  # 2 en redditbias
                }
            elif "edos" in dataset.name.lower():
                MAP_TXT_NUM = {
                    "sexist": 1,
                    "not sexist": 0,
                }
        elif classification_prompt == "3labels":
            if dataset.name == "REDDIT_BIAS":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 1,
                    "1.0": 1,  # 1 = sexist
                    "2": 1,
                    "2.0": 1,  # 2 en redditbias es neutro
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 1,
                    "1.0": 1,  # unsure
                    "2": 1,
                    "2.0": 1,  # sexist
                }
            elif dataset.name == "SYNTHETIC_PHRASES":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 2,
                    "1.0": 2,  # 1 = sexist
                    "0.5": 1,  # 0.5 en synthetic_phrases
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 1,
                    "1.0": 1,  # unsure
                    "2": 2,
                    "2.0": 2,  # sexist
                }
            elif "edos" in dataset.name.lower():
                MAP_TXT_NUM = {
                    "sexist": 2,
                    "unsure": 1,
                    "not sexist": 0,
                }
        else:
            if dataset.name == "REDDIT_BIAS":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 3,
                    "1.0": 3,  # 1 = sexist
                    "2": 0,
                    "2.0": 0,  # 2 en redditbias es neutro
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 3,
                    "1.0": 3,  # low confidence
                    "2": 3,
                    "2.0": 3,  # high confidence
                    "3": 3,
                    "3.0": 3,  # sexist
                }
            elif dataset.name == "SYNTHETIC_PHRASES":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 2,
                    "1.0": 2,  # 1 = sexist
                    "0.5": 1,  # 0.5 en synthetic_phrases
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 1,
                    "1.0": 1,  # low confidence
                    "2": 1,
                    "2.0": 1,  # high confidence
                    "3": 2,
                    "3.0": 2,  # sexist
                }
            elif "edos" in dataset.name.lower():
                MAP_TXT_NUM = {
                    "sexist": 3,
                    "sexist (high confidence)": 2,
                    "sexist (low confidence)": 1,
                    "not sexist": 0,
                }

        # encabezados LaTeX
        header_cols = (
            "\\textbf{Modelo} & "
            + " & ".join("\\textbf{" + h + "}" for _, h in GENERATIVE_SHOW_COLS)
            + r" \\"
        )
        tabular = rf"\begin{{tabular}}{{lccccc}}"
        label = rf"\label{{tab:{dataset.name.lower()}_results}}"
        rows = []

        label_col = (
            "manual_score"
            if dataset.name == "SYNTHETIC_PHRASES"
            # else "label_reddit_bias" if dataset.name == "REDDIT_BIAS"
            else (
                "sexism_grade"
                if dataset.name in ("EDOS_4_SEXISM_REDUCED", "EDOS_3_SEXISM_REDUCED")
                else "label_sexist"
            )
        )
        pred_col = "preds"
        map_type = (
            MAP_NUM_GOLD
            if dataset.name in ("REDDIT_BIAS", "SYNTHETIC_PHRASES")
            else MAP_TXT_NUM
        )

        for model_name in [model.value.split("/")[-1] for model in ModelsEnum] + [
            model.value.split("/")[-1] for model in ModelsGenerativeEnum
        ]:
            # Obtener ruta de los resultados
            pred_csv = f"{carpeta}/predictions_{classification_prompt}_{model_name}.csv"
            out_csv = (
                f"{carpeta}/results_{classification_prompt}_{model_name}{suffix}.csv"
            )
            if not os.path.exists(pred_csv):
                raise HTTPException(404, "predictions_*.csv no encontrado")

            # ------- leer CSV ----
            df = pd.read_csv(pred_csv)

            # Quitamos las filas de 0.5
            df = df[df[label_col] != 0.5]

            y_true_num = txt_to_int(map_type, df[label_col])
            y_pred_num = txt_to_int(MAP_PRED, df[pred_col])

            # --- verificar que todas las etiquetas han sido mapeadas -------------
            if y_true_num.isna().any() or y_pred_num.isna().any():
                raise ValueError("Alguna etiqueta no coincide con el mapeo MAP_TXT_NUM")

            # sklearn acepta floats como etiquetas discretas
            y_true = y_true_num.tolist()
            y_pred = y_pred_num.tolist()

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )

            # Guardar resultados en CSV
            metrics = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
            pd.DataFrame([metrics]).to_csv(out_csv, index=False)

            # valores a mostrar en tabla
            values = [model_name] + [col for col, _ in GENERATIVE_SHOW_COLS]
            # formateo
            fmt_values = [
                "\\emph{" + model_name + "}",  # nombre con cursiva
                f"{f1:.3f}",  # F1
                f"{acc:.3f}",  # Acc.
                f"{prec:.3f}",  # Prec.
                f"{rec:.3f}",  # Rec.
            ]
            rows.append(" & ".join(fmt_values) + r" \\")

            # Calculamos el classification report
            from sklearn.metrics import classification_report

            out_cr = f"{carpeta}/classification_report_{classification_prompt}_{model_name}.csv"
            cls_report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            pd.DataFrame(cls_report).T.round(4).to_csv(out_cr, index_label="class")

            # --- calcula matriz ---------------------------------
            conf_mat = confusion_matrix(y_true, y_pred)

            import matplotlib.pyplot as plt
            import seaborn as sns

            # --- plot & save ------------------------------------
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.title("Confusion Matrix")

            # nombre de archivo con timestamp para evitar sobrescrituras
            img_path = (
                f"{carpeta}/confusion_matrix_{classification_prompt}_{model_name}.png"
            )
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close()  # libera memoria

            # --- opcional: registrar la ruta --------------------
            print(f"Matriz de confusión guardada en: {img_path}")

        # Crear la tabla LaTeX
        table = rf"""
\begin{{table}}[ht]
\centering
\setlength{{\tabcolsep}}{{6pt}}
{tabular}
\toprule
{header_cols}
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{{caption}}}
{label}
\end{{table}}

\FloatBarrier
""".strip()

        return table

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


import pandas as pd


@router.post("/appendix/results-table-bert")
def get_table_results_bert(
    csv_file: UploadFile = File(...),
    dataset: str = "edos5k",
    model: str = "bert-base-uncased",
):
    """Genera las tablas de los apéndices del documento"""
    df = pd.read_csv(csv_file.file)

    # Ponemos todos los epochs a 20
    df["num_epochs"] = 20

    # Eliminamos duplicados y conservamos la fila con mayor eval_f1
    df = df.loc[
        df.groupby(["num_epochs", "learning_rate", "weight_decay", "batch_size"])[
            "eval_f1"
        ].idxmax()
    ]

    # conserva solo algunas columnas y redondea
    cols = [
        "model_name",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "eval_loss",
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_runtime",
    ]
    alias_cols = [
        "Model Name",
        "Num Epochs",
        "Learning Rate",
        "Weight Decay",
        "Batch Size",
        "Ev. Loss",
        "Ev. Accuracy",
        "Ev. Precision",
        "Ev. Recall",
        "Ev. F1",
        "Ev. Runtime",
    ]
    df = df[cols].round(3)

    # resalta la mejor métrica F1
    best = df["eval_f1"].idxmax()
    df.loc[best, "model_name"] = r"\textbf{" + df.loc[best, "model_name"] + r"}"

    print(
        df.to_latex(
            longtable=True,
            index=False,
            escape=False,
            columns=cols,
            header=alias_cols,
            float_format="%.3f",
            caption=f"Resultados de {model} en el dataset {dataset}",
            label=f"tab:{dataset}-{model}-grid",
        )
    )


@router.post("/evaluation/grouped-by-correctness")
def evaluate_grouped_by_correctness(
    dataset: DatasetEnum,
    classification_prompt: Literal["3labels", "4labels", "binary"],
    model_gen: Optional[ModelsGenerativeEnum] = None,
    model_class: Optional[ModelsEnum] = None,
):
    """
    Evalúa cualquier archivo predictions_<model>.csv (2, 3 o 4 clases).
    Genera results_<model>.csv con classification_report completo.
    """
    try:
        # Obtenemos el nombre del modelo y el dataset
        if model_class is None and model_gen is None:
            raise HTTPException(
                400, "Debe especificar al menos un modelo (model_class o model_gen)"
            )
        model_name = (
            model_class.value.split("/")[-1]
            if model_class
            else model_gen.value.split("/")[-1]
        )
        dataset_folder = str(dataset.csv_path).split("\\")[-1].split(".csv")[0]
        carpeta = f"app/results/{dataset_folder}"

        MAP_PRED = {
            "0": 0,
            "0.0": 0,
            "1": 1,
            "1.0": 1,
            "2": 2,
            "2.0": 2,
            "3": 3,
            "3.0": 3,
        }

        if classification_prompt == "binary":
            if dataset.name == "SYNTHETIC_PHRASES":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 1,
                    "1.0": 1,  # 1 = sexist
                    "0.5": 1,  # 0.5 en synthetic_phrases
                }
            elif dataset.name == "REDDIT_BIAS":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 1,
                    "1.0": 1,  # 1 = sexist
                    "2": 0,
                    "2.0": 0,  # 2 en redditbias
                }
        elif classification_prompt == "3labels":
            if dataset.name == "REDDIT_BIAS":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 1,
                    "1.0": 1,  # 1 = sexist
                    "2": 1,
                    "2.0": 1,  # 2 en redditbias es neutro
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 1,
                    "1.0": 1,  # unsure
                    "2": 1,
                    "2.0": 1,  # sexist
                }
            elif dataset.name == "SYNTHETIC_PHRASES":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 2,
                    "1.0": 2,  # 1 = sexist
                    "0.5": 1,  # 0.5 en synthetic_phrases
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 1,
                    "1.0": 1,  # unsure
                    "2": 2,
                    "2.0": 2,  # sexist
                }
        else:
            if dataset.name == "REDDIT_BIAS":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 3,
                    "1.0": 3,  # 1 = sexist
                    "2": 0,
                    "2.0": 0,  # 2 en redditbias es neutro
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 3,
                    "1.0": 3,  # low confidence
                    "2": 3,
                    "2.0": 3,  # high confidence
                    "3": 3,
                    "3.0": 3,  # sexist
                }
            elif dataset.name == "SYNTHETIC_PHRASES":
                MAP_NUM_GOLD = {
                    "0": 0,
                    "0.0": 0,
                    "1": 2,
                    "1.0": 2,  # 1 = sexist
                    "0.5": 1,  # 0.5 en synthetic_phrases
                }
                MAP_PRED = {
                    "0": 0,
                    "0.0": 0,  # not sexist
                    "1": 1,
                    "1.0": 1,  # low confidence
                    "2": 1,
                    "2.0": 1,  # high confidence
                    "3": 2,
                    "3.0": 2,  # sexist
                }

        label_col = (
            "manual_score"
            if dataset.name == "SYNTHETIC_PHRASES"
            # else "label_reddit_bias" if dataset.name == "REDDIT_BIAS"
            else (
                "sexism_grade"
                if dataset.name in ("EDOS_4_SEXISM_REDUCED", "EDOS_3_SEXISM_REDUCED")
                else "label_sexist"
            )
        )
        pred_col = "preds"
        map_type = (
            MAP_NUM_GOLD
            if dataset.name in ("REDDIT_BIAS", "SYNTHETIC_PHRASES")
            else MAP_TXT_NUM
        )

        # Obtener ruta de los resultados
        pred_csv = f"{carpeta}/predictions_{classification_prompt}_{model_name}.csv"
        out_json = f"{carpeta}/phrases_{classification_prompt}_{model_name}.json"
        out_xlsx = f"{carpeta}/phrases_{classification_prompt}_{model_name}.xlsx"

        if not os.path.exists(pred_csv):
            raise HTTPException(404, f"{pred_csv} no encontrado")

        # ------------------------------------------------------------------ #
        # 1. Cargar CSV y mapear etiquetas
        # ------------------------------------------------------------------ #
        df = pd.read_csv(pred_csv)

        label_col = (
            "manual_score"
            if dataset.name == "SYNTHETIC_PHRASES"
            else (
                "sexism_grade"
                if dataset.name in ("EDOS_4_SEXISM_REDUCED", "EDOS_3_SEXISM_REDUCED")
                else "label_sexist"
            )
        )
        pred_col = "preds"

        # quitar las filas con 0.5 (solo se da en SYNTHECTIC_PHRASES)
        df = df[df[label_col] != 0.5]

        # Mapeos numéricos (MAP_PRED ya ajustado por prompt + dataset)
        y_true_num = txt_to_int(map_type, df[label_col])
        y_pred_num = txt_to_int(MAP_PRED, df[pred_col])

        TEXT_COL = "text" if "text" in df.columns else "phrase"

        # ------------------------------------------------------------------ #
        # 2. Agrupar -> phrases[clase_real][clase_pred].append(frase)
        # ------------------------------------------------------------------ #
        phrases = defaultdict(lambda: defaultdict(list))

        for true_c, pred_c, frase in zip(y_true_num, y_pred_num, df[TEXT_COL]):
            phrases[int(true_c)][int(pred_c)].append(frase)

        # convertir a dict “normal” para serializar
        phrases_jsonable = {
            str(k): {str(pk): v for pk, v in sub.items()} for k, sub in phrases.items()
        }

        # ------------------------------------------------------------------ #
        # 3. Guardar JSON
        # ------------------------------------------------------------------ #
        with open(out_json, "w", encoding="utf-8") as f_json:
            json.dump(phrases_jsonable, f_json, ensure_ascii=False, indent=2)

        # ------------------------------------------------------------------ #
        # 4. Guardar Excel
        # ------------------------------------------------------------------ #
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            for true_c, pred_dict in phrases.items():
                sheet_df = pd.DataFrame(
                    {
                        str(pred_c): pd.Series(texts)
                        for pred_c, texts in pred_dict.items()
                    }
                )
                sheet_df.to_excel(writer, sheet_name=f"true_{true_c}", index=False)

        # ------------------------------------------------------------------ #
        # 5. Devolver resultado
        # ------------------------------------------------------------------ #
        return {
            "phrases": phrases_jsonable,  # útil para código
            "excel_path": out_xlsx.replace("app/", ""),  # útil para humanos
        }

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


import os
import pandas as pd
from pathlib import Path
import re
from typing import List, Dict

# --------------------- 1. Diccionarios de mapeo ----------------------------
MAP_BIN = {"0": "not sexist", "1": "sexist"}
MAP_3LABEL = {"0": "not sexist", "1": "unsure", "2": "sexist"}
MAP_4LABEL = {"0": "not sexist", "1": "low conf.", "2": "high conf.", "3": "sexist"}
# gold label (true)
MAP_TRUE = {"sexist": "sexist", "1": "biased"}

# para escapado LaTeX
LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def tex_escape(text: str) -> str:
    return re.sub(r"[&%$#_{}~^\\]", lambda m: LATEX_SPECIAL[m.group()], text)


# --------------------- 2. Función principal --------------------------------
@router.post("/analysis/latex-comment-table")
def make_comment_table(
    dataset: str,
    label_col: str,
    models: List[str],
    comments: List[str],
    base_dir: str = "app/results",
) -> str:
    """
    Construye una tabla LaTeX con:
      • fila comentario (4 col) + true label (1 col)
      • fila cabecera Modelo / Binary / 3 / 4
      • una fila por modelo con las predicciones mapeadas a texto

    Asume ficheros CSV:
        {base_dir}/{dataset}/predictions_{prompt}_{model}.csv
    Y que cada CSV tiene columnas: text, preds (numéricas), label/gold (numérica)
    """
    prompts = ["binary", "3labels", "4labels"]
    map_dict = {"binary": MAP_BIN, "3labels": MAP_3LABEL, "4labels": MAP_4LABEL}

    # --- 2.1 Cargar todos los CSV en memoria -------------------------------
    dfs: Dict[str, Dict[str, pd.Series]] = {p: {} for p in prompts}
    gold = None
    for prompt in prompts:
        label_col = "label_sexist"
        if prompt == "3labels":
            label_col = "label_sexist_3"
        elif prompt == "4labels":
            label_col = "label_sexist_4"
        for model in models:
            real_dataset = dataset
            if dataset == "EDOS":
                real_dataset = (
                    "edos_labelled_reduced_10k"
                    if prompt == "binary"
                    else (
                        "edos_labelled_3_sexism_grade_reduced"
                        if prompt == "3labels"
                        else "edos_labelled_4_sexism_grade_reduced"
                    )
                )

            path = Path(base_dir) / real_dataset / f"predictions_{prompt}_{model}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            # guardamos por tupla (prompt, model)
            dfs[prompt][model] = df.set_index("text")["preds"]
            # gold una vez basta
            if gold is None and label_col in df.columns:
                gold = df.set_index("text")[label_col]

    if gold is None:
        raise ValueError("No se encontró la columna 'label' con el oro.")

    wrong, right = consensus_examples(dfs, gold)
    print(f"{len(right)} consensos correctos · {len(wrong)} consensos erróneos")
    # → úsalo para elegir comentarios representativos:
    sample_wrong = wrong[:5]
    sample_right = right[:5]

    # --- 2.2 Generar bloques LaTeX ----------------------------------------
    blocks = []
    for cmt in comments:
        # Escapar la frase
        cmt_tex = tex_escape(cmt)
        # True label
        true_lab = gold.loc[cmt]

        # «Comment» + «True label»
        tmpl_comment = (
            #  -- apertura multicolumn -----------------------------
            r"\multicolumn{{4}}{{@{{}}>{{\raggedright\arraybackslash}}p{{"
            r"\dimexpr 4.4\linewidth/5 - 4\tabcolsep\relax}}@{{}}}}"  # ← cierre columna
            #  -- la frase -----------------------------------------
            r"{{\emph{{ {text} }}}} & "  # llaves dobles = llaves reales
            #  -- true label ---------------------------------------
            r"\textbf{{{label}}}\\"
        ).format(text=cmt_tex, label=true_lab)

        # Cabecera de predicciones
        tmpl_header = (
            r"\multicolumn{2}{@{}>{\raggedright\arraybackslash}p{"
            r"\dimexpr 2.3\linewidth/5 - 4\tabcolsep\relax}@{}}"  # cierra multicolumn
            r"{\textbf{Model}}"
            r" & \textbf{Binary} & \textbf{3 labels} & \textbf{4 labels}\\"
        )

        block = [tmpl_comment, r"\addlinespace[0.6ex]", tmpl_header, r"\midrule"]

        # filas modelo
        for model in models:
            preds = []
            for prompt in prompts:
                num = int(dfs[prompt][model].loc[cmt])
                preds.append(map_dict[prompt].get(str(num), "NA"))
            model_tex = tex_escape(model)
            block.append(
                rf"\multicolumn{{2}}{{l}}{{{model_tex}}} & " + " & ".join(preds) + r"\\"
            )

        block.append(r"\addlinespace[1.5ex]")
        block.append(r"\toprule\toprule")
        block.append(r"\addlinespace[1.5ex]")
        blocks.append("\n".join(block))

    # --- 2.3 Envolver todo con tabularx ------------------------------------
    body = "\n".join(blocks).rstrip(r"\toprule\toprule").rstrip()
    table = rf"""
\begin{{table}}[ht]
\small
\centering
\begin{{tabularx}}{{\textwidth}}{{c c c c c}}
\toprule
\multicolumn{{4}}{{c}}{{\textbf{{Comment}}}} & \textbf{{True label}}\\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\caption{{Ejemplos de frases con sus predicciones, dataset: XXXXXXX.}}
\label{{tab:{dataset.lower()}_comments_analysis}}
\end{{table}}
""".strip()
    return table


from functools import reduce
from typing import Tuple, List, Dict


def consensus_examples(
    dfs: Dict[str, Dict[str, pd.Series]], gold: pd.Series
) -> Tuple[List[str], List[str]]:
    """
    Devuelve (consensus_wrong, consensus_right).
    Cada frase debe:
        • estar presente en todas las Series,
        • tener la MISMA predicción en cada Series,
        • diferir / coincidir con la etiqueta real según corresponda.
    """
    # 1. textos presentes en todos los dataframes
    all_indices = [
        set(series.index)
        for prompt_dict in dfs.values()
        for series in prompt_dict.values()
    ]
    common_texts = reduce(set.intersection, all_indices)

    consensus_wrong, consensus_right = [], []

    for text in common_texts:
        # 2. recoger todas las predicciones de esa frase
        preds = [
            int(series.loc[text])
            for prompt_dict in dfs.values()
            for series in prompt_dict.values()
        ]

        # 3. comprobar consenso
        if len(set(preds)) == 1:  # unanimidad
            pred = preds[0]
            true = int(gold.loc[text])
            if pred == true:
                consensus_right.append(text)
            else:
                consensus_wrong.append(text)

    return consensus_wrong, consensus_right


@router.post("/tmp/create-comments-label-table")
def labels_tables():
    try:
        # Primero leemos el df de EDOS 10k
        df = pd.read_csv(DatasetEnum.EDOS_REDUCED_FULL.csv_path)

        # Obtenemos el dataframe donde está la etiqueta final
        df_aggregated = pd.read_csv("app/files/edos_labelled_aggregated.csv")

        # Filtramos el df_aggregated para quedarnos solo con las frases que están en el df de EDOS 10k
        df = df_aggregated[df_aggregated["rewire_id"].isin(df["rewire_id"])]

        # Obtenemos el dataframe de las 60k anotaciones
        csv_full_anotations = "app/files/edos_labelled_individual_annotations.csv"
        df_full = pd.read_csv(csv_full_anotations)

        # Nos quedamos con las frases que están en el df de EDOS 10k
        df_full = df_full[df_full["rewire_id"].isin(df["rewire_id"])]

        def classify_4_labels(label_list):
            # Hacemos un count de sexist
            count_sexist = label_list.count("sexist")
            # Si tiene 3 es "sexist"
            if count_sexist == 3:
                return "sexist"
            # Si tiene 2 es "sexist (high confidence)"
            if count_sexist == 2:
                return "sexist (high confidence)"
            # Si tiene 1 es "sexist (low confidence)"
            if count_sexist == 1:
                return "sexist (low confidence)"
            # Si tiene 0 es "not sexist"
            return "not sexist"

        def classify_3_labels(label_list):
            # Hacemos un count de sexist
            count_sexist = label_list.count("sexist")
            # Si tiene 3 es "sexist"
            if count_sexist == 3:
                return "sexist"
            # Si tiene 2 o 1 es "unsure"
            if count_sexist == 2 or count_sexist == 1:
                return "unsure"
            # Si tiene 0 es "not sexist"
            return "not sexist"

        # Agrupamos por texto y obtenemos la label_sexism
        df_full["label_sexist_4"] = df_full.groupby("rewire_id")[
            "label_sexist"
        ].transform(lambda x: classify_4_labels(x.tolist()))
        df_full["label_sexist_3"] = df_full.groupby("rewire_id")[
            "label_sexist"
        ].transform(lambda x: classify_3_labels(x.tolist()))

        # Guardamos el dataframe con las etiquetas, agrupamos primero por texto para obtener una sola fila por texto
        df_full = df_full.groupby("rewire_id").first().reset_index()

        # Nos quedamos solo con las columnas que nos interesan
        df_full = df_full[["rewire_id", "label_sexist_4", "label_sexist_3"]]

        # Mergeamos con el df de EDOS 10k para obtener la columna "sexism_grade"
        df_full = df_full.merge(df, on="rewire_id", how="left")

        # Guardamos el dataframe de 3 etiquetas y 4 etiquetas
        df_full.to_csv("app/files/edos_labelled_sexism_grade_reduced.csv", index=False)

        return True

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@router.post("/predictions/classification-models-sexism")
def predictions_classification_models_sexism():
    """
    Genera un CSV de predicciones para el modelo y dataset especificados.
    """
    try:
        # Obtenemos la carpeta del dataset
        dataset_name = "edos_labelled_sexism_grade_reduced"
        carpeta = f"app/results/{dataset_name}"

        # Cargar el dataset
        df = pd.read_csv("app/files/edos_labelled_sexism_grade_reduced.csv")

        # Ordenamos por longitud del texto
        df = df.sort_values("text", key=lambda s: s.str.len())

        # Se queda solo con los de test si es EDOS
        if "edos" in dataset_name.lower():
            df = df[df["split"] == "test"]
        elif "reddit" in dataset_name.lower():
            df = df[df["label_sexist"] != 2]  # eliminamos esta que no dice nada

        for model in ModelsEnum:
            for classification_prompt in ["binary", "3labels", "4labels"]:
                dataset_model = DatasetEnum.EDOS_REDUCED_FULL
                if classification_prompt == "3labels":
                    dataset_model = DatasetEnum.EDOS_3_SEXISM_REDUCED
                elif classification_prompt == "4labels":
                    dataset_model = DatasetEnum.EDOS_4_SEXISM_REDUCED

                # Archivo de predicciones
                base_name = model.value.split("/")[-1]
                predictions_file = (
                    f"{carpeta}/predictions_{classification_prompt}_{base_name}.csv"
                )

                # Obtenemos la lista de textos
                texts = df["text"].tolist()

                # Path donde está guardado el modelo
                model_path = get_model_path(dataset_model, model)

                # Clasifica el texto
                df["preds"] = predict_sexism_batch(model_path, texts)

                # Aplicamos el mapeo de texto a número
                if classification_prompt == "binary":
                    map_type = {
                        "sexist": 1,
                        "not sexist": 0,
                    }
                elif classification_prompt == "3labels":
                    map_type = {
                        "sexist": 2,
                        "not sexist": 0,
                        "unsure": 1,
                    }
                elif classification_prompt == "4labels":
                    map_type = {
                        "sexist (high confidence)": 2,
                        "sexist (low confidence)": 1,
                        "not sexist": 0,
                        "sexist": 3,
                    }

                df["preds"] = txt_to_int(map_type, df["preds"])

                # Guardamos el archivo CSV con los resultados
                df.to_csv(predictions_file, index=False)
                print("Predicciones guardadas en", predictions_file)

        return {
            "message": f"Terminado para {model.name}. Nuevas filas calculadas: {len(texts)}"
        }

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@router.post("/evaluation/grouped-by-correctness-edos")
def evaluate_grouped_by_correctness_edos(
    classification_prompt: Literal["3labels", "4labels", "binary"],
    model_gen: Optional[ModelsGenerativeEnum] = None,
    model_class: Optional[ModelsEnum] = None,
):
    """
    Evalúa cualquier archivo predictions_<model>.csv (2, 3 o 4 clases).
    Genera results_<model>.csv con classification_report completo.
    """
    try:
        # Obtenemos el nombre del modelo y el dataset
        if model_class is None and model_gen is None:
            raise HTTPException(
                400, "Debe especificar al menos un modelo (model_class o model_gen)"
            )
        model_name = (
            model_class.value.split("/")[-1]
            if model_class
            else model_gen.value.split("/")[-1]
        )
        dataset_folder = "edos_labelled_sexism_grade_reduced"
        carpeta = f"app/results/{dataset_folder}"

        MAP_PRED = {
            "0": 0,
            "0.0": 0,
            "1": 1,
            "1.0": 1,
            "2": 2,
            "2.0": 2,
            "3": 3,
            "3.0": 3,
        }

        label_col = "label_sexist"
        if classification_prompt == "3labels":
            label_col = "label_sexist_3"
        elif classification_prompt == "4labels":
            label_col = "label_sexist_4"

        pred_col = "preds"
        map_type = MAP_TXT_NUM

        # Obtener ruta de los resultados
        pred_csv = f"{carpeta}/predictions_{classification_prompt}_{model_name}.csv"
        out_json = f"{carpeta}/phrases_{classification_prompt}_{model_name}.json"
        out_xlsx = f"{carpeta}/phrases_{classification_prompt}_{model_name}.xlsx"

        if not os.path.exists(pred_csv):
            raise HTTPException(404, f"{pred_csv} no encontrado")

        # ------------------------------------------------------------------ #
        # 1. Cargar CSV y mapear etiquetas
        # ------------------------------------------------------------------ #
        df = pd.read_csv(pred_csv)

        # Mapeos numéricos (MAP_PRED ya ajustado por prompt + dataset)
        y_true_num = txt_to_int(map_type, df[label_col])
        y_pred_num = txt_to_int(MAP_PRED, df[pred_col])

        TEXT_COL = "text" if "text" in df.columns else "phrase"

        # ------------------------------------------------------------------ #
        # 2. Agrupar -> phrases[clase_real][clase_pred].append(frase)
        # ------------------------------------------------------------------ #
        phrases = defaultdict(lambda: defaultdict(list))

        for true_c, pred_c, frase in zip(y_true_num, y_pred_num, df[TEXT_COL]):
            phrases[int(true_c)][int(pred_c)].append(frase)

        # convertir a dict “normal” para serializar
        phrases_jsonable = {
            str(k): {str(pk): v for pk, v in sub.items()} for k, sub in phrases.items()
        }

        # ------------------------------------------------------------------ #
        # 3. Guardar JSON
        # ------------------------------------------------------------------ #
        with open(out_json, "w", encoding="utf-8") as f_json:
            json.dump(phrases_jsonable, f_json, ensure_ascii=False, indent=2)

        # ------------------------------------------------------------------ #
        # 4. Guardar Excel
        # ------------------------------------------------------------------ #
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            for true_c, pred_dict in phrases.items():
                sheet_df = pd.DataFrame(
                    {
                        str(pred_c): pd.Series(texts)
                        for pred_c, texts in pred_dict.items()
                    }
                )
                sheet_df.to_excel(writer, sheet_name=f"true_{true_c}", index=False)

        # ------------------------------------------------------------------ #
        # 5. Devolver resultado
        # ------------------------------------------------------------------ #
        return {
            "phrases": phrases_jsonable,  # útil para código
            "excel_path": out_xlsx.replace("app/", ""),  # útil para humanos
        }

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")


@router.post("/evaluation/phrases-get-result")
def evaluation_phrases_get_result(
    phrases: List[str],
    model_gen: Optional[ModelsGenerativeEnum] = None,
    model_class: Optional[ModelsEnum] = None,
):
    """
    Evalúa cualquier archivo predictions_<model>.csv (2, 3 o 4 clases).
    Genera results_<model>.csv con classification_report completo.
    """
    try:
        # Obtenemos el nombre del modelo y el dataset
        if model_class is None and model_gen is None:
            raise HTTPException(
                400, "Debe especificar al menos un modelo (model_class o model_gen)"
            )
        model_name = (
            model_class.value.split("/")[-1]
            if model_class
            else model_gen.value.split("/")[-1]
        )
        dataset_folder = "edos_labelled_sexism_grade_reduced"
        carpeta = f"app/results/{dataset_folder}"

        results = {}

        for phrase in phrases:
            results[phrase] = {}
            for classification_prompt in ["binary", "3labels", "4labels"]:
                label_col = "label_sexist"
                if classification_prompt == "3labels":
                    label_col = "label_sexist_3"
                elif classification_prompt == "4labels":
                    label_col = "label_sexist_4"

                # Obtener ruta de los resultados
                pred_csv = (
                    f"{carpeta}/predictions_{classification_prompt}_{model_name}.csv"
                )

                if not os.path.exists(pred_csv):
                    raise HTTPException(404, f"{pred_csv} no encontrado")

                df = pd.read_csv(pred_csv)

                # Filtrar por la frase
                df_phrase = df[df["text"] == phrase]

                if df_phrase.empty:
                    results[phrase][classification_prompt] = {
                        "error": "Frase no encontrada en las predicciones."
                    }

                if classification_prompt == "binary":
                    map_type = {
                        1: "sexist",
                        0: "not sexist",
                    }
                elif classification_prompt == "3labels":
                    map_type = {
                        2: "sexist",
                        0: "not sexist",
                        1: "unsure",
                    }
                elif classification_prompt == "4labels":
                    map_type = {
                        2: "sexist (high confidence)",
                        1: "sexist (low confidence)",
                        0: "not sexist",
                        3: "sexist",
                    }

                results[phrase][classification_prompt] = {
                    "preds": (
                        map_type[int(df_phrase["preds"].values[0])]
                        if not df_phrase.empty
                        else None
                    ),
                    "label": (
                        df_phrase[label_col].values[0] if not df_phrase.empty else None
                    ),
                }

        return results

    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")
