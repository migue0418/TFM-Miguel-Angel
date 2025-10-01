import pandas as pd
from pathlib import Path
from typing import Literal, List

from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum, ModelsGenerativeEnum

# Parámetros fijos: modelos y columnas que queremos mostrar
MODELS = [model.value.split("/")[-1] for model in ModelsEnum]
SHOW_COLS = [
    ("learning_rate", "LR"),
    ("weight_decay", "WD"),
    ("batch_size", "Batch"),
    ("eval_f1", "F1"),
    ("eval_accuracy", "Acc."),
    ("eval_precision", "Prec."),
    ("eval_recall", "Rec."),
]
GENERATIVE_MODELS = [model.value.split("/")[-1] for model in ModelsGenerativeEnum]
GENERATIVE_SHOW_COLS = [
    ("f1", "F1"),
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]


def txt_to_int(mapper, serie: pd.Series) -> pd.Series:
    serie_norm = serie.astype(str).str.strip().str.lower()
    out = serie_norm.map(mapper)

    # detecta etiquetas sin correspondencia
    if out.isna().any():
        missing = serie_norm[out.isna()].unique()
        raise ValueError(f"Etiquetas sin mapear: {missing}")

    return out.astype(int)


def _best_run(csv_path: Path) -> pd.Series:
    """Devuelve la mejor fila según F1 y Accuracy."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise FileNotFoundError(f"{csv_path} vacío o inexistente.")
    return df.sort_values(["eval_f1", "eval_accuracy"], ascending=False).iloc[0]


def _generative_results(csv_path: Path) -> pd.Series:
    """Devuelve la fila de resultados de generativa."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise FileNotFoundError(f"{csv_path} vacío o inexistente.")
    return df.iloc[0]


def get_latex_table(
    dataset: str, dataset_model: DatasetEnum, caption: str, full_results: bool = False
) -> str:
    """
    Devuelve una cadena con la tabla LaTeX para el dataset indicado,
    comparando los mejores runs de BERT y ModernBERT.
    """

    rows: List[str] = []
    if full_results:
        # encabezados LaTeX
        header_cols = (
            "\\textbf{Modelo} & "
            + " & ".join("\\textbf{" + h + "}" for _, h in GENERATIVE_SHOW_COLS)
            + r" \\"
        )
        tabular = rf"\begin{{tabular}}{{lccccc}}"
        label = rf"\label{{tab:{dataset}_results}}"
        for model in MODELS:
            folder = (
                "3_labels_classification"
                if "3_SEXISM" in dataset_model.name
                else "4_labels_classification"
            )
            csv_path = Path("app/results") / folder / dataset / model / "resultados.csv"
            best = _generative_results(csv_path)

            # valores a mostrar
            values = [model] + [best[col] for col, _ in GENERATIVE_SHOW_COLS]
            # formateo
            fmt_values = [
                "\\emph{" + model + "}",  # nombre con cursiva
                f"{values[1]:.3f}",  # F1
                f"{values[2]:.3f}",  # Acc.
                f"{values[3]:.3f}",  # Prec.
                f"{values[4]:.3f}",  # Rec.
            ]
            rows.append(" & ".join(fmt_values) + r" \\")

        # Añadir modelos generativos
        # for model in GENERATIVE_MODELS:
        #     csv_path = Path("app/results") / dataset / model / f"results_{model}.csv"
        #     results = _generative_results(csv_path)

        #     # valores a mostrar
        #     values = [model] + [results[col] for col, _ in GENERATIVE_SHOW_COLS]
        #     # formateo
        #     fmt_values = [
        #         "\\emph{" + model + "}",  # nombre con cursiva
        #         f"{values[1]:.3f}",  # F1
        #         f"{values[2]:.3f}",  # Acc.
        #         f"{values[3]:.3f}",  # Prec.
        #         f"{values[4]:.3f}",  # Rec.
        #     ]
        #     rows.append(" & ".join(fmt_values) + r"\\")
    else:
        # encabezados LaTeX
        header_cols = (
            "\\textbf{Modelo} & "
            + " & ".join("\\textbf{" + h + "}" for _, h in SHOW_COLS)
            + r" \\"
        )
        tabular = rf"\begin{{tabular}}{{lccccccc}}"
        label = rf"\label{{tab:{dataset}_best}}"
        for model in MODELS:
            folder = (
                "3_labels_classification"
                if "3_SEXISM" in dataset_model.name
                else "4_labels_classification"
            )
            csv_path = Path("app/results") / folder / dataset / model / "resultados.csv"
            best = _best_run(csv_path)

            # valores a mostrar
            values = [model] + [best[col] for col, _ in SHOW_COLS]
            # formateo
            fmt_values = [
                "\\emph{" + model + "}",  # nombre con cursiva
                f"{values[1]:.0e}",  # LR   → 5e-05
                f"{values[2]:.0e}" if values[2] else "0",  # WD   → 1e-03 …
                f"{int(values[3])}",  # Batch
                f"{values[4]:.3f}",  # F1
                f"{values[5]:.3f}",  # Acc.
                f"{values[6]:.3f}",  # Prec.
                f"{values[7]:.3f}",  # Rec.
            ]
            rows.append(" & ".join(fmt_values) + r" \\")

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
""".strip()

    return table
