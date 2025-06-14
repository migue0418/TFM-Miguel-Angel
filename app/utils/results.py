import pandas as pd
from pathlib import Path
from typing import Literal, List

# ------------------------------------------------------------
# Parámetros fijos: modelos y columnas que queremos mostrar
# ------------------------------------------------------------
MODELS = ["bert-base-uncased", "ModernBERT-base"]
SHOW_COLS = [
    ("learning_rate", "LR"),
    ("weight_decay", "WD"),
    ("batch_size", "Batch"),
    ("eval_f1", "F1"),
    ("eval_accuracy", "Acc."),
    ("eval_precision", "Prec."),
    ("eval_recall", "Rec."),
]


def _best_run(csv_path: Path) -> pd.Series:
    """Devuelve la mejor fila según F1 y Accuracy."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise FileNotFoundError(f"{csv_path} vacío o inexistente.")
    return df.sort_values(["eval_f1", "eval_accuracy"], ascending=False).iloc[0]


def get_latex_table(dataset: Literal["reduced_edos", "reduced_edos_10k"]) -> str:
    """
    Devuelve una cadena con la tabla LaTeX para el dataset indicado,
    comparando los mejores runs de BERT y ModernBERT.
    """

    rows: List[str] = []
    for model in MODELS:
        csv_path = Path("app/results") / dataset / model / "resultados.csv"
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

    # encabezados LaTeX
    header_cols = (
        "\\textbf{Modelo} & "
        + " & ".join("\\textbf{" + h + "}" for _, h in SHOW_COLS)
        + r" \\"
    )
    table = rf"""
\begin{{table}}[ht]
\centering
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{lccccccc}}
\toprule
{header_cols}
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Resultados de clasificación binaria en EDOS}}
\label{{tab:{dataset}_best}}
\end{{table}}
""".strip()

    return table
