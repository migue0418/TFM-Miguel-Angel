from typing import Literal
from fastapi import APIRouter, HTTPException

from app.utils.results import _best_run, get_latex_table

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
    dataset: Literal["reduced_edos", "reduced_edos_10k"],
):
    try:
        # Import the results CSV file based on the dataset and model
        latex_table = get_latex_table(dataset)

        return latex_table
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
