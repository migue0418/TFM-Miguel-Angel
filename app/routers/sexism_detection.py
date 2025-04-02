from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from app.enums.datasets_enums import DatasetEnum
from app.enums.models_enums import ModelsEnum
from app.utils.sexism_classification import (
    generate_4_sexism_labels_dataset,
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
