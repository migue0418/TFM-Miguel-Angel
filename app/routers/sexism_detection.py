from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from app.utils.sexism_classification import generate_4_sexism_labels_dataset


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
