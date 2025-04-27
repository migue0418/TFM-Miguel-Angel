from enum import Enum
from pathlib import Path
from app.core.config import files_path, redditbias_data_path


class DatasetEnum(str, Enum):
    EDOS = ("EDOS", files_path / "edos_labelled_aggregated.csv", "edos")
    EDOS_REDUCED = (
        "EDOS_REDUCED",
        files_path / "edos_labelled_reduced.csv",
        "reduced_edos",
    )
    EDOS_REDUCED_FULL = (
        "EDOS_REDUCED_FULL",
        files_path / "edos_labelled_reduced_10k.csv",
        "reduced_edos_full",
    )
    EDOS_REDUCED_CLEAN = (
        "EDOS_REDUCED_CLEAN",
        files_path / "edos_labelled_cleaned_reduced.csv",
        "reduced_edos_cleaned",
    )
    EDOS_COMBINED_REDDIT_BIAS = (
        "EDOS_COMBINED_REDDIT_BIAS",
        files_path / "edos_reduced_reddit_bias.csv",
        "reduced_edos_with_reddit_bias",
    )
    REDDIT_BIAS = (
        "REDDIT_BIAS",
        (
            redditbias_data_path
            / "gender"
            / "reddit_comments_gender_female_processed_phrase_annotated.csv"
        ),
        "reddit_bias",
    )
    EDOS_4_SEXISM = (
        "EDOS_4_SEXISM",
        files_path / "edos_labelled_4_sexism_grade.csv",
        "edos_4_sexism",
    )
    EDOS_4_SEXISM_REDUCED = (
        "EDOS_4_SEXISM_REDUCED",
        files_path / "edos_labelled_4_sexism_grade_reduced.csv",
        "edos_4_sexism_reduced",
    )

    def __new__(cls, enum_name: str, csv_path: Path, model_folder_path: str):
        obj = str.__new__(cls, enum_name)
        obj._value_ = enum_name
        return obj

    def __init__(self, enum_name: str, csv_path: Path, model_folder_path: str):
        self.csv_path = csv_path
        self.model_folder_path = model_folder_path
