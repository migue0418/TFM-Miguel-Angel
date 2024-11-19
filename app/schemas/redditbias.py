from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Optional, ClassVar


class Topic(BaseModel):
    name: Literal["gender", "race", "orientation", "religion1", "religion2"]
    minority_group: Optional[Literal["female", "black", "lgbtq", "jews", "muslims"]] = (
        None
    )
    majority_group: Optional[Literal["male", "white", "straight", "christians"]] = None

    # ClassVar with the type and minority/majority groups of each one
    valid_groups: ClassVar[dict] = {
        "gender": {"minority": "female", "majority": "male"},
        "race": {"minority": "black", "majority": "white"},
        "orientation": {"minority": "lgbtq", "majority": "straight"},
        "religion1": {"minority": "jews", "majority": "christians"},
        "religion2": {"minority": "muslims", "majority": "christians"},
    }

    @field_validator("minority_group", mode="before")
    def set_minority_group(cls, v, values):
        if v is None:
            name = values.name
            return cls.valid_groups.get(name, {}).get("minority")
        return v

    @field_validator("majority_group", mode="before")
    def set_majority_group(cls, v, values):
        if v is None:
            name = values.name
            return cls.valid_groups.get(name, {}).get("majority")
        return v

    @model_validator(mode="after")
    def ensure_defaults(cls, values):
        # Set the default value if None
        values.minority_group = values.minority_group or cls.set_minority_group(
            None, values
        )
        values.majority_group = values.majority_group or cls.set_majority_group(
            None, values
        )

        # Validate the data and verify consistency
        name = values.name
        minority_group = values.minority_group
        majority_group = values.majority_group
        valid_minority = cls.valid_groups.get(name, {}).get("minority")
        valid_majority = cls.valid_groups.get(name, {}).get("majority")

        if minority_group != valid_minority:
            raise ValueError(
                f"Invalid minority group '{minority_group}' for topic '{name}'"
            )
        if majority_group != valid_majority:
            raise ValueError(
                f"Invalid majority group '{majority_group}' for topic '{name}'"
            )

        return values
