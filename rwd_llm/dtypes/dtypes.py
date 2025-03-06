from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


# global pydantic config
class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(use_enum_values=True)


T = TypeVar("T", bound="BaseObject")


class BaseObject(BaseModel):
    id: str
    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        raw_dict = super().model_dump(**kwargs)
        # put metadata keys in the root of the dict
        metadata = raw_dict.pop("metadata", None) or {}
        for key, value in metadata.items():
            if key in raw_dict:
                raise ValueError(f"Found key {key} in metadata and in object.")
            raw_dict[key] = value
        return raw_dict

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        metadata = {}
        for field in list(data):  # iterate over a copy of the keys
            if field not in cls.model_fields:
                field_val = data.pop(field)
                metadata[field] = field_val
        data["metadata"] = metadata
        return cls(**data)


class Patient(BaseObject):
    type: str = "Patient"


class Document(BaseObject):
    text: str


class ClinicalNote(Document):
    patient_id: str
    date: str


class Annotation(BaseObject):
    document_id: str
    begin: int
    end: int


class Label(BaseObject):
    label: Any = None
