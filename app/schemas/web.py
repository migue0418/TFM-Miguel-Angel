from pydantic import BaseModel
from typing import Optional


class Domain(BaseModel):
    id_domain: int
    domain_url: str
    absolute_url: str


class NewDomain(BaseModel):
    domain_url: str
    absolute_url: str


class EditDomain(NewDomain):
    id_domain: int


class URL(BaseModel):
    id_url: int
    id_domain: int
    absolute_url: str
    relative_url: str
    html_content: Optional[str] = None


class NewURL(BaseModel):
    id_domain: int
    absolute_url: str
    relative_url: str
    html_content: Optional[str] = None


class EditURL(NewURL):
    id_url: int
