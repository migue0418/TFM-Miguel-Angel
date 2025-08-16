from typing import Optional
from pydantic import BaseModel


class TextInput(BaseModel):
    text: str


class URLCheckerInput(BaseModel):
    url: str
    filter_tag: Optional[str] = None


class DomainAnalyzerInput(BaseModel):
    domain: str
    filter_tag: Optional[str] = None
