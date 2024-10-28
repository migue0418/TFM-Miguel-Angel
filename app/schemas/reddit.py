from pydantic import BaseModel
from typing import List, Optional


class Comment(BaseModel):
    comment_id: str
    author: Optional[str]
    comment: str
    created: float


class Submission(BaseModel):
    id: str
    title: str
    score: int
    url: str
    created: float
    comments: List[Comment]
