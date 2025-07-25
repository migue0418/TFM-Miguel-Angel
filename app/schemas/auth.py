from typing import List, Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserCredentials(BaseModel):
    username: str
    password: str


class User(BaseModel):
    id_user: Optional[int] = None
    username: str
    is_active: Optional[bool] = None


class UserInDB(User):
    hashed_password: Optional[str] = None


class UserRoles(UserInDB):
    roles: List[str]
