from typing import List
from typing import Optional

from pydantic import BaseModel

from app.schemas.auth import UserCredentials


class ChangePassword(UserCredentials):
    new_password: str


class CreateUserForm(UserCredentials):
    email: Optional[str]
    nombre: Optional[str]
    roles: List[int]


class EditUserForm(BaseModel):
    id_user: int
    username: str
    password: Optional[str]
    email: Optional[str]
    nombre: Optional[str]
    roles: List[int]


class CreateRolForm(BaseModel):
    nombre: str
    descripcion: str


class EditRolForm(CreateRolForm):
    id_rol: int
