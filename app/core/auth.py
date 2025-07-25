from datetime import timedelta
from typing import List
from typing import Optional

import jwt
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Security

from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES
from app.core.config import ALGORITHM_JWT
from app.core.config import oauth2_scheme
from app.core.config import pwd_context
from app.core.config import SECRET_KEY_JWT
from app.database.OAuth import UsersAuth
from app.schemas.auth import UserCredentials
from app.schemas.auth import UserRoles
from app.utils.dates import get_current_date


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Función para crear un token JWT"""
    current_date = get_current_date()
    if expires_delta:
        expire = current_date + expires_delta
    else:
        expire = current_date + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY_JWT, algorithm=ALGORITHM_JWT)
    return encoded_jwt


def verify_token(token: str):
    """Decodificar el JWT, obtener los roles y verificar su autenticidad"""
    try:
        payload = jwt.decode(token, SECRET_KEY_JWT, algorithms=[ALGORITHM_JWT])
        user = UserRoles(
            id_user=payload["id_user"],
            username=payload["sub"],
            roles=payload["roles"],
            is_active=payload["is_active"],
        )
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(token: str = Security(oauth2_scheme)):
    """Obtener el usuario desde el token"""
    return verify_token(token)


async def get_user_from_db(username: str):
    """Obtener el usuario desde base de datos"""
    user_service = UsersAuth()
    try:
        user = await user_service.get_user_roles(username=username)
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting the user: {str(e)}")


def verify_password(plain_password, hashed_password):
    """Verifica la contraseña aplicándole el hash y viendo si son la misma"""
    return pwd_context.verify(plain_password, hashed_password)


async def authenticate_user(credentials: UserCredentials):
    """Autentica al usuario usando su username y contraseña"""
    user = await get_user_from_db(credentials.username)
    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect username")
    if not user.is_active:
        raise HTTPException(status_code=401, detail="This user is not active")
    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return user


def role_required(
    required_roles: List[str],
    user: UserRoles = Depends(get_current_user),
):
    """Verificar si el usuario tiene uno de los roles requeridos"""
    if not any(role in user.roles for role in required_roles):
        raise HTTPException(
            status_code=403,
            detail="You don't have the necessary role to access this route",
        )
