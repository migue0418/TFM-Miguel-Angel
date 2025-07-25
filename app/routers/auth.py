from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.core.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    role_required,
)
from app.database.OAuth import UsersAuth
from app.schemas.auth import UserCredentials, UserRoles
from app.schemas.users_roles import (
    ChangePassword,
    CreateRolForm,
    CreateUserForm,
    EditRolForm,
    EditUserForm,
)
from app.utils.users_roles_management import (
    add_rol,
    create_user_with_roles,
    edit_user_with_roles,
    get_users_roles,
    update_rol,
)
from app.utils.users_roles_management import get_roles as _get_roles

router = APIRouter(
    prefix="/auth",
    tags=["Auth Functions"],
)


@router.post("/token")
async def login(user_credentials: OAuth2PasswordRequestForm = Depends()):
    # Aquí puedes validar al usuario con tu base de datos
    user = await authenticate_user(user_credentials)
    if user:
        # Si el usuario es válido, se genera un token
        data = {
            "id_user": user.id_user,
            "sub": user.username,
            "roles": user.roles,
            "is_active": user.is_active,
        }
        access_token = create_access_token(data=data)
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise ValueError("Usuario y contraseña incorrectos")


@router.get("/me")
async def get_my_data(user_payload: UserRoles = Depends(get_current_user)):
    try:
        # Obtenemos el usuario de BBDD
        return await get_users_roles(id_user=user_payload.id_user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users")
async def get_users(
    id_user: Optional[int] = None,
    user_payload: UserRoles = Depends(get_current_user),
):
    """
    Obtener un listado de usuarios o un usuario específico junto con sus roles
    """
    if id_user is None:
        role_required(["admin"], user_payload)
    try:
        return await get_users_roles(id_user=id_user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/users/{id_user}")
async def get_user(
    id_user: int,
    user_payload: UserRoles = Depends(get_current_user),
):
    """
    Obtener un usuario dado su ID
    """
    role_required(["admin"], user_payload)
    user_service = UsersAuth()
    try:
        user = await user_service.get_user(id_user=id_user)
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting the user: {str(e)}")


@router.put("/users/edit")
async def edit_user(
    form_data: EditUserForm,
    # user_payload: UserRoles = Depends(get_current_user),
):
    """
    Endpoint para editar un usuario.
    """
    # role_required(["admin"], user_payload)
    try:
        result = await edit_user_with_roles(user_data=form_data)
        if result:
            return "Usuario actualizado con éxito"
        else:
            raise HTTPException(status_code=400, detail="Ha ocurrido un error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@router.post("/users/create")
async def create_user(
    form_data: CreateUserForm,
    # user_payload: UserRoles = Depends(get_current_user),
):
    """
    Endpoint para crear un usuario.
    """
    # role_required(["admin"], user_payload)
    try:
        # Ejecutamos las transacciones necesarias para crear el usuario
        result = await create_user_with_roles(user_data=form_data)
        if result:
            return "Usuario creado con éxito"
        else:
            raise HTTPException(status_code=400, detail="Ha ocurrido un error")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.delete("/users/{id_user}")
async def delete_user(
    id_user: int,
    user_payload: UserRoles = Depends(get_current_user),
):
    """
    Borra un usuario dado su ID
    """
    role_required(["admin"], user_payload)
    user_service = UsersAuth()
    try:
        result, _ = await user_service.delete_user(id_user=id_user)
        if result:
            return result
        else:
            raise HTTPException(status_code=400, detail="Error al eliminar el usuario")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting the user: {str(e)}")


@router.post("/users/change-password")
async def change_password(
    credentials: ChangePassword,
    user_payload: UserRoles = Depends(get_current_user),
):
    """
    Cambiar la contraseña de un usuario dada su contraseña actual.
    """
    role_required(["admin", "user"], user_payload)
    user_service = UsersAuth()
    try:
        # Comprobamos si las credenciales son válidas
        user = await authenticate_user(
            credentials=UserCredentials(**credentials.model_dump()),
        )
        if user:
            await user_service.edit_user(
                id_user=user.id_user,
                password=credentials.new_password,
            )
            return "Contraseña modificada con éxito"
        else:
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating user: {str(e)}")


@router.get("/roles")
async def get_roles(
    id_rol: Optional[int] = None,
    user_payload: UserRoles = Depends(get_current_user),
):
    """
    Obtener un listado de usuarios o un usuario específico junto con sus roles
    """
    role_required(["admin"], user_payload)
    try:
        return await _get_roles(id_rol=id_rol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/roles/create")
async def create_rol(
    form_data: CreateRolForm,
    # user_payload: UserRoles = Depends(get_current_user),
):
    """
    Crea un nuevo rol
    """
    # role_required(["admin"], user_payload)
    try:
        result = await add_rol(form_data=form_data)
        if result:
            return "Rol creado con éxito"
        else:
            raise HTTPException(status_code=400, detail="No se pudo crear el rol")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/roles/edit")
async def edit_rol(
    form_data: EditRolForm,
    user_payload: UserRoles = Depends(get_current_user),
):
    """
    Edita un rol
    """
    role_required(["admin"], user_payload)
    try:
        result = await update_rol(form_data=form_data)
        if result:
            return "Rol actualizado con éxito"
        else:
            raise HTTPException(status_code=400, detail="No se pudo actualizar el rol")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
