from typing import Any, Optional

from app.core.config import pwd_context
from app.core.dependencies import db_tfm as db
from app.schemas.auth import UserInDB, UserRoles
from app.utils.dates import get_current_date


class UsersAuth:
    def __init__(self):
        self.db = db

    async def hash_password(self, password: str) -> str:
        """Genera el hash de una contraseña."""
        return pwd_context.hash(password)

    async def get_user(
        self,
        id_user: Optional[int] = None,
        username: Optional[str] = None,
    ) -> UserInDB | dict[str, Any] | None:
        """
        Obtiene un usuario de la tabla `users`.

        :param id_user: ID del usuario.
        :param username: Nombre de usuario.
        :return: Diccionario con los datos del usuario.
        """
        param = (id_user,) if id_user else (username,)
        if param is None:
            raise ValueError("Debe introducir un username o un id_user")
        query_where = " id_user = ?" if id_user else " username = ?"
        query = f"SELECT id_user, username, hashed_password, is_active FROM users WHERE {query_where}"
        result = await self.db.fetch_one(query=query, params=param)
        if result:
            result = UserInDB(**result)
        return result

    async def get_user_roles(
        self,
        id_user: Optional[int] = None,
        username: Optional[str] = None,
    ) -> UserRoles | dict[str, Any] | None:
        """
        Obtiene un usuario de la tabla `users`.

        :param id_user: ID del usuario.
        :param username: Nombre de usuario.
        :return: Diccionario con los datos del usuario.
        """
        param = (id_user,) if id_user else (username,)
        if param is None:
            raise ValueError("Debe introducir un username o un id_user")
        query_where = " u.id_user = ?" if id_user else " u.username = ?"
        query = f"""SELECT u.id_user, u.username, u.hashed_password, u.is_active, STRING_AGG(r.nombre, ', ') AS roles
                    FROM users u
                    LEFT JOIN users_roles ur on ur.id_user = u.id_user AND ur.is_active = 1
                    LEFT JOIN roles r on r.id_rol = ur.id_rol WHERE {query_where}
                    GROUP BY u.id_user, u.username, u.hashed_password, u.is_active;"""
        result = await self.db.fetch_one(query=query, params=param)
        if result:
            # Añadimos la lista de roles
            result["roles"] = result["roles"].split(", ") if result["roles"] else []
            result = UserRoles(**result)
        return result

    async def create_user(
        self,
        username: str,
        password: str,
        is_active: int = 1,
    ) -> int | None:
        """
        Crea un nuevo usuario en la tabla `users`.

        :param username: Nombre de usuario.
        :param password: Contraseña en texto plano.
        :param is_active: Indicador de si el usuario está activo (1 = activo, 0 = inactivo).
        :return: ID del usuario creado.
        """
        hashed_password = await self.hash_password(password)
        current_date = get_current_date()
        data = {
            "username": username,
            "hashed_password": hashed_password,
            "created_at": current_date,
            "modified_at": current_date,
            "is_active": is_active,
        }
        await self.db.insert("users", data)

        # Recuperar el ID del último usuario creado
        result = await self.db.fetch_one("SELECT IDENT_CURRENT('users') AS id_user")
        return result["id_user"] if result else None

    async def edit_user(
        self,
        id_user: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        is_active: Optional[int] = None,
    ):
        """
        Edita los datos de un usuario existente en la tabla `users`.

        :param id_user: ID del usuario a editar.
        :param username: Nuevo nombre de usuario (opcional).
        :param password: Nueva contraseña en texto plano (opcional).
        :param is_active: Indicador de si el usuario está activo (1 = activo, 0 = inactivo) (opcional).
        """
        set_fields: dict[str, Any] = {}
        # Descomentar si se quiere editar el nombre de usuario
        # if username:
        #     set_fields["username"] = username
        if password:
            set_fields["hashed_password"] = await self.hash_password(password)
        if is_active is not None:
            set_fields["is_active"] = is_active
        set_fields["modified_at"] = get_current_date()

        if set_fields:
            await self.db.update("users", set_fields, {"id_user": id_user})

    async def delete_user(
        self,
        id_user: int,
    ) -> UserRoles | dict[str, Any] | None:
        """
        Obtiene un usuario de la tabla `users`.

        :param id_user: ID del usuario.
        :param username: Nombre de usuario.
        :return: Diccionario con los datos del usuario.
        """
        param = (id_user,)
        if param is None:
            raise ValueError("Debe introducir un username o un id_user")
        query = """DELETE FROM users WHERE id_user = ?"""
        result = await self.db.execute(query=query, params=param)
        return result
