import json
from typing import Optional

from app.core.dependencies import db_tfm
from app.database.OAuth import UsersAuth
from app.schemas.users_roles import (
    CreateRolForm,
    CreateUserForm,
    EditRolForm,
    EditUserForm,
)
from app.utils.dates import get_current_date


async def get_users_roles(id_user: Optional[int] = None):
    query_where = " WHERE u.id_user = ?" if id_user else ""
    params = (id_user,) if id_user else None
    # SQLite
    query = f"""SELECT
                u.id_user,
                u.username,
                u.nombre,
                u.email,
                u.is_active,
                COALESCE(
                    (
                    SELECT
                        json_group_array(
                        json_object(
                            'id_rol', r.id_rol,
                            'nombre',  r.nombre
                        )
                        )
                    FROM users_roles AS ur
                    JOIN roles AS r
                        ON r.id_rol = ur.id_rol
                    WHERE
                        ur.id_user   = u.id_user
                        AND ur.is_active = 1
                    ),
                    '[]'
                ) AS roles_json
                FROM users AS u
                {query_where};"""

    result = await db_tfm.fetch_all(query=query, params=params)
    for res in result:
        res["roles"] = json.loads(res.pop("roles_json", "[]"))

    return result[0] if id_user else result


async def create_user_with_roles(user_data: CreateUserForm):
    # Fecha actual
    current_date = get_current_date()

    # Contraseña hash
    oauth_service = UsersAuth()
    hashed_password = await oauth_service.hash_password(user_data.password)

    # Preparamos los datos del usuario
    data_user = {
        "nombre": user_data.nombre,
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "is_active": 1,
        "created_at": current_date,
        "modified_at": current_date,
    }
    transactions = [
        {
            "operation_type": "insert",
            "table": "users",
            "operation_data": data_user,
            "return_id": True,
        },
    ]

    # Por cada rol que tenga el usuario, añadimos una operación
    for id_rol in user_data.roles:
        # Preparamos las operaciones
        data_user_roles = {
            "id_user": "<ID>users",
            "id_rol": id_rol,
            "is_active": 1,
            "created_at": current_date,
            "modified_at": current_date,
        }

        transactions.append(
            {
                "operation_type": "insert",
                "table": "users_roles",
                "operation_data": data_user_roles,
            },
        )

    # Ejecutamos las transacciones
    result, _ = await db_tfm.execute_transactions(transactions=transactions)

    return result


async def edit_user_with_roles(user_data: EditUserForm):
    # Fecha actual
    current_date = get_current_date()

    # Preparamos los datos del usuario
    data_user_dict = {
        "nombre": user_data.nombre,
        "username": user_data.username,
        "email": user_data.email,
        "is_active": 1,
        "modified_at": current_date,
    }
    data_user_id = {
        "id_user": user_data.id_user,
    }

    if user_data.password:
        # Contraseña hash
        oauth_service = UsersAuth()
        data_user_dict["hashed_password"] = await oauth_service.hash_password(
            user_data.password,
        )

    transactions = [
        {
            "operation_type": "update",
            "table": "users",
            "operation_data": data_user_dict,
            "where_data": data_user_id,
        },
        # Borramos los roles que tuviera
        {
            "operation_type": "delete",
            "table": "users_roles",
            "where_data": data_user_id,
        },
    ]

    # Por cada rol que tenga el usuario, añadimos una operación
    for id_rol in user_data.roles:
        # Preparamos las operaciones
        data_user_roles = {
            "id_user": user_data.id_user,
            "id_rol": id_rol,
            "is_active": 1,
            "created_at": current_date,
            "modified_at": current_date,
        }

        transactions.append(
            {
                "operation_type": "insert",
                "table": "users_roles",
                "operation_data": data_user_roles,
            },
        )

    # Ejecutamos las transacciones
    result, _ = await db_tfm.execute_transactions(transactions=transactions)

    return result


async def get_roles(id_rol: Optional[int] = None):
    query_where = " WHERE id_rol = ?" if id_rol else ""
    params = (id_rol,) if id_rol else None
    query = f"SELECT * FROM roles {query_where};"

    result = await db_tfm.fetch_all(query=query, params=params)

    return result[0] if id_rol and result else result


async def add_rol(form_data: CreateRolForm):
    data_dict = {
        "nombre": form_data.nombre,
        "descripcion": form_data.descripcion,
        "created_at": get_current_date(),
    }

    result = await db_tfm.insert(table="roles", data=data_dict)

    return result


async def update_rol(form_data: EditRolForm):
    data_dict = {
        "nombre": form_data.nombre,
        "descripcion": form_data.descripcion,
    }

    data_id = {
        "id_rol": form_data.id_rol,
    }

    result = await db_tfm.update(
        table="roles",
        set_fields=data_dict,
        where_conditions=data_id,
    )

    return result
