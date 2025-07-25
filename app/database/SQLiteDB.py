from typing import Any, Dict, List, Optional, Tuple

import aiosqlite


class SQLiteDB:
    """
    Asynchronous SQLite wrapper using aiosqlite.

    Methods:
      - connect(): open the database connection
      - close(): close the database connection
      - insert(table, data, return_id=True): insert a row, return lastrowid or affected rows
      - update(table, set_fields, where_conditions): update rows, return affected rows
      - fetch_one(table, where=None, output_fields=None): fetch single row as dict
      - fetch_all(table, where=None, output_fields=None, limit=None): fetch multiple rows as list of dicts
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open a connection to the SQLite database."""
        self.conn = await aiosqlite.connect(self.db_path)
        # return rows as dictionaries
        self.conn.row_factory = aiosqlite.Row

    async def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            await self.conn.close()
            self.conn = None

    async def insert(
        self,
        table: str,
        data: Dict[str, Any],
        return_id: bool = True,
    ) -> Optional[int]:
        """
        Insert a row into `table` using `data` dict.
        If return_id is True, returns lastrowid, else returns affected row count.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")
        cols = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        async with self.conn.execute(sql, tuple(data.values())) as cursor:
            if return_id:
                await self.conn.commit()
                return cursor.lastrowid
            else:
                count = cursor.rowcount
                await self.conn.commit()
                return count

    async def update(
        self,
        table: str,
        set_fields: Dict[str, Any],
        where_conditions: Dict[str, Any],
    ) -> int:
        """
        Update `table` setting columns in `set_fields` where rows match `where_conditions`.
        Returns number of rows affected.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")
        set_clause = ", ".join([f"{k} = ?" for k in set_fields])
        where_clause = " AND ".join([f"{k} = ?" for k in where_conditions])
        sql = f"UPDATE {table} SET {set_clause}" + (
            f" WHERE {where_clause}" if where_conditions else ""
        )
        params = tuple(set_fields.values()) + tuple(where_conditions.values())
        async with self.conn.execute(sql, params) as cursor:
            count = cursor.rowcount
            await self.conn.commit()
            return count

    async def fetch_one(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single row from `table`.
        Optionally filter by `where` dict and select specific `output_fields`.
        Returns a dict or None if no row found.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def fetch_all(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows from `table`.
        Returns list of dicts (empty if no rows).
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete(
        self,
        table: str,
        where_conditions: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query with params.
        Returns number of rows affected.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        where_clause = " AND ".join([f"{k} = ?" for k in where_conditions])
        query = f"DELETE FROM {table} WHERE {where_clause}"
        params = tuple(where_conditions.values())

        async with self.conn.execute(query, params) as cursor:
            count = cursor.rowcount
            await self.conn.commit()
            return count

    async def execute(
        self,
        query: str,
        params: Optional[tuple] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query with params.
        Returns number of rows affected.
        """
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")

        async with self.conn.execute(query, params) as cursor:
            count = cursor.rowcount
            await self.conn.commit()
            return count

    async def execute_transactions(
        self,
        transactions: List[Dict[str, Any]],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Ejecuta en bloque varias operaciones (insert/update/delete).
        Si todas van bien, hace COMMIT y devuelve (True, ids_returned).
        Si falla alguna, hace ROLLBACK y devuelve (False, {}).

        Cada transacción debe tener:
          - operation_type: "insert"|"update"|"delete"
          - table: nombre de la tabla
          - operation_data: dict de columnas→valores (para insert/update)
          - where_data: dict de columnas→valores (para update/delete)
          - return_id: bool (solo para insert)
          - return_prefix: str opcional, prefijo para la clave en ids_returned
        """
        if self.conn is None:
            raise RuntimeError("La conexión no está abierta. Llama a connect().")

        ids_returned: Dict[str, Any] = {}
        try:
            await self.conn.execute("BEGIN")
            for tx in transactions:
                op = tx["operation_type"].lower()
                tbl = tx["table"]
                data = tx.get("operation_data", {})
                where = tx.get("where_data", {})
                want_id = tx.get("return_id", False)
                prefix = tx.get("return_prefix", "")

                # Comprobamos si no hay referencia a un <ID> para cambiarlo
                for key, value in data.items():
                    if value and isinstance(value, str) and "<ID>" in value:
                        # Los valores son "<ID>Tabla", quitamos la parte de <ID> para coger la clave necesaria
                        table_value = value.replace("<ID>", "")
                        # Reemplazamos con la clave obtenida previamente
                        data[key] = ids_returned.get(table_value)

                # Lo mismo en la parte del where data
                for key, value in where.items():
                    if value and isinstance(value, str) and "<ID>" in value:
                        table_value = value.replace("<ID>", "")
                        where[key] = ids_returned.get(table_value)

                if op == "insert":
                    cols = ", ".join(data.keys())
                    phs = ", ".join("?" for _ in data)
                    sql = f"INSERT INTO {tbl} ({cols}) VALUES ({phs})"
                    cursor = await self.conn.execute(sql, tuple(data.values()))
                    if want_id:
                        ids_returned[prefix + tbl] = cursor.lastrowid

                elif op == "update":
                    set_c = ", ".join(f"{c}=?" for c in data)
                    where_c = " AND ".join(f"{c}=?" for c in where)
                    sql = f"UPDATE {tbl} SET {set_c} WHERE {where_c}"
                    params = tuple(data.values()) + tuple(where.values())
                    await self.conn.execute(sql, params)

                elif op == "delete":
                    where_c = " AND ".join(f"{c}=?" for c in where)
                    sql = f"DELETE FROM {tbl} WHERE {where_c}"
                    await self.conn.execute(sql, tuple(where.values()))

                else:
                    raise ValueError(f"Operación no soportada: {op}")

            await self.conn.commit()
            return True, ids_returned

        except Exception:
            await self.conn.rollback()
            return False, {}
