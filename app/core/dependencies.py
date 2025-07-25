from pathlib import Path
from app.database.SQLiteDB import SQLiteDB

# Ruta al fichero de la BBDD
BASE_DIR = Path(__file__).resolve().parent.parent  # …/app
DB_PATH = BASE_DIR / "database" / "db_tfm.db"  # …/app/database/db_tfm.db

db_tfm = SQLiteDB(DB_PATH.as_posix())
