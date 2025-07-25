from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.dependencies import db_tfm
from app.routers import (
    frontend,
    auth,
    gtp_text,
    reddit,
    bias_detection,
    web_crawling,
    sexism_detection,
    results,
)
from app.routers.RedditBias import data_preparation, evaluation

# Registrar modelos en la base de datos
from app.database.db_sqlalchemy import Base, engine

Base.metadata.create_all(bind=engine)


# Manejador de ciclo de vida de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando la aplicación...")
    print("Access documentation: http://127.0.0.1:8000/docs")

    # Inicializamos la conexión de la bbdd
    await db_tfm.connect()

    yield

    print("Cerrando la aplicacion...")
    await db_tfm.close()


# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for the Master's Thesis",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    contact={
        "name": "Miguel Ángel Benítez Alguacil",
        "url": "https://github.com/migue0418",
    },
)

# Monta archivos estáticos del frontend
app.mount(
    "/static",
    StaticFiles(directory="frontend/build/static", html=True),
    name="app",
)

# Middleware para CORS del frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # o ['*'] para todos (no recomend. en prod)
    allow_credentials=True,  # cookies / Authorization
    allow_methods=["*"],  # GET, POST, PUT, DELETE, …
    allow_headers=["*"],  # Content-Type, Authorization, …
)

# Incluir los routers
app.include_router(frontend.router)
app.include_router(auth.router)
app.include_router(gtp_text.router)
app.include_router(reddit.router)
app.include_router(data_preparation.router)
app.include_router(evaluation.router)
app.include_router(sexism_detection.router)
app.include_router(bias_detection.router)
app.include_router(web_crawling.router)
app.include_router(results.router)


@app.get("/")
def read_root():
    return {"project_name": settings.PROJECT_NAME, "version": settings.VERSION}


@app.get("/favicon.ico")
def favicon():
    file_path = os.path.join("frontend", "build", "favicon.ico")
    return FileResponse(file_path)
