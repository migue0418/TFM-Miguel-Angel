from fastapi import FastAPI
from app.core.config import settings
from app.routers import gtp_text, reddit, bias_detection, web_crawling, sexism_detection
from app.routers.RedditBias import data_preparation, evaluation

# Registrar modelos en la base de datos
from app.database.db_sqlalchemy import Base, engine

Base.metadata.create_all(bind=engine)

# Crear la aplicación FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for the Master's Thesis",
    version=settings.VERSION,
    docs_url="/docs",
    contact={
        "name": "Miguel Ángel Benítez Alguacil",
        "url": "https://github.com/migue0418",
    },
)

# Incluir los routers
app.include_router(gtp_text.router)
app.include_router(reddit.router)
app.include_router(data_preparation.router)
app.include_router(evaluation.router)
app.include_router(sexism_detection.router)
app.include_router(bias_detection.router)
app.include_router(web_crawling.router)


@app.get("/")
def read_root():
    return {"project_name": settings.PROJECT_NAME, "version": settings.VERSION}


@app.on_event("startup")
async def startup_event():
    print("Access documentation: http://127.0.0.1:8000/docs")
