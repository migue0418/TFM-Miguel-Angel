from fastapi import FastAPI
from app.core.config import settings
from app.routers import gtp_text, reddit

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API para pruebas del TFM",
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


@app.get("/")
def read_root():
    return {"project_name": settings.PROJECT_NAME, "version": settings.VERSION}


@app.on_event("startup")
async def startup_event():
    print("API GPT está en funcionamiento.")
    print("Accede a la documentación en: http://127.0.0.1:8000/docs")
