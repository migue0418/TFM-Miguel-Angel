import os

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(
    prefix="/app",
    tags=["Frontend Pages"],
)


@router.get("/favicon.ico")
def favicon():
    file_path = os.path.join("frontend", "build", "favicon.ico")
    return FileResponse(file_path)


@router.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_index(full_path: str):
    index_path = os.path.join("frontend", "build", "index.html")
    with open(index_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())
