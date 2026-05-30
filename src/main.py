from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router as api_router
from src.config.settings import get_app_config, resolve_path

STATIC_DIR = resolve_path("static")
UPLOAD_DIR = resolve_path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    yield


app = FastAPI(
    title="Mason-RAG",
    description="企业知识库 RAG 系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "Mason-RAG",
        "version": "0.1.0",
        "message": "企业知识库 RAG 系统已启动",
    }


@app.get("/chat")
async def chat_page():
    return FileResponse(STATIC_DIR / "chat.html")


if __name__ == "__main__":
    import uvicorn

    cfg = get_app_config()
    uvicorn.run(
        "src.main:app",
        host=cfg["host"],
        port=cfg["port"],
        reload=cfg["debug"],
    )