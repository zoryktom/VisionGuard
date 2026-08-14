"""VisionGuard FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import events, health, infer, stream
from visionguard.__about__ import __app_name__, __version__
from visionguard.config import load_config, repo_root
from visionguard.logging_utils import setup_logging

setup_logging(load_config().system.log_level)
UI_DIR = repo_root() / "ui"


def create_app() -> FastAPI:
    """Application factory."""

    config = load_config()
    application = FastAPI(
        title=__app_name__,
        version=__version__,
        description="Real-time AI system for detecting safety hazards in video streams.",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health.router, prefix="/api/v1", tags=["ops"])
    application.include_router(infer.router, prefix="/api/v1", tags=["inference"])
    application.include_router(stream.router, prefix="/api/v1", tags=["stream"])
    application.include_router(events.router, prefix="/api/v1", tags=["events"])

    if UI_DIR.is_dir():
        application.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

        @application.get("/", include_in_schema=False)
        def dashboard() -> FileResponse:
            return FileResponse(UI_DIR / "index.html")

    return application


app = create_app()
