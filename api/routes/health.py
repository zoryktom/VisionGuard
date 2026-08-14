"""Health and metrics routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.deps import get_pipeline, pipeline_info
from api.schemas import HealthResponse, StatsResponse
from visionguard.__about__ import __version__

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + backend identity."""

    info = pipeline_info()
    return HealthResponse(version=__version__, device=info["device"], backend=info["backend"])


@router.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness: pipeline constructed successfully."""

    get_pipeline()
    return health()


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    """Rolling FPS / latency / risk."""

    info = pipeline_info()
    snap = get_pipeline().telemetry.snapshot()
    return StatsResponse(**snap, device=info["device"], backend=info["backend"])


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus exposition format."""

    return get_pipeline().telemetry.prometheus()
