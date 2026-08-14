"""Pydantic request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from visionguard.__about__ import __version__


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str = "ok"
    version: str = __version__
    device: str
    backend: str


class DetectionOut(BaseModel):
    """Serialized detection."""

    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    category: str
    severity: str


class EventOut(BaseModel):
    """Serialized hazard event."""

    event_id: str
    name: str
    category: str
    severity: str
    confidence: float
    track_ids: list[int]
    bbox: tuple[float, float, float, float]
    frame_index: int
    timestamp: float
    zone: str | None = None
    message: str = ""


class InferResponse(BaseModel):
    """Single-image inference result."""

    detections: list[DetectionOut]
    events: list[EventOut]
    risk_score: float
    inference_ms: float
    fps: float
    frame_index: int


class StatsResponse(BaseModel):
    """Rolling telemetry."""

    frames: int
    fps: float
    inference_ms: float
    risk: float
    events: int
    device: str = Field(default="")
    backend: str = Field(default="")
