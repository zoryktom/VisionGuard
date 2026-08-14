"""YAML + environment configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from visionguard.exceptions import ConfigError

_REPO_ROOT = Path(__file__).resolve().parents[2]


class CaptureConfig(BaseModel):
    """Video capture settings."""

    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    buffer_size: int = 2
    drop_stale_frames: bool = True


class InferenceConfig(BaseModel):
    """Detector backend settings."""

    backend: str = "ultralytics"
    weights: str = "yolov8n.pt"
    custom_weights: str = "models/checkpoints/visionguard_yolov8n.pt"
    imgsz: int = 640
    confidence: float = 0.35
    iou: float = 0.45
    max_detections: int = 100
    half: bool = True
    batch_size: int = 1


class TrackingConfig(BaseModel):
    """Multi-object tracker settings."""

    enabled: bool = True
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    ema_alpha: float = 0.4


class HazardsRuntimeConfig(BaseModel):
    """Temporal fusion thresholds."""

    persist_frames: int = 8
    cooldown_seconds: float = 4.0
    near_miss_px: float = 120.0
    risk_ema: float = 0.25


class OverlayConfig(BaseModel):
    """On-frame visualization toggles."""

    show_tracks: bool = True
    show_zones: bool = True
    show_risk: bool = True
    show_fps: bool = True
    logo: bool = True


class APIConfig(BaseModel):
    """HTTP service settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    mjpeg_quality: int = 80
    event_buffer: int = 256


class SystemConfig(BaseModel):
    """Process-level settings."""

    device: str = "auto"
    seed: int = 42
    num_workers: int = 2
    log_level: str = "INFO"


class AppConfig(BaseModel):
    """Root configuration object."""

    system: SystemConfig = Field(default_factory=SystemConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    hazards: HazardsRuntimeConfig = Field(default_factory=HazardsRuntimeConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    api: APIConfig = Field(default_factory=APIConfig)


class EnvOverrides(BaseSettings):
    """Optional environment overrides (VISIONGUARD_*)."""

    model_config = SettingsConfigDict(env_prefix="VISIONGUARD_", extra="ignore")

    device: str | None = None
    model_backend: str | None = None
    model_weights: str | None = None
    confidence: float | None = None
    iou: float | None = None
    source: str | None = None
    host: str | None = None
    port: int | None = None
    log_level: str | None = None


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""

    return _REPO_ROOT


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repo root if it is not absolute."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root() / candidate).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    file_path = resolve_path(path)
    if not file_path.exists():
        raise ConfigError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {file_path}")
    return data


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load default.yaml, optional overlay file, env vars, then explicit overrides."""

    payload = load_yaml("configs/default.yaml")
    if path is not None:
        payload = _deep_merge(payload, load_yaml(path))
    if overrides:
        payload = _deep_merge(payload, overrides)

    env = EnvOverrides()
    if env.device:
        payload.setdefault("system", {})["device"] = env.device
    if env.log_level:
        payload.setdefault("system", {})["log_level"] = env.log_level
    if env.model_backend:
        payload.setdefault("inference", {})["backend"] = env.model_backend
    if env.model_weights:
        payload.setdefault("inference", {})["weights"] = env.model_weights
    if env.confidence is not None:
        payload.setdefault("inference", {})["confidence"] = env.confidence
    if env.iou is not None:
        payload.setdefault("inference", {})["iou"] = env.iou
    if env.source:
        payload.setdefault("capture", {})["source"] = env.source
    if env.host:
        payload.setdefault("api", {})["host"] = env.host
    if env.port is not None:
        payload.setdefault("api", {})["port"] = env.port

    return AppConfig.model_validate(payload)
