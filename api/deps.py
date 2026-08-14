"""Shared FastAPI dependencies."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

from visionguard.config import AppConfig, load_config
from visionguard.hazards.zones import ZoneMap
from visionguard.pipeline import RealTimePipeline
from visionguard.utils.device import describe_device

_LOCK = Lock()
_PIPELINE: RealTimePipeline | None = None


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Load process-wide config."""

    return load_config()


def reset_pipeline() -> None:
    """Drop the singleton (tests / config reload)."""

    global _PIPELINE
    with _LOCK:
        _PIPELINE = None
        get_app_config.cache_clear()


def get_pipeline() -> RealTimePipeline:
    """Lazily construct a singleton pipeline (thread-safe)."""

    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                config = get_app_config()
                zones_path = "configs/zones.example.yaml"
                try:
                    zones = ZoneMap.from_yaml(zones_path)
                except Exception:
                    zones = ZoneMap.empty()
                _PIPELINE = RealTimePipeline(config, zones=zones)
    return _PIPELINE


def pipeline_info() -> dict[str, str]:
    """Device / backend strings for health checks."""

    pipe = get_pipeline()
    return {
        "device": describe_device(pipe.device),
        "backend": pipe.config.inference.backend,
    }
