"""Shared types for the VisionGuard runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

NDArrayU8: TypeAlias = npt.NDArray[np.uint8]
NDArrayF32: TypeAlias = npt.NDArray[np.float32]
BBoxXYXY: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = Sequence[Point]


class Severity(str, Enum):
    """Hazard severity used for risk fusion and UI coloring."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HazardCategory(str, Enum):
    """Top-level taxonomy buckets."""

    BEHAVIOR = "behavior"
    ENVIRONMENT = "environment"
    OBJECT = "object"
    INTERACTION = "interaction"


@dataclass(slots=True)
class Detection:
    """Single-frame detector output in pixel XYXY coordinates."""

    bbox: BBoxXYXY
    class_id: int
    class_name: str
    confidence: float
    category: HazardCategory = HazardCategory.OBJECT
    severity: Severity = Severity.MEDIUM

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(slots=True)
class Track:
    """Persistent identity wrapping a detection across frames."""

    track_id: int
    detection: Detection
    hits: int
    age: int
    time_since_update: int
    ema_confidence: float
    zone: str | None = None

    @property
    def bbox(self) -> BBoxXYXY:
        return self.detection.bbox

    @property
    def class_name(self) -> str:
        return self.detection.class_name

    @property
    def centroid(self) -> Point:
        return self.detection.centroid


@dataclass(slots=True)
class HazardEvent:
    """Deduplicated, temporally confirmed safety event."""

    event_id: str
    name: str
    category: HazardCategory
    severity: Severity
    confidence: float
    track_ids: tuple[int, ...]
    bbox: BBoxXYXY
    frame_index: int
    timestamp: float
    zone: str | None = None
    message: str = ""


@dataclass(slots=True)
class FrameResult:
    """Output of one pass through the real-time pipeline."""

    frame_index: int
    timestamp: float
    image: NDArrayU8
    detections: list[Detection]
    tracks: list[Track]
    events: list[HazardEvent]
    risk_score: float
    fps: float
    inference_ms: float
    extras: dict[str, object] = field(default_factory=dict)
