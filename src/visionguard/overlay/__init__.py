"""Industrial HUD overlays."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from visionguard.config import OverlayConfig
from visionguard.hazards.taxonomy import HazardTaxonomy
from visionguard.hazards.zones import ZoneMap
from visionguard.types import FrameResult, NDArrayU8, Severity

_SEVERITY_BGR = {
    Severity.LOW: (180, 180, 180),
    Severity.MEDIUM: (32, 176, 255),
    Severity.HIGH: (20, 140, 255),
    Severity.CRITICAL: (92, 59, 255),
}


def _put(
    image: NDArrayU8,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (240, 240, 240),
    thick: int = 1,
) -> None:
    cv2.putText(
        image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA
    )
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class OverlayRenderer:
    """Draw tracks, zones, risk, and a VisionGuard watermark."""

    def __init__(
        self, taxonomy: HazardTaxonomy, config: OverlayConfig, zones: ZoneMap | None = None
    ) -> None:
        self.taxonomy = taxonomy
        self.config = config
        self.zones = zones or ZoneMap.empty()

    def render(self, result: FrameResult) -> NDArrayU8:
        """Return a copy of the frame with HUD overlays."""

        frame = result.image.copy()
        h, w = frame.shape[:2]
        if self.config.show_zones:
            self._zones(frame, w, h)
        if self.config.show_tracks:
            self._tracks(frame, result)
        if self.config.show_risk:
            self._risk(frame, result.risk_score)
        if self.config.show_fps:
            _put(
                frame,
                f"{result.fps:5.1f} FPS   inf {result.inference_ms:5.1f} ms",
                (16, h - 18),
                0.5,
                (0, 229, 168),
            )
        if self.config.logo:
            _put(frame, "VISIONGUARD", (16, 28), 0.7, (0, 229, 168), 2)
            _put(frame, "HAZARD DETECTION", (16, 52), 0.4, (180, 180, 180), 1)
        self._events(frame, result)
        return frame

    def _zones(self, frame: NDArrayU8, width: int, height: int) -> None:
        overlay = frame.copy()
        for zone in self.zones.zones:
            pts = np.array(zone.pixel_polygon(width, height), dtype=np.int32)
            color = (70, 70, 40) if zone.zone_type != "restricted" else (40, 40, 120)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, (0, 229, 168), 1, cv2.LINE_AA)
            if len(pts):
                _put(
                    frame, zone.name, (int(pts[0][0]) + 6, int(pts[0][1]) + 18), 0.4, (0, 229, 168)
                )
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    def _tracks(self, frame: NDArrayU8, result: FrameResult) -> None:
        for track in result.tracks:
            spec = self.taxonomy.by_detector_name.get(track.class_name)
            color = _SEVERITY_BGR[track.detection.severity]
            if spec:
                color = (spec.color[2], spec.color[1], spec.color[0])  # RGB→BGR
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.track_id}  {track.class_name}  {track.ema_confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )

    def _risk(self, frame: NDArrayU8, risk: float) -> None:
        h, w = frame.shape[:2]
        x0, y0 = w - 150, 16
        cv2.rectangle(frame, (x0, y0), (w - 16, y0 + 64), (16, 16, 16), -1)
        cv2.rectangle(frame, (x0, y0), (w - 16, y0 + 64), (0, 229, 168), 1)
        _put(frame, "RISK", (x0 + 10, y0 + 22), 0.45, (160, 160, 160))
        color = (0, 229, 168) if risk < 35 else ((32, 176, 255) if risk < 70 else (92, 59, 255))
        _put(frame, f"{risk:5.1f}", (x0 + 10, y0 + 50), 0.8, color, 2)

    def _events(self, frame: NDArrayU8, result: FrameResult) -> None:
        y = 80
        for event in result.events[:4]:
            color = _SEVERITY_BGR[event.severity]
            text = f"{event.severity.value.upper()}  {event.name}"
            _put(frame, text, (16, y), 0.5, color, 1)
            y += 22


def draw_legend(frame: NDArrayU8, labels: Iterable[str]) -> NDArrayU8:
    """Tiny helper used by notebooks."""

    y = 80
    out = frame.copy()
    for label in labels:
        _put(out, label, (16, y))
        y += 20
    return out
