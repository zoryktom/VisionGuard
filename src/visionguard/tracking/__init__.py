"""Kalman + IoU tracker (ByteTrack-inspired, no extra deps)."""

from __future__ import annotations

import numpy as np

from visionguard.config import TrackingConfig
from visionguard.types import Detection, Track
from visionguard.utils.geometry import iou_xyxy


class KalmanBox:
    """Constant-velocity Kalman filter on XYXY boxes."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        self.mean = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.cov = np.eye(8, dtype=np.float32) * 10.0

    def predict(self) -> None:
        """Advance the constant-velocity model by one frame."""

        f = np.eye(8, dtype=np.float32)
        f[0, 4] = 1.0
        f[1, 5] = 1.0
        f[2, 6] = 1.0
        f[3, 7] = 1.0
        self.mean = f @ self.mean
        self.cov = f @ self.cov @ f.T + np.eye(8, dtype=np.float32) * 1.0

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        """Correct with a new measurement."""

        x1, y1, x2, y2 = bbox
        z = np.array(
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)],
            dtype=np.float32,
        )
        h = np.zeros((4, 8), dtype=np.float32)
        h[0, 0] = h[1, 1] = h[2, 2] = h[3, 3] = 1.0
        r = np.eye(4, dtype=np.float32) * 2.0
        s = h @ self.cov @ h.T + r
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ (z - h @ self.mean)
        self.cov = (np.eye(8, dtype=np.float32) - k @ h) @ self.cov

    def as_bbox(self) -> tuple[float, float, float, float]:
        """Convert state back to XYXY."""

        cx, cy, w, h = self.mean[:4]
        w, h = max(1.0, float(w)), max(1.0, float(h))
        return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


class _TrackState:
    def __init__(self, track_id: int, detection: Detection, ema_alpha: float) -> None:
        self.track_id = track_id
        self.detection = detection
        self.kf = KalmanBox(detection.bbox)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_confidence = detection.confidence
        self.ema_alpha = ema_alpha
        self.zone: str | None = None

    def predict_step(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        x1, y1, x2, y2 = self.kf.as_bbox()
        self.detection.bbox = (x1, y1, x2, y2)

    def correct(self, detection: Detection) -> None:
        self.kf.update(detection.bbox)
        detection.bbox = self.kf.as_bbox()
        self.detection = detection
        self.hits += 1
        self.time_since_update = 0
        a = self.ema_alpha
        self.ema_confidence = a * detection.confidence + (1 - a) * self.ema_confidence

    def as_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            detection=self.detection,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            ema_confidence=self.ema_confidence,
            zone=self.zone,
        )


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Greedy assignment (stable enough for modest N; O(n^2))."""

    if cost.size == 0:
        return []
    used_r: set[int] = set()
    used_c: set[int] = set()
    pairs: list[tuple[int, int]] = []
    flat = [(cost[r, c], r, c) for r in range(cost.shape[0]) for c in range(cost.shape[1])]
    for _, r, c in sorted(flat):
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    return pairs


class ByteTracker:
    """Class-aware IoU matcher with Kalman motion."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Predict, match, spawn, and prune tracks."""

        for track in self._tracks:
            track.predict_step()

        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self._tracks)))
        if self._tracks and detections:
            iou = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
            for t, track in enumerate(self._tracks):
                for d, det in enumerate(detections):
                    if track.detection.class_name != det.class_name:
                        iou[t, d] = 0.0
                    else:
                        iou[t, d] = iou_xyxy(track.kf.as_bbox(), det.bbox)
            cost = 1.0 - iou
            for t, d in _hungarian(cost):
                if iou[t, d] < self.config.iou_threshold:
                    continue
                self._tracks[t].correct(detections[d])
                if t in unmatched_trks:
                    unmatched_trks.remove(t)
                if d in unmatched_dets:
                    unmatched_dets.remove(d)

        for d in unmatched_dets:
            self._tracks.append(_TrackState(self._next_id, detections[d], self.config.ema_alpha))
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.config.max_age]
        confirmed = [t.as_track() for t in self._tracks if t.hits >= self.config.min_hits]
        return confirmed
