"""Tracker tests."""

from visionguard.config import TrackingConfig
from visionguard.tracking import ByteTracker
from visionguard.types import Detection, HazardCategory, Severity


def _det(x: float, name: str = "person", conf: float = 0.9) -> Detection:
    return Detection(
        bbox=(x, 40, x + 40, 120),
        class_id=0,
        class_name=name,
        confidence=conf,
        category=HazardCategory.BEHAVIOR,
        severity=Severity.LOW,
    )


def test_tracker_persists_identity() -> None:
    tracker = ByteTracker(TrackingConfig(min_hits=2, max_age=5, iou_threshold=0.2))
    t1 = tracker.update([_det(100)])
    t2 = tracker.update([_det(104)])
    t3 = tracker.update([_det(108)])
    assert t1 == []  # not yet confirmed
    assert len(t2) == 1
    assert len(t3) == 1
    assert t2[0].track_id == t3[0].track_id


def test_tracker_separates_classes() -> None:
    tracker = ByteTracker(TrackingConfig(min_hits=1, iou_threshold=0.1))
    tracks = tracker.update([_det(100, "person"), _det(100, "fire")])
    names = {t.class_name for t in tracks}
    assert names == {"person", "fire"}
    assert len({t.track_id for t in tracks}) == 2
