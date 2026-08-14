"""Hazard fusion tests."""

from visionguard.config import HazardsRuntimeConfig
from visionguard.hazards.fusion import HazardFusion
from visionguard.hazards.taxonomy import default_taxonomy
from visionguard.types import Detection, Track


def _track(tid: int, det: Detection) -> Track:
    return Track(
        track_id=tid,
        detection=det,
        hits=5,
        age=5,
        time_since_update=0,
        ema_confidence=det.confidence,
    )


def test_fire_persists_into_event(fire_det: Detection) -> None:
    fusion = HazardFusion(
        default_taxonomy(), HazardsRuntimeConfig(persist_frames=3, cooldown_seconds=0.0)
    )
    track = _track(1, fire_det)
    events = []
    for i in range(4):
        ev, risk = fusion.update([track], i, timestamp=float(i))
        events.extend(ev)
    assert any(e.name == "fire" for e in events)
    assert risk > 0


def test_near_miss_person_vehicle(person_det: Detection, vehicle_det: Detection) -> None:
    fusion = HazardFusion(
        default_taxonomy(),
        HazardsRuntimeConfig(persist_frames=8, cooldown_seconds=0.0, near_miss_px=500),
    )
    tracks = [_track(1, person_det), _track(2, vehicle_det)]
    events = []
    for i in range(6):
        ev, _ = fusion.update(tracks, i, timestamp=float(i))
        events.extend(ev)
    assert any(e.name == "pedestrian_vehicle_near_miss" for e in events)


def test_low_severity_person_does_not_event(person_det: Detection) -> None:
    fusion = HazardFusion(
        default_taxonomy(), HazardsRuntimeConfig(persist_frames=2, cooldown_seconds=0.0)
    )
    events = []
    for i in range(5):
        ev, _ = fusion.update([_track(3, person_det)], i, timestamp=float(i))
        events.extend(ev)
    assert events == []
