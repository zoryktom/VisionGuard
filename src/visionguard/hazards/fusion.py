"""Temporal hazard fusion: persistence, proximity, calibrated risk."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field

from visionguard.config import HazardsRuntimeConfig
from visionguard.hazards.taxonomy import HazardTaxonomy
from visionguard.types import (
    BBoxXYXY,
    HazardCategory,
    HazardEvent,
    Severity,
    Track,
)
from visionguard.utils.geometry import euclidean


def _union(a: BBoxXYXY, b: BBoxXYXY) -> BBoxXYXY:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _event_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


@dataclass
class _Hypothesis:
    count: int = 0
    last_seen: float = 0.0
    last_emitted: float = 0.0
    confidence: float = 0.0
    bbox: BBoxXYXY = (0.0, 0.0, 0.0, 0.0)
    zone: str | None = None
    track_ids: tuple[int, ...] = ()


@dataclass
class FusionState:
    """Mutable temporal state kept across frames."""

    hypotheses: dict[str, _Hypothesis] = field(default_factory=lambda: defaultdict(_Hypothesis))
    risk_ema: float = 0.0
    frame_index: int = 0


class HazardFusion:
    """Convert noisy per-frame tracks into durable, de-duplicated events.

    A detection is promoted to an event only after it persists for
    ``persist_frames``. Pairwise proximity rules emit interaction events
    (near-miss, slip risk). Instantaneous risk is an EMA of severity-weighted
    evidence, multiplied by zone factors.
    """

    def __init__(self, taxonomy: HazardTaxonomy, config: HazardsRuntimeConfig) -> None:
        self.taxonomy = taxonomy
        self.config = config
        self.state = FusionState()

    def _name_matches(self, track_name: str, rule_name: str) -> bool:
        """Match a track label against a rule side (detector name or hazard id)."""

        if track_name == rule_name:
            return True
        spec = self.taxonomy.by_detector_name.get(track_name) or self.taxonomy.by_hazard_id.get(
            track_name
        )
        if spec is None:
            return False
        return rule_name in {spec.detector_name, spec.hazard_id}

    def reset(self) -> None:
        """Clear temporal state (new video)."""

        self.state = FusionState()

    def update(
        self, tracks: list[Track], frame_index: int, timestamp: float | None = None
    ) -> tuple[list[HazardEvent], float]:
        """Ingest tracks for one frame. Returns (events, risk_score 0-100)."""

        now = timestamp if timestamp is not None else time.time()
        self.state.frame_index = frame_index
        events: list[HazardEvent] = []
        events.extend(self._persist_tracks(tracks, frame_index, now))
        events.extend(self._proximity(tracks, frame_index, now))
        risk = self._score(tracks)
        return events, risk

    def _persist_tracks(
        self, tracks: list[Track], frame_index: int, now: float
    ) -> list[HazardEvent]:
        events: list[HazardEvent] = []
        seen: set[str] = set()
        for track in tracks:
            spec = self.taxonomy.by_detector_name.get(track.class_name)
            if spec is None or spec.severity == Severity.LOW:
                continue
            key = f"track:{track.class_name}:{track.track_id}"
            seen.add(key)
            hyp = self.state.hypotheses[key]
            hyp.count += 1
            hyp.last_seen = now
            hyp.confidence = track.ema_confidence
            hyp.bbox = track.bbox
            hyp.zone = track.zone
            hyp.track_ids = (track.track_id,)
            if hyp.count < self.config.persist_frames:
                continue
            if now - hyp.last_emitted < self.config.cooldown_seconds:
                continue
            hyp.last_emitted = now
            events.append(
                HazardEvent(
                    event_id=_event_key(key, frame_index),
                    name=spec.hazard_id,
                    category=spec.category,
                    severity=spec.severity,
                    confidence=track.ema_confidence,
                    track_ids=(track.track_id,),
                    bbox=track.bbox,
                    frame_index=frame_index,
                    timestamp=now,
                    zone=track.zone,
                    message=spec.description,
                )
            )
        self._decay_missing(seen, now)
        return events

    def _proximity(self, tracks: list[Track], frame_index: int, now: float) -> list[HazardEvent]:
        events: list[HazardEvent] = []
        for rule in self.taxonomy.proximity_rules:
            lefts = [t for t in tracks if self._name_matches(t.class_name, rule.left)]
            rights = [t for t in tracks if self._name_matches(t.class_name, rule.right)]
            for a in lefts:
                for b in rights:
                    if a.track_id == b.track_id:
                        continue
                    dist = euclidean(a.centroid, b.centroid)
                    if dist > self.config.near_miss_px:
                        continue
                    pair = tuple(sorted((a.track_id, b.track_id)))
                    key = f"prox:{rule.event}:{pair}"
                    hyp = self.state.hypotheses[key]
                    hyp.count += 1
                    hyp.last_seen = now
                    hyp.confidence = min(a.ema_confidence, b.ema_confidence)
                    hyp.bbox = _union(a.bbox, b.bbox)
                    hyp.zone = a.zone or b.zone
                    hyp.track_ids = pair
                    if hyp.count < max(3, self.config.persist_frames // 2):
                        continue
                    if now - hyp.last_emitted < self.config.cooldown_seconds:
                        continue
                    hyp.last_emitted = now
                    events.append(
                        HazardEvent(
                            event_id=_event_key(key, frame_index),
                            name=rule.event,
                            category=HazardCategory.INTERACTION,
                            severity=rule.severity,
                            confidence=hyp.confidence,
                            track_ids=pair,
                            bbox=hyp.bbox,
                            frame_index=frame_index,
                            timestamp=now,
                            zone=hyp.zone,
                            message=f"{rule.left} near {rule.right} ({dist:.0f}px)",
                        )
                    )
        return events

    def _decay_missing(self, seen: set[str], now: float) -> None:
        stale = [
            key
            for key, hyp in self.state.hypotheses.items()
            if key not in seen and now - hyp.last_seen > 2.0
        ]
        for key in stale:
            del self.state.hypotheses[key]

    def _score(self, tracks: list[Track]) -> float:
        instant = 0.0
        for track in tracks:
            spec = self.taxonomy.by_detector_name.get(track.class_name)
            if spec is None:
                continue
            weight = self.taxonomy.severity_weights.get(spec.severity.value, 0.4)
            zone_key = "general"
            if track.zone:
                for ztype, _mult in self.taxonomy.zone_multipliers.items():
                    if ztype in track.zone or track.zone == ztype:
                        zone_key = ztype
                        break
                # zone names like east_exit → type exit
                if "exit" in (track.zone or ""):
                    zone_key = "exit"
                elif "ppe" in (track.zone or ""):
                    zone_key = "ppe_required"
                elif "lane" in (track.zone or "") or "vehicle" in (track.zone or ""):
                    zone_key = "vehicle_lane"
                elif "restrict" in (track.zone or "") or "chemical" in (track.zone or ""):
                    zone_key = "restricted"
            mult = self.taxonomy.zone_multipliers.get(zone_key, 1.0)
            instant += weight * track.ema_confidence * 40.0 * mult
        instant = max(0.0, min(100.0, instant))
        alpha = self.config.risk_ema
        self.state.risk_ema = alpha * instant + (1.0 - alpha) * self.state.risk_ema
        return float(self.state.risk_ema)
