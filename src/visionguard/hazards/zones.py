"""Named polygonal safety zones."""

from __future__ import annotations

from dataclasses import dataclass

from visionguard.config import load_yaml
from visionguard.types import Point, Polygon, Track
from visionguard.utils.geometry import point_in_polygon


@dataclass(frozen=True, slots=True)
class Zone:
    """A named polygon in normalized (0-1) or pixel coordinates."""

    name: str
    zone_type: str
    polygon: Polygon
    normalized: bool = True

    def contains(self, point: Point, width: int, height: int) -> bool:
        """True if ``point`` (pixels) is inside this zone."""

        if self.normalized:
            px = [(x * width, y * height) for x, y in self.polygon]
        else:
            px = list(self.polygon)
        return point_in_polygon(point, px)

    def pixel_polygon(self, width: int, height: int) -> list[tuple[int, int]]:
        """Integer pixel vertices for drawing."""

        if self.normalized:
            return [(int(x * width), int(y * height)) for x, y in self.polygon]
        return [(int(x), int(y)) for x, y in self.polygon]


class ZoneMap:
    """Collection of safety zones."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones

    def assign(self, tracks: list[Track], width: int, height: int) -> None:
        """Stamp ``track.zone`` with the highest-priority matching zone type."""

        priority = ["restricted", "exit", "ppe_required", "vehicle_lane", "general"]
        rank = {name: i for i, name in enumerate(priority)}
        for track in tracks:
            hits = [z for z in self.zones if z.contains(track.centroid, width, height)]
            if not hits:
                track.zone = None
                continue
            hits.sort(key=lambda z: rank.get(z.zone_type, 99))
            track.zone = hits[0].name

    @classmethod
    def from_yaml(cls, path: str) -> ZoneMap:
        """Load zones from YAML."""

        data = load_yaml(path)
        zones = [
            Zone(
                name=str(item["name"]),
                zone_type=str(item["type"]),
                polygon=[(float(p[0]), float(p[1])) for p in item["polygon"]],
                normalized=bool(item.get("normalized", True)),
            )
            for item in data.get("zones", [])
        ]
        return cls(zones)

    @classmethod
    def empty(cls) -> ZoneMap:
        """No zones configured."""

        return cls([])
