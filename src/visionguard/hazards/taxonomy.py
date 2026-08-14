"""Hazard taxonomy loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass

from visionguard.config import load_yaml, resolve_path
from visionguard.types import HazardCategory, Severity


@dataclass(frozen=True, slots=True)
class HazardSpec:
    """One class in the VisionGuard taxonomy."""

    class_id: int
    hazard_id: str
    detector_name: str
    category: HazardCategory
    severity: Severity
    description: str
    coco_aliases: tuple[str, ...]
    color: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ProximityRule:
    """Pairwise spatial interaction that emits a compound event."""

    left: str
    right: str
    event: str
    severity: Severity


class HazardTaxonomy:
    """Bidirectional maps between detector labels and hazard metadata."""

    def __init__(
        self,
        specs: list[HazardSpec],
        rules: list[ProximityRule],
        zone_multipliers: dict[str, float],
        severity_weights: dict[str, float],
    ) -> None:
        self.classes = specs
        self.proximity_rules = rules
        self.zone_multipliers = zone_multipliers
        self.severity_weights = severity_weights
        self.by_detector_name = {s.detector_name: s for s in specs}
        self.by_hazard_id = {s.hazard_id: s for s in specs}
        alias: dict[str, HazardSpec] = {}
        for spec in specs:
            alias[spec.detector_name.lower()] = spec
            alias[spec.hazard_id.lower()] = spec
            for name in spec.coco_aliases:
                alias[name.lower()] = spec
        self._alias = alias

    def map_raw_label(self, name: str) -> HazardSpec | None:
        """Map a backend label (COCO or custom) onto a hazard spec."""

        return self._alias.get(name.lower().strip())

    def names(self) -> list[str]:
        """Detector class names in class_id order."""

        return [s.detector_name for s in sorted(self.classes, key=lambda s: s.class_id)]

    @classmethod
    def from_yaml(cls, path: str = "configs/hazards.yaml") -> HazardTaxonomy:
        """Load taxonomy from disk."""

        data = load_yaml(path)
        specs: list[HazardSpec] = []
        class_id = 0
        categories = data.get("categories", {})
        for cat_name, cat_body in categories.items():
            color = tuple(int(c) for c in cat_body.get("color", [180, 180, 180]))
            if len(color) != 3:
                color = (180, 180, 180)
            category = HazardCategory(cat_name)
            for item in cat_body.get("classes", []):
                specs.append(
                    HazardSpec(
                        class_id=class_id,
                        hazard_id=str(item["id"]),
                        detector_name=str(item["detector_name"]),
                        category=category,
                        severity=Severity(str(item["severity"])),
                        description=str(item.get("description", "")),
                        coco_aliases=tuple(str(a) for a in item.get("coco_aliases", [])),
                        color=(color[0], color[1], color[2]),
                    )
                )
                class_id += 1
        rules = [
            ProximityRule(
                left=str(r["left"]),
                right=str(r["right"]),
                event=str(r["event"]),
                severity=Severity(str(r["severity"])),
            )
            for r in data.get("proximity_rules", [])
        ]
        multipliers = {str(k): float(v) for k, v in data.get("zone_multipliers", {}).items()}
        weights = {str(k): float(v) for k, v in data.get("severity_weights", {}).items()}
        return cls(specs, rules, multipliers, weights)


def default_taxonomy() -> HazardTaxonomy:
    """Load the bundled taxonomy."""

    return HazardTaxonomy.from_yaml(str(resolve_path("configs/hazards.yaml")))
