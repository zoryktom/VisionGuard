"""Hazard package."""

from visionguard.hazards.fusion import HazardFusion
from visionguard.hazards.taxonomy import HazardTaxonomy, default_taxonomy
from visionguard.hazards.zones import ZoneMap

__all__ = ["HazardFusion", "HazardTaxonomy", "ZoneMap", "default_taxonomy"]
