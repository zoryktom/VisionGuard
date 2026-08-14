"""Taxonomy tests."""

from visionguard.hazards.taxonomy import default_taxonomy
from visionguard.types import HazardCategory, Severity


def test_taxonomy_loads_and_maps_coco() -> None:
    tax = default_taxonomy()
    assert len(tax.classes) >= 12
    person = tax.map_raw_label("person")
    assert person is not None
    assert person.category is HazardCategory.BEHAVIOR
    phone = tax.map_raw_label("cell phone")
    assert phone is not None
    assert phone.detector_name == "phone"
    truck = tax.map_raw_label("truck")
    assert truck is not None
    assert truck.detector_name == "vehicle"


def test_unknown_label_is_none() -> None:
    tax = default_taxonomy()
    assert tax.map_raw_label("toaster") is None


def test_severity_weights_cover_enum() -> None:
    tax = default_taxonomy()
    for sev in Severity:
        assert sev.value in tax.severity_weights
