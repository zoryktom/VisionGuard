"""Shared fixtures."""

from __future__ import annotations

import os

os.environ.setdefault("VISIONGUARD_MODEL_BACKEND", "dummy")
os.environ.setdefault("VISIONGUARD_DEVICE", "cpu")

import numpy as np
import pytest

from visionguard.capture import synthetic_frame
from visionguard.config import AppConfig, load_config
from visionguard.hazards.taxonomy import default_taxonomy
from visionguard.types import Detection, HazardCategory, Severity


@pytest.fixture(scope="session")
def taxonomy():
    return default_taxonomy()


@pytest.fixture
def config() -> AppConfig:
    return load_config(overrides={"inference": {"backend": "dummy"}, "system": {"device": "cpu"}})


@pytest.fixture
def frame() -> np.ndarray:
    return synthetic_frame(640, 384, seed=1)


@pytest.fixture
def person_det() -> Detection:
    return Detection(
        bbox=(100, 100, 180, 280),
        class_id=0,
        class_name="person",
        confidence=0.9,
        category=HazardCategory.BEHAVIOR,
        severity=Severity.LOW,
    )


@pytest.fixture
def fire_det() -> Detection:
    return Detection(
        bbox=(400, 80, 500, 160),
        class_id=7,
        class_name="fire",
        confidence=0.92,
        category=HazardCategory.ENVIRONMENT,
        severity=Severity.CRITICAL,
    )


@pytest.fixture
def vehicle_det() -> Detection:
    return Detection(
        bbox=(480, 200, 620, 300),
        class_id=11,
        class_name="vehicle",
        confidence=0.88,
        category=HazardCategory.OBJECT,
        severity=Severity.MEDIUM,
    )
