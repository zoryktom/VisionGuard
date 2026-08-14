"""Pipeline + dummy detector tests."""

from visionguard.capture import synthetic_frame
from visionguard.config import load_config
from visionguard.hazards.zones import ZoneMap
from visionguard.inference import DummyDetector, build_detector
from visionguard.pipeline import RealTimePipeline


def test_dummy_detector_finds_blobs(taxonomy, frame) -> None:
    dets = DummyDetector(taxonomy).predict(frame)
    names = {d.class_name for d in dets}
    assert "person" in names or "fire" in names or "vehicle" in names or "spill" in names


def test_build_dummy_backend(taxonomy) -> None:
    cfg = load_config(overrides={"inference": {"backend": "dummy"}})
    det = build_detector(cfg.inference, taxonomy)
    assert isinstance(det, DummyDetector)


def test_pipeline_processes_synthetic_frame() -> None:
    pipe = RealTimePipeline.from_files(backend="dummy", zones_path="configs/zones.example.yaml")
    frame = synthetic_frame(640, 384, seed=2)
    result = pipe.process_frame(frame)
    annotated = pipe.annotate(result)
    assert annotated.shape == frame.shape
    assert result.inference_ms >= 0
    assert 0 <= result.risk_score <= 100
    snap = pipe.telemetry.snapshot()
    assert snap["frames"] == 1


def test_zone_map_loads() -> None:
    zones = ZoneMap.from_yaml("configs/zones.example.yaml")
    assert len(zones.zones) >= 3
