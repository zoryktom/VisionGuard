"""Overlay smoke test."""

from visionguard.capture import synthetic_frame
from visionguard.pipeline import RealTimePipeline


def test_overlay_draws_without_error() -> None:
    pipe = RealTimePipeline.from_files(backend="dummy")
    frame = synthetic_frame(320, 192, seed=4)
    result = pipe.process_frame(frame)
    out = pipe.annotate(result)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
