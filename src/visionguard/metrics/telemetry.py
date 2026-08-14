"""Lightweight in-process telemetry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from visionguard.types import FrameResult, HazardEvent


@dataclass
class Telemetry:
    """Rolling window of FPS, latency, risk, and events."""

    window: int = 120
    inference_ms: deque[float] = field(default_factory=deque)
    fps: deque[float] = field(default_factory=deque)
    risk: deque[float] = field(default_factory=deque)
    events: deque[HazardEvent] = field(default_factory=deque)
    frames: int = 0

    def observe(self, result: FrameResult) -> None:
        """Record one frame."""

        self.frames += 1
        self._push(self.inference_ms, result.inference_ms)
        self._push(self.fps, result.fps)
        self._push(self.risk, result.risk_score)
        for event in result.events:
            self.events.append(event)
            while len(self.events) > 256:
                self.events.popleft()

    def _push(self, buf: deque[float], value: float) -> None:
        buf.append(value)
        while len(buf) > self.window:
            buf.popleft()

    def snapshot(self) -> dict[str, float | int]:
        """JSON-serializable gauges."""

        def mean(buf: deque[float]) -> float:
            return float(sum(buf) / len(buf)) if buf else 0.0

        return {
            "frames": self.frames,
            "fps": mean(self.fps),
            "inference_ms": mean(self.inference_ms),
            "risk": mean(self.risk),
            "events": len(self.events),
        }

    def prometheus(self) -> str:
        """OpenMetrics text exposition."""

        snap = self.snapshot()
        lines = [
            "# HELP visionguard_fps Smoothed frames per second",
            "# TYPE visionguard_fps gauge",
            f"visionguard_fps {snap['fps']}",
            "# HELP visionguard_inference_ms Mean inference latency",
            "# TYPE visionguard_inference_ms gauge",
            f"visionguard_inference_ms {snap['inference_ms']}",
            "# HELP visionguard_risk_score Fused site risk 0-100",
            "# TYPE visionguard_risk_score gauge",
            f"visionguard_risk_score {snap['risk']}",
            "# HELP visionguard_frames_total Frames processed",
            "# TYPE visionguard_frames_total counter",
            f"visionguard_frames_total {snap['frames']}",
            "# HELP visionguard_events_buffered Events in ring buffer",
            "# TYPE visionguard_events_buffered gauge",
            f"visionguard_events_buffered {snap['events']}",
        ]
        return "\n".join(lines) + "\n"
