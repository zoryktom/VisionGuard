"""End-to-end real-time pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from visionguard.capture import FrameSource, open_source
from visionguard.config import AppConfig, load_config, resolve_path
from visionguard.hazards.fusion import HazardFusion
from visionguard.hazards.taxonomy import HazardTaxonomy, default_taxonomy
from visionguard.hazards.zones import ZoneMap
from visionguard.inference import Detector, build_detector
from visionguard.logging_utils import get_logger
from visionguard.metrics.telemetry import Telemetry
from visionguard.overlay import OverlayRenderer
from visionguard.tracking import ByteTracker
from visionguard.types import Detection, FrameResult, NDArrayU8
from visionguard.utils.device import describe_device, resolve_device
from visionguard.utils.time import Ema, now_s, stopwatch

logger = get_logger(__name__)


class RealTimePipeline:
    """Capture → detect → track → fuse → overlay."""

    def __init__(
        self,
        config: AppConfig,
        taxonomy: HazardTaxonomy | None = None,
        zones: ZoneMap | None = None,
        detector: Detector | None = None,
    ) -> None:
        self.config = config
        self.taxonomy = taxonomy or default_taxonomy()
        self.zones = zones or ZoneMap.empty()
        self.device = resolve_device(config.system.device)
        self.detector = detector or build_detector(config.inference, self.taxonomy, self.device)
        self.tracker = ByteTracker(config.tracking)
        self.fusion = HazardFusion(self.taxonomy, config.hazards)
        self.overlay = OverlayRenderer(self.taxonomy, config.overlay, self.zones)
        self.telemetry = Telemetry()
        self._fps = Ema(0.2, 0.0)
        self._frame_index = 0
        logger.info(
            "Pipeline device=%s backend=%s", describe_device(self.device), config.inference.backend
        )

    def process_frame(self, frame: NDArrayU8) -> FrameResult:
        """Run the stack on a single BGR frame."""

        timestamp = now_s()
        with stopwatch() as elapsed:
            detections: list[Detection] = self.detector.predict(frame)
        inference_ms = elapsed[0]

        tracks = self.tracker.update(detections) if self.config.tracking.enabled else []
        if not self.config.tracking.enabled:
            from visionguard.types import Track

            tracks = [
                Track(
                    track_id=i + 1,
                    detection=det,
                    hits=1,
                    age=1,
                    time_since_update=0,
                    ema_confidence=det.confidence,
                )
                for i, det in enumerate(detections)
            ]

        h, w = frame.shape[:2]
        self.zones.assign(tracks, w, h)
        events, risk = self.fusion.update(tracks, self._frame_index, timestamp)
        fps = self._fps.update(1000.0 / max(inference_ms, 1e-3))
        result = FrameResult(
            frame_index=self._frame_index,
            timestamp=timestamp,
            image=frame,
            detections=detections,
            tracks=tracks,
            events=events,
            risk_score=risk,
            fps=fps,
            inference_ms=inference_ms,
        )
        self.telemetry.observe(result)
        self._frame_index += 1
        return result

    def annotate(self, result: FrameResult) -> NDArrayU8:
        """Draw HUD overlays."""

        return self.overlay.render(result)

    def iter_source(self, source: FrameSource) -> Iterator[FrameResult]:
        """Yield results from an openable frame source."""

        with source:
            for frame in source:
                yield self.process_frame(frame)

    def run_file(
        self,
        source: str,
        output: str | None = None,
        display: bool = False,
        max_frames: int | None = None,
    ) -> list[FrameResult]:
        """Process a video/folder and optionally write an annotated video."""

        cap_cfg = self.config.capture.model_copy(update={"source": source})
        src = open_source(source, cap_cfg)
        writer: cv2.VideoWriter | None = None
        results: list[FrameResult] = []
        try:
            src.open()
            for i, frame in enumerate(src):
                if max_frames is not None and i >= max_frames:
                    break
                result = self.process_frame(frame)
                annotated = self.annotate(result)
                results.append(result)
                if output:
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = resolve_path(output)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(out_path),
                            fourcc,
                            float(self.config.capture.fps),
                            (annotated.shape[1], annotated.shape[0]),
                        )
                    writer.write(annotated)
                if display:
                    cv2.imshow("VisionGuard", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            src.close()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        return results

    @classmethod
    def from_files(
        cls,
        config_path: str | Path | None = None,
        zones_path: str | Path | None = None,
        backend: str | None = None,
    ) -> RealTimePipeline:
        """Build a pipeline from YAML files."""

        overrides = {"inference": {"backend": backend}} if backend else None
        config = load_config(config_path, overrides=overrides)
        taxonomy = default_taxonomy()
        zones = ZoneMap.from_yaml(str(zones_path)) if zones_path else ZoneMap.empty()
        return cls(config, taxonomy, zones)
