"""Detector backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from visionguard.config import InferenceConfig, resolve_path
from visionguard.exceptions import InferenceError
from visionguard.hazards.taxonomy import HazardTaxonomy
from visionguard.logging_utils import get_logger
from visionguard.types import Detection, HazardCategory, NDArrayU8, Severity
from visionguard.utils.device import resolve_device

logger = get_logger(__name__)


class Detector(ABC):
    """Frame-level object detector."""

    @abstractmethod
    def predict(self, frame: NDArrayU8) -> list[Detection]:
        """Run detection on a BGR frame."""

    def warmup(self) -> None:
        """Optional CUDA/MPS warmup."""

        return None

    def close(self) -> None:
        """Release backend resources."""

        return None


class DummyDetector(Detector):
    """Color-blob detector for CI, demos, and tests (no weights required)."""

    def __init__(self, taxonomy: HazardTaxonomy) -> None:
        self.taxonomy = taxonomy

    def predict(self, frame: NDArrayU8) -> list[Detection]:
        hsv = _bgr_to_hsv(frame)
        detections: list[Detection] = []
        detections.extend(
            self._find(frame, hsv, "person", (85, 80, 80), (110, 255, 255), min_area=800)
        )
        detections.extend(
            self._find(frame, hsv, "spill", (18, 80, 80), (40, 255, 255), min_area=600)
        )
        detections.extend(
            self._find(frame, hsv, "fire", (0, 140, 100), (12, 255, 255), min_area=500)
        )
        detections.extend(
            self._find(frame, hsv, "vehicle", (0, 0, 50), (180, 50, 160), min_area=2000)
        )
        return detections

    def _find(
        self,
        frame: NDArrayU8,
        hsv: NDArrayU8,
        class_name: str,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
        min_area: int,
    ) -> list[Detection]:
        import cv2

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        out: list[Detection] = []
        spec = self.taxonomy.by_detector_name.get(class_name)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            class_id = spec.class_id if spec else 0
            out.append(
                Detection(
                    bbox=(float(x), float(y), float(x2), float(y2)),
                    class_id=class_id,
                    class_name=spec.detector_name if spec else class_name,
                    confidence=min(0.99, 0.55 + area / (h * w)),
                    category=spec.category if spec else HazardCategory.OBJECT,
                    severity=spec.severity if spec else Severity.MEDIUM,
                )
            )
        return out


def _bgr_to_hsv(frame: NDArrayU8) -> NDArrayU8:
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


class UltralyticsDetector(Detector):
    """YOLOv8 / YOLO11 / RT-DETR via the Ultralytics API."""

    def __init__(
        self,
        config: InferenceConfig,
        taxonomy: HazardTaxonomy,
        device: torch.device,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise InferenceError("ultralytics is not installed") from exc

        weights = _resolve_weights(config)
        self._model = YOLO(str(weights))
        self._config = config
        self._taxonomy = taxonomy
        self._device = device
        self._names: dict[int, str] = dict(self._model.names)

    def warmup(self) -> None:
        dummy = np.zeros((self._config.imgsz, self._config.imgsz, 3), dtype=np.uint8)
        self.predict(dummy)

    def predict(self, frame: NDArrayU8) -> list[Detection]:
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self._config.imgsz,
                conf=self._config.confidence,
                iou=self._config.iou,
                max_det=self._config.max_detections,
                device=str(self._device),
                half=self._config.half and self._device.type == "cuda",
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover
            raise InferenceError(f"YOLO inference failed: {exc}") from exc

        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, conf, cls, strict=False):
            raw_name = self._names.get(int(class_id), str(class_id))
            mapped = self._taxonomy.map_raw_label(raw_name)
            if mapped is None:
                continue
            detections.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    class_id=mapped.class_id,
                    class_name=mapped.detector_name,
                    confidence=float(score),
                    category=mapped.category,
                    severity=mapped.severity,
                )
            )
        return detections


class VGNetDetector(Detector):
    """Tiny in-repo single-stage detector (see ``visionguard.training.vgnet``)."""

    def __init__(
        self,
        weights: str,
        taxonomy: HazardTaxonomy,
        device: torch.device,
        confidence: float = 0.35,
    ) -> None:
        from visionguard.training.vgnet import VGNet, load_vgnet

        path = resolve_path(weights)
        if not path.exists():
            raise InferenceError(f"VGNet weights not found: {path}")
        self._model: VGNet = load_vgnet(path, num_classes=len(taxonomy.classes), device=device)
        self._model.eval()
        self._taxonomy = taxonomy
        self._device = device
        self._confidence = confidence

    @torch.inference_mode()
    def predict(self, frame: NDArrayU8) -> list[Detection]:
        from visionguard.training.vgnet import decode_detections

        return decode_detections(self._model, frame, self._taxonomy, self._device, self._confidence)


def _resolve_weights(config: InferenceConfig) -> Path | str:
    custom = resolve_path(config.custom_weights)
    if custom.exists():
        logger.info("Using custom VisionGuard weights: %s", custom)
        return custom
    bundled = resolve_path(config.weights)
    if bundled.exists():
        return bundled
    return config.weights  # ultralytics will download official names like yolov8n.pt


def build_detector(
    config: InferenceConfig,
    taxonomy: HazardTaxonomy,
    device: torch.device | None = None,
) -> Detector:
    """Instantiate a detector backend from config."""

    backend = config.backend.lower().strip()
    if backend in {"dummy", "synthetic"}:
        return DummyDetector(taxonomy)
    resolved = device or resolve_device("auto")
    if backend in {"ultralytics", "yolo", "yolov8", "rtdetr"}:
        return UltralyticsDetector(config, taxonomy, resolved)
    if backend == "vgnet":
        return VGNetDetector(config.custom_weights, taxonomy, resolved, config.confidence)
    raise InferenceError(f"Unknown detector backend: {config.backend}")
