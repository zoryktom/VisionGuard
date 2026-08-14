"""Video source abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from visionguard.config import CaptureConfig, resolve_path
from visionguard.exceptions import CaptureError
from visionguard.types import NDArrayU8
from visionguard.utils.image import to_bgr


class FrameSource(ABC):
    """Iterable sequence of BGR frames."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying resource."""

    @abstractmethod
    def read(self) -> NDArrayU8 | None:
        """Return the next frame, or ``None`` at end-of-stream."""

    @abstractmethod
    def close(self) -> None:
        """Release the resource."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[NDArrayU8]:
        self.open()
        try:
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.close()


class OpenCVSource(FrameSource):
    """Webcam, file, or RTSP source via OpenCV."""

    def __init__(self, source: str | int, config: CaptureConfig) -> None:
        self._source = source
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._width = config.width
        self._height = config.height

    def open(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise CaptureError(f"Unable to open video source: {self._source}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._config.buffer_size)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self._config.width)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._config.height)
        self._cap = cap

    def read(self) -> NDArrayU8 | None:
        if self._cap is None:
            raise CaptureError("Source is not open")
        if self._config.drop_stale_frames:
            for _ in range(max(0, self._config.buffer_size - 1)):
                self._cap.grab()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return to_bgr(frame)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


class ImageFolderSource(FrameSource):
    """Ordered directory of images (jpg/png/bmp)."""

    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._paths: list[Path] = []
        self._index = 0
        self._width = 0
        self._height = 0

    def open(self) -> None:
        if not self._folder.is_dir():
            raise CaptureError(f"Not a directory: {self._folder}")
        self._paths = sorted(p for p in self._folder.iterdir() if p.suffix.lower() in self._EXTS)
        if not self._paths:
            raise CaptureError(f"No images in {self._folder}")
        probe = cv2.imread(str(self._paths[0]))
        if probe is None:
            raise CaptureError(f"Failed to read {self._paths[0]}")
        self._height, self._width = probe.shape[:2]
        self._index = 0

    def read(self) -> NDArrayU8 | None:
        if self._index >= len(self._paths):
            return None
        path = self._paths[self._index]
        self._index += 1
        frame = cv2.imread(str(path))
        if frame is None:
            return self.read()
        return to_bgr(frame)

    def close(self) -> None:
        self._paths = []
        self._index = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height


def open_source(source: str, config: CaptureConfig | None = None) -> FrameSource:
    """Factory: webcam index, file, RTSP URL, or image directory."""

    cfg = config or CaptureConfig(source=source)
    raw: str | int = source
    if source.isdigit():
        raw = int(source)
        return OpenCVSource(raw, cfg)

    path = (
        resolve_path(source) if not source.startswith(("rtsp://", "http://", "https://")) else None
    )
    if path is not None and path.is_dir():
        return ImageFolderSource(path)
    if path is not None and path.exists():
        return OpenCVSource(str(path), cfg)
    return OpenCVSource(source, cfg)


def synthetic_frame(width: int = 1280, height: int = 720, seed: int = 0) -> NDArrayU8:
    """Deterministic warehouse-like frame used by tests and dummy inference."""

    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (28, 24, 18)
    cv2.rectangle(frame, (0, int(height * 0.62)), (width, height), (42, 38, 32), -1)
    cv2.line(frame, (int(width * 0.7), 0), (int(width * 0.7), height), (70, 64, 52), 2)

    def blob(color: tuple[int, int, int], xy: tuple[int, int], size: tuple[int, int]) -> None:
        x, y = xy
        w, h = size
        jitter = rng.integers(-6, 7, size=2)
        x1, y1 = x + int(jitter[0]), y + int(jitter[1])
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, -1)

    blob((210, 160, 30), (180, 260), (70, 180))  # person (cyan vest, BGR)
    blob((20, 200, 230), (520, 300), (90, 70))  # spill (yellow)
    blob((30, 30, 240), (860, 180), (110, 80))  # fire (red)
    blob((110, 110, 110), (980, 340), (160, 90))  # vehicle (gray)
    noise = rng.integers(0, 12, size=frame.shape, dtype=np.uint8)
    return cv2.add(frame, noise)
