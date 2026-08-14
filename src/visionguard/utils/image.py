"""Image conversion utilities."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from visionguard.types import NDArrayU8


def to_bgr(image: NDArrayU8) -> NDArrayU8:
    """Ensure an array is HxWx3 BGR uint8."""

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def bgr_to_rgb(image: NDArrayU8) -> NDArrayU8:
    """Convert BGR to RGB."""

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_jpeg(image: NDArrayU8, quality: int = 80) -> bytes:
    """Encode a BGR frame as JPEG bytes."""

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("Failed to encode JPEG")
    return buf.tobytes()


def pil_to_bgr(image: Image.Image) -> NDArrayU8:
    """Convert a PIL image to BGR uint8."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
