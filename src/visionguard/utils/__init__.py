"""Utility package."""

from visionguard.utils.device import describe_device, resolve_device
from visionguard.utils.geometry import euclidean, iou_xyxy, point_in_polygon
from visionguard.utils.image import encode_jpeg, to_bgr
from visionguard.utils.time import Ema, now_s, stopwatch

__all__ = [
    "Ema",
    "describe_device",
    "encode_jpeg",
    "euclidean",
    "iou_xyxy",
    "now_s",
    "point_in_polygon",
    "resolve_device",
    "stopwatch",
    "to_bgr",
]
