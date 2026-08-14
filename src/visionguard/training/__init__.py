"""Training package."""

from visionguard.training.dataset import DetectionAugment, YOLODataset
from visionguard.training.evaluate import DetMetrics, evaluate_detections
from visionguard.training.trainer import export_weights, train_ultralytics, train_vgnet
from visionguard.training.vgnet import VGNet

__all__ = [
    "DetectionAugment",
    "DetMetrics",
    "VGNet",
    "YOLODataset",
    "evaluate_detections",
    "export_weights",
    "train_ultralytics",
    "train_vgnet",
]
