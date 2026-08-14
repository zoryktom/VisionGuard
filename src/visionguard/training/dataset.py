"""YOLO-format detection dataset + light augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from visionguard.config import resolve_path
from visionguard.exceptions import TrainingError
from visionguard.types import NDArrayU8


@dataclass(frozen=True, slots=True)
class BoxLabel:
    """Normalized YOLO label (cx, cy, w, h in 0-1) plus class id."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Convert to pixel XYXY."""

        bw, bh = self.w * width, self.h * height
        cx, cy = self.cx * width, self.cy * height
        return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def parse_yolo_label(path: Path) -> list[BoxLabel]:
    """Parse a YOLO .txt label file."""

    if not path.exists():
        return []
    labels: list[BoxLabel] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid, cx, cy, w, h = parts
        labels.append(BoxLabel(int(cid), float(cx), float(cy), float(w), float(h)))
    return labels


def letterbox(image: NDArrayU8, imgsz: int) -> tuple[NDArrayU8, float, tuple[int, int]]:
    """Resize with unchanged aspect ratio, pad to square."""

    h, w = image.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


class DetectionAugment:
    """CPU augmentations that keep YOLO boxes consistent (no extra deps)."""

    def __init__(self, hflip: float = 0.5, hsv: float = 0.4, seed: int | None = None) -> None:
        self.hflip = hflip
        self.hsv = hsv
        self.rng = np.random.default_rng(seed)

    def __call__(
        self, image: NDArrayU8, labels: list[BoxLabel]
    ) -> tuple[NDArrayU8, list[BoxLabel]]:
        """Apply a random subset of transforms."""

        out = image.copy()
        if self.rng.random() < self.hflip:
            out = cv2.flip(out, 1)
            labels = [BoxLabel(lb.class_id, 1.0 - lb.cx, lb.cy, lb.w, lb.h) for lb in labels]
        if self.hsv > 0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= float(self.rng.uniform(1.0 - self.hsv, 1.0 + self.hsv))
            hsv[:, :, 2] *= float(self.rng.uniform(1.0 - self.hsv, 1.0 + self.hsv))
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return out, labels


class YOLODataset(Dataset[tuple[torch.Tensor, list[BoxLabel]]]):
    """Folder of images/ + labels/ with matching stems."""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path | None = None,
        imgsz: int = 640,
        augment: DetectionAugment | None = None,
    ) -> None:
        self.images_dir = resolve_path(images_dir)
        self.labels_dir = (
            resolve_path(labels_dir) if labels_dir else self.images_dir.parent / "labels"
        )
        self.imgsz = imgsz
        self.augment = augment
        self.paths = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.paths:
            raise TrainingError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[BoxLabel]]:
        path = self.paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise TrainingError(f"Failed to read {path}")
        labels = parse_yolo_label(self.labels_dir / f"{path.stem}.txt")
        if self.augment is not None:
            image, labels = self.augment(image, labels)
        image, _, _ = letterbox(image, self.imgsz)
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return tensor, labels


def detection_collate(
    batch: list[tuple[torch.Tensor, list[BoxLabel]]],
) -> tuple[torch.Tensor, list[list[BoxLabel]]]:
    """Stack images; keep labels as a python list (variable length)."""

    images = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return images, labels
