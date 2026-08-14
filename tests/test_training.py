"""Training loop + metrics tests."""

from pathlib import Path

import cv2
import torch

from visionguard.capture import synthetic_frame
from visionguard.training.dataset import DetectionAugment, YOLODataset, parse_yolo_label
from visionguard.training.evaluate import evaluate_detections
from visionguard.training.trainer import train_vgnet
from visionguard.training.vgnet import VGNet, vgnet_loss


def test_parse_yolo_label(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("1 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    labels = parse_yolo_label(path)
    assert len(labels) == 1
    assert labels[0].class_id == 1
    x1, y1, x2, y2 = labels[0].xyxy(100, 100)
    assert x2 > x1 and y2 > y1


def test_augment_hflip_changes_cx() -> None:
    import numpy as np

    from visionguard.training.dataset import BoxLabel

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    aug = DetectionAugment(hflip=1.0, hsv=0.0, seed=0)
    labels = [BoxLabel(0, 0.25, 0.5, 0.1, 0.1)]
    _, out = aug(img, labels)
    assert abs(out[0].cx - 0.75) < 1e-6


def test_vgnet_forward_and_loss() -> None:
    model = VGNet(num_classes=4, width=8)
    x = torch.zeros(2, 3, 64, 64)
    out = model(x)
    assert out["cls"].shape[0] == 2
    losses = vgnet_loss(out, [[(0, 0.5, 0.5, 0.2, 0.2)], []], imgsz=64)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_evaluate_perfect_match() -> None:
    box = (10.0, 10.0, 40.0, 40.0)
    metrics = evaluate_detections([[(box, 0, 0.9)]], [[(box, 0)]])
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.map50 == 1.0


def test_train_vgnet_one_epoch(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    frame = synthetic_frame(128, 96, seed=0)
    cv2.imwrite(str(img_dir / "a.jpg"), frame)
    (lbl_dir / "a.txt").write_text("0 0.4 0.5 0.2 0.3\n", encoding="utf-8")
    ckpt = train_vgnet(
        images_dir=img_dir,
        labels_dir=lbl_dir,
        epochs=1,
        imgsz=64,
        batch_size=1,
        num_classes=4,
        device_pref="cpu",
        output=tmp_path / "vgnet.pt",
    )
    assert ckpt.exists()
    ds = YOLODataset(img_dir, lbl_dir, imgsz=64)
    assert len(ds) == 1
