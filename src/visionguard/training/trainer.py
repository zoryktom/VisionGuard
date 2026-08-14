"""Training entrypoints: Ultralytics YOLO (production) and VGNet (from-scratch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from visionguard.config import load_yaml, resolve_path
from visionguard.exceptions import TrainingError
from visionguard.logging_utils import get_logger
from visionguard.training.dataset import DetectionAugment, YOLODataset, detection_collate
from visionguard.training.vgnet import VGNet, export_onnx, vgnet_loss
from visionguard.utils.device import describe_device, resolve_device

logger = get_logger(__name__)


def _training_cfg(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def train_ultralytics(config_path: str | Path = "configs/training.yaml") -> Path:
    """Fine-tune YOLOv8 via Ultralytics (recommended production path)."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise TrainingError("ultralytics is required for this trainer") from exc

    cfg = _training_cfg(config_path)
    data_yaml = resolve_path(cfg["data"]["yaml"])
    if not data_yaml.exists():
        raise TrainingError(f"Dataset yaml not found: {data_yaml}")
    model_name = cfg["model"].get("pretrained") or f"{cfg['model']['architecture']}.pt"
    device = resolve_device(cfg["train"].get("device", "auto"))
    model = YOLO(model_name)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": int(cfg["train"]["epochs"]),
        "imgsz": int(cfg["data"]["imgsz"]),
        "batch": int(cfg["train"]["batch_size"]),
        "lr0": float(cfg["train"]["lr0"]),
        "lrf": float(cfg["train"]["lrf"]),
        "weight_decay": float(cfg["train"]["weight_decay"]),
        "patience": int(cfg["train"]["patience"]),
        "amp": bool(cfg["train"]["amp"]),
        "device": str(device),
        "project": str(resolve_path(cfg["experiment"]["output_dir"])),
        "name": cfg["experiment"]["name"],
        "exist_ok": True,
        "seed": int(cfg["experiment"].get("seed", 42)),
        "mosaic": float(cfg["train"].get("mosaic", 1.0)),
        "mixup": float(cfg["train"].get("mixup", 0.0)),
        "fliplr": float(cfg["train"].get("fliplr", 0.5)),
        "degrees": float(cfg["train"].get("degrees", 0.0)),
        "hsv_h": float(cfg["train"].get("hsv_h", 0.015)),
        "hsv_s": float(cfg["train"].get("hsv_s", 0.7)),
        "hsv_v": float(cfg["train"].get("hsv_v", 0.4)),
        "workers": int(cfg["train"].get("workers", 2)),
    }
    logger.info("Ultralytics train device=%s data=%s", describe_device(device), data_yaml)
    model.train(**train_kwargs)
    best = (
        Path(model.trainer.best)
        if getattr(model, "trainer", None)
        else resolve_path(
            f"{cfg['experiment']['output_dir']}/{cfg['experiment']['name']}/weights/best.pt"
        )
    )
    return Path(best)


def train_vgnet(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    epochs: int = 5,
    imgsz: int = 320,
    batch_size: int = 8,
    lr: float = 1e-3,
    num_classes: int = 16,
    device_pref: str = "auto",
    output: str | Path = "models/checkpoints/vgnet.pt",
) -> Path:
    """Train VGNet on a YOLO-format folder (works on CPU)."""

    device = resolve_device(device_pref)
    dataset = YOLODataset(images_dir, labels_dir, imgsz=imgsz, augment=DetectionAugment(seed=0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=0,
    )
    model = VGNet(num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    model.train()
    for epoch in range(epochs):
        running = 0.0
        steps = 0
        for images, labels in loader:
            images = images.to(device)
            targets = [
                [(lb.class_id, lb.cx, lb.cy, lb.w, lb.h) for lb in sample] for sample in labels
            ]
            optim.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = vgnet_loss(outputs, targets, imgsz)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            running += float(losses["loss"].item())
            steps += 1
        scheduler.step()
        mean = running / max(1, steps)
        logger.info(
            "vgnet epoch=%d/%d loss=%.4f device=%s",
            epoch + 1,
            epochs,
            mean,
            describe_device(device),
        )
    out = resolve_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": num_classes, "imgsz": imgsz}, out)
    return out


def export_weights(
    config_path: str | Path = "configs/training.yaml",
    format: str = "onnx",
    vgnet_checkpoint: str | Path | None = None,
) -> list[Path]:
    """Export Ultralytics YOLO and/or VGNet."""

    exported: list[Path] = []
    if vgnet_checkpoint:
        ckpt = resolve_path(vgnet_checkpoint)
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = VGNet(num_classes=int(payload.get("num_classes", 16)))
        model.load_state_dict(payload["state_dict"])
        dest = ckpt.with_suffix(".onnx")
        exported.append(export_onnx(model, dest, imgsz=int(payload.get("imgsz", 320))))
        return exported

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise TrainingError("ultralytics is required to export YOLO weights") from exc

    cfg = _training_cfg(config_path)
    weights = resolve_path(
        f"{cfg['experiment']['output_dir']}/{cfg['experiment']['name']}/weights/best.pt"
    )
    if not weights.exists():
        raise TrainingError(f"No checkpoint at {weights}; train first")
    model = YOLO(str(weights))
    for fmt in cfg.get("export", {}).get("formats", [format]):
        model.export(
            format=fmt,
            imgsz=int(cfg["data"]["imgsz"]),
            opset=int(cfg.get("export", {}).get("opset", 17)),
            simplify=bool(cfg.get("export", {}).get("simplify", True)),
            dynamic=bool(cfg.get("export", {}).get("dynamic", False)),
            half=bool(cfg.get("export", {}).get("half", False)),
        )
        exported.append(weights.with_suffix(f".{fmt if fmt != 'torchscript' else 'torchscript'}"))
    return exported
