"""VGNet: a tiny single-stage detector you can train without Ultralytics.

The network is intentionally small (≈1.2M params) so it:
- trains on synthetic data on CPU in minutes
- exports cleanly to ONNX
- demonstrates a real PyTorch training loop (CE + CIoU)

It is *not* a YOLOv8 replacement. Production deployments should fine-tune
Ultralytics YOLO / RT-DETR via ``visionguard.training.trainer``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from visionguard.hazards.taxonomy import HazardTaxonomy
from visionguard.training.dataset import letterbox
from visionguard.types import Detection, HazardCategory, NDArrayU8, Severity


def conv_bn_act(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """3x3 conv + batchnorm + SiLU."""

    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class VGNet(nn.Module):
    """Stride-16 detector: backbone → 1x1 heads for cls / box / obj."""

    def __init__(self, num_classes: int = 16, width: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            conv_bn_act(3, width, 2),
            conv_bn_act(width, width * 2, 2),
            conv_bn_act(width * 2, width * 4, 2),
            conv_bn_act(width * 4, width * 4, 1),
            conv_bn_act(width * 4, width * 8, 2),
            conv_bn_act(width * 8, width * 8, 1),
        )
        hidden = width * 8
        self.cls = nn.Conv2d(hidden, num_classes, 1)
        self.obj = nn.Conv2d(hidden, 1, 1)
        self.box = nn.Conv2d(hidden, 4, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return dict with cls, obj, box feature maps."""

        feat = self.backbone(x)
        return {"cls": self.cls(feat), "obj": self.obj(feat), "box": self.box(feat)}


def ciou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complete IoU on XYXY tensors (N, 4)."""

    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + 1e-6
    iou = inter / union
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw * cw + ch * ch + 1e-6
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    pw, ph = (px2 - px1).clamp(min=1e-6), (py2 - py1).clamp(min=1e-6)
    tw, th = (tx2 - tx1).clamp(min=1e-6), (ty2 - ty1).clamp(min=1e-6)
    v = (4 / (np.pi**2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    return (1 - iou + rho2 / c2 + alpha * v).mean()


def vgnet_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[list[tuple[int, float, float, float, float]]],
    imgsz: int,
) -> dict[str, torch.Tensor]:
    """Assign each GT box to the nearest stride-16 cell and compute losses.

    ``targets[b]`` is a list of (class_id, cx, cy, w, h) in normalized 0-1.
    """

    cls_map, obj_map, box_map = outputs["cls"], outputs["obj"], outputs["box"]
    bsz, _, gh, gw = cls_map.shape
    device = cls_map.device
    obj_t = torch.zeros((bsz, 1, gh, gw), device=device)
    cls_t = torch.zeros((bsz, gh, gw), device=device, dtype=torch.long)
    box_t = torch.zeros((bsz, 4, gh, gw), device=device)
    pos_mask = torch.zeros((bsz, 1, gh, gw), device=device)

    for b, labels in enumerate(targets):
        for cid, cx, cy, w, h in labels:
            gx = min(gw - 1, max(0, int(cx * gw)))
            gy = min(gh - 1, max(0, int(cy * gh)))
            obj_t[b, 0, gy, gx] = 1.0
            cls_t[b, gy, gx] = int(cid)
            box_t[b, :, gy, gx] = torch.tensor(
                [
                    (cx - w / 2) * imgsz,
                    (cy - h / 2) * imgsz,
                    (cx + w / 2) * imgsz,
                    (cy + h / 2) * imgsz,
                ],
                device=device,
            )
            pos_mask[b, 0, gy, gx] = 1.0

    obj_loss = F.binary_cross_entropy_with_logits(obj_map, obj_t)
    cls_loss = F.cross_entropy(cls_map, cls_t, reduction="none")
    cls_loss = (cls_loss * pos_mask.squeeze(1)).sum() / pos_mask.sum().clamp(min=1.0)

    # Decode box predictions: sigmoid offsets in cell + size in image pixels.
    sy = torch.linspace(0, gh - 1, gh, device=device).view(1, 1, gh, 1)
    sx = torch.linspace(0, gw - 1, gw, device=device).view(1, 1, 1, gw)
    px = (sx + torch.sigmoid(box_map[:, 0:1])) / gw * imgsz
    py = (sy + torch.sigmoid(box_map[:, 1:2])) / gh * imgsz
    pw = torch.exp(box_map[:, 2:3].clamp(max=4)) * (imgsz / gw)
    ph = torch.exp(box_map[:, 3:4].clamp(max=4)) * (imgsz / gh)
    pred_xyxy = torch.cat([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], dim=1)
    if pos_mask.sum() > 0:
        mask = pos_mask.expand_as(pred_xyxy).bool()
        pred_pos = pred_xyxy.permute(0, 2, 3, 1)[mask.permute(0, 2, 3, 1)[:, :, :, 0]]
        tgt_pos = box_t.permute(0, 2, 3, 1)[mask.permute(0, 2, 3, 1)[:, :, :, 0]]
        box_loss = ciou_loss(pred_pos, tgt_pos)
    else:
        box_loss = pred_xyxy.sum() * 0.0

    total = obj_loss + cls_loss + box_loss
    return {"loss": total, "obj": obj_loss, "cls": cls_loss, "box": box_loss}


@torch.inference_mode()
def decode_detections(
    model: VGNet,
    frame: NDArrayU8,
    taxonomy: HazardTaxonomy,
    device: torch.device,
    confidence: float,
    imgsz: int = 320,
) -> list[Detection]:
    """Run VGNet on a BGR frame and return VisionGuard detections."""

    canvas, scale, (left, top) = letterbox(frame, imgsz)
    tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)
    out = model(tensor)
    cls_map = out["cls"][0].softmax(0)
    obj_map = out["obj"][0, 0].sigmoid()
    box_map = out["box"][0]
    _, gh, gw = cls_map.shape
    dets: list[Detection] = []
    obj_np = obj_map.cpu().numpy()
    ys, xs = np.where(obj_np > confidence)
    for gy, gx in zip(ys.tolist(), xs.tolist(), strict=False):
        obj = float(obj_np[gy, gx])
        cls_id = int(cls_map[:, gy, gx].argmax().item())
        cls_p = float(cls_map[cls_id, gy, gx].item())
        score = obj * cls_p
        if score < confidence:
            continue
        bx = box_map[:, gy, gx]
        px = (gx + torch.sigmoid(bx[0]).item()) / gw * imgsz
        py = (gy + torch.sigmoid(bx[1]).item()) / gh * imgsz
        pw = float(torch.exp(bx[2].clamp(max=4)).item()) * (imgsz / gw)
        ph = float(torch.exp(bx[3].clamp(max=4)).item()) * (imgsz / gh)
        x1 = (px - pw / 2 - left) / scale
        y1 = (py - ph / 2 - top) / scale
        x2 = (px + pw / 2 - left) / scale
        y2 = (py + ph / 2 - top) / scale
        spec = next((s for s in taxonomy.classes if s.class_id == cls_id), None)
        dets.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=spec.detector_name if spec else str(cls_id),
                confidence=score,
                category=spec.category if spec else HazardCategory.OBJECT,
                severity=spec.severity if spec else Severity.MEDIUM,
            )
        )
    return dets


def load_vgnet(path: Path, num_classes: int, device: torch.device) -> VGNet:
    """Load a VGNet checkpoint."""

    model = VGNet(num_classes=num_classes)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def export_onnx(model: VGNet, path: Path, imgsz: int = 320) -> Path:
    """Export VGNet to ONNX."""

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["images"],
        output_names=["cls", "obj", "box"],
        opset_version=17,
    )
    return path
