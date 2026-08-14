"""Detection metrics: precision, recall, F1, mAP@0.50."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visionguard.types import BBoxXYXY
from visionguard.utils.geometry import iou_xyxy


@dataclass(slots=True)
class DetMetrics:
    """Aggregate detection metrics."""

    precision: float
    recall: float
    f1: float
    map50: float
    tp: int
    fp: int
    fn: int

    def as_dict(self) -> dict[str, float]:
        """JSON-friendly payload."""

        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "map50": self.map50,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
        }


def _match(
    preds: list[tuple[BBoxXYXY, int, float]],
    gts: list[tuple[BBoxXYXY, int]],
    iou_thr: float,
) -> tuple[int, int, int]:
    used = set()
    tp = fp = 0
    ordered = sorted(preds, key=lambda p: p[2], reverse=True)
    for box, cls, _ in ordered:
        hit = None
        best = 0.0
        for i, (gt_box, gt_cls) in enumerate(gts):
            if i in used or gt_cls != cls:
                continue
            iou = iou_xyxy(box, gt_box)
            if iou >= iou_thr and iou > best:
                best = iou
                hit = i
        if hit is None:
            fp += 1
        else:
            used.add(hit)
            tp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn


def average_precision(
    preds: list[tuple[BBoxXYXY, int, float]],
    gts: list[tuple[BBoxXYXY, int]],
    iou_thr: float = 0.5,
) -> float:
    """11-point interpolated AP for a single image list (micro, all classes)."""

    if not gts:
        return 1.0 if not preds else 0.0
    scores = []
    matched = set()
    for box, cls, _score in sorted(preds, key=lambda p: p[2], reverse=True):
        hit = None
        best = 0.0
        for i, (gt_box, gt_cls) in enumerate(gts):
            if i in matched or gt_cls != cls:
                continue
            iou = iou_xyxy(box, gt_box)
            if iou >= iou_thr and iou > best:
                best = iou
                hit = i
        scores.append(1.0 if hit is not None else 0.0)
        if hit is not None:
            matched.add(hit)
    if not scores:
        return 0.0
    tps = np.cumsum(scores)
    fps = np.cumsum(1.0 - np.array(scores))
    recalls = tps / max(len(gts), 1)
    precisions = tps / np.maximum(tps + fps, 1e-9)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        ap += float(precisions[mask].max()) if mask.any() else 0.0
    return ap / 11.0


def evaluate_detections(
    predictions: list[list[tuple[BBoxXYXY, int, float]]],
    ground_truth: list[list[tuple[BBoxXYXY, int]]],
    iou_thr: float = 0.5,
) -> DetMetrics:
    """Micro-averaged P/R/F1 plus interpolated mAP@iou."""

    tp = fp = fn = 0
    aps: list[float] = []
    for preds, gts in zip(predictions, ground_truth, strict=False):
        t, f, n = _match(preds, gts, iou_thr)
        tp += t
        fp += f
        fn += n
        aps.append(average_precision(preds, gts, iou_thr))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return DetMetrics(precision, recall, f1, float(np.mean(aps) if aps else 0.0), tp, fp, fn)
