# Training

VisionGuard ships two trainers.

## 1. VGNet (from scratch, MIT)

A ~1.2M-parameter stride-16 detector with a real PyTorch loop: AdamW, cosine LR, BCE objectness, CE classification, CIoU box loss.

```bash
python scripts/generate_synthetic_data.py --frames 64
visionguard train --engine vgnet --images data/datasets/synthetic/images --epochs 10
visionguard export --vgnet models/checkpoints/vgnet.pt
```

Use this path in interviews to talk about assignment, losses, and export — not just `YOLO.train()`.

## 2. Ultralytics YOLOv8 (production)

Fine-tune `yolov8n/s/m` on a YOLO-format dataset described by `configs/training.yaml`.

```bash
visionguard train --engine yolo --config configs/training.yaml
visionguard export --config configs/training.yaml --format onnx
```

Dataset yaml (generated for synthetic data):

```yaml
path: data/datasets/synthetic
train: train.txt
val: val.txt
nc: 16
names:
  0: person
  # ...
```

## Metrics

`visionguard evaluate` reports precision, recall, F1, and interpolated mAP@0.50 using greedy IoU matching. The same helpers live in `visionguard.training.evaluate` for unit tests.

## Custom workplace dataset

1. Label PPE / fire / spill / vehicle classes in [Roboflow](https://roboflow.com) or CVAT (YOLO txt).
2. Point `configs/training.yaml` `data.yaml` at your export.
3. Align class names with `configs/hazards.yaml` `detector_name`.
4. Fine-tune YOLO, copy `best.pt` to `models/checkpoints/visionguard_yolov8n.pt`.
5. Set `VISIONGUARD_MODEL_BACKEND=ultralytics`.
