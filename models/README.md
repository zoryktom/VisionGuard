# Models

Checkpoints are **not** stored in git.

| Path | Role |
|---|---|
| `models/checkpoints/visionguard_yolov8n.pt` | Fine-tuned YOLO (copied after `visionguard train --engine yolo`) |
| `models/checkpoints/vgnet.pt` | From-scratch VGNet |
| `yolov8n.pt` | Official Ultralytics nano (auto-downloaded on first YOLO run) |

```bash
python scripts/download_weights.py --model yolov8n.pt
```

Set `VISIONGUARD_MODEL_WEIGHTS` or `inference.custom_weights` in YAML. The engine prefers a custom checkpoint when the file exists.
