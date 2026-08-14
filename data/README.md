# Data

| Tree | Purpose |
|---|---|
| `data/datasets/synthetic/` | Generated warehouse frames + YOLO labels |
| `data/samples/` | Optional short clips you add locally |
| `data/schemas/` | Class list for exporters |

Generate the synthetic set:

```bash
python scripts/generate_synthetic_data.py --frames 48 --out data/datasets/synthetic
```

Bring your own data in YOLO format (`images/` + `labels/` + `data.yaml`) and point `configs/training.yaml` at it.

Public datasets that transfer well to this taxonomy:

- [PPE Detection (Roboflow)](https://universe.roboflow.com) — helmet / vest
- [Fire & Smoke](https://github.com/gaiasd/DFireDataset)
- COCO — person, vehicle, phone, knife (zero-shot demo)
