"""Generate a tiny YOLO-format synthetic warehouse dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import typer
from rich.console import Console

from visionguard.capture import synthetic_frame
from visionguard.config import resolve_path
from visionguard.hazards.taxonomy import default_taxonomy
from visionguard.types import NDArrayU8

app = typer.Typer(add_completion=False)
console = Console()


def _write_yolo(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _boxes_from_dummy(frame: NDArrayU8, taxonomy) -> list[tuple[int, float, float, float, float]]:
    from visionguard.inference import DummyDetector

    det = DummyDetector(taxonomy)
    h, w = frame.shape[:2]
    out: list[tuple[int, float, float, float, float]] = []
    for d in det.predict(frame):
        x1, y1, x2, y2 = d.bbox
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
        out.append((d.class_id, cx, cy, bw, bh))
    return out


@app.command()
def main(
    frames: int = typer.Option(48, "--frames"),
    out: Path = typer.Option(Path("data/datasets/synthetic")),
    width: int = typer.Option(640),
    height: int = typer.Option(384),
    val_ratio: float = typer.Option(0.2),
) -> None:
    """Write images, labels, and a Ultralytics data.yaml."""

    taxonomy = default_taxonomy()
    root = resolve_path(out)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    names = taxonomy.names()
    n_val = max(1, int(frames * val_ratio))
    train_ids: list[str] = []
    val_ids: list[str] = []

    for i in range(frames):
        frame = synthetic_frame(width, height, seed=i)
        name = f"frame_{i:04d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
        _write_yolo(lbl_dir / f"{name}.txt", _boxes_from_dummy(frame, taxonomy))
        (val_ids if i < n_val else train_ids).append(f"images/{name}.jpg")

    (root / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (root / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    yaml = [
        f"path: {root}",
        "train: train.txt",
        "val: val.txt",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        yaml.append(f"  {i}: {name}")
    (root / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote[/] {frames} frames → {root}")


if __name__ == "__main__":
    app()
