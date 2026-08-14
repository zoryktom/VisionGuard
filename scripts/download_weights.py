"""Download official Ultralytics weights into models/checkpoints."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(model: str = typer.Option("yolov8n.pt")) -> None:
    """Fetch a pretrained checkpoint (requires network + ultralytics)."""

    from ultralytics import YOLO

    dest_dir = Path("models/checkpoints")
    dest_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(model)
    console.print(f"[green]Ready[/] {model}  classes={len(yolo.names)}")


if __name__ == "__main__":
    app()
